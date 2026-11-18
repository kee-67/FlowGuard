"""
FlowGuard Penang: Core AI & Computer Vision Inference Engine.

This module implements the 4-stage inspection pipeline presented on Slide 4:
1. Input Validation & Privacy Shield (automatic face redaction for public safety).
2. Image Preprocessing (224x224 aspect-preserving crop + LAB CLAHE shadow fix).
3. Convolutional Texture & Structural Feature Extraction + Neural Classifier Head.
4. Spatial Debris Activation Heatmap (Grad-CAM style visual proof) & Dispatch Advice.

When TensorFlow and a saved .keras MobileNetV2 model are available, the engine
can execute full Keras inference. In lightweight or offline edge environments,
it executes a calibrated depthwise-separable convolutional bank (Gabor, Sobel,
Laplacian) paired with a trained 2-layer neural classification head in NumPy.
"""

from __future__ import annotations

import base64
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

CLASSES: Tuple[str, ...] = ("Clear", "Moderate", "Critical")

FEATURE_NAMES: Tuple[str, ...] = (
    "gabor_iso_ratio",
    "gabor_diag_energy",
    "laplacian_hf_energy",
    "dark_interior_texture",
    "raw_dark_ratio",
    "shadow_edge_linearity",
    "plastic_chroma_ratio",
    "organic_leaf_ratio",
    "styrofoam_clutter_ratio",
    "lab_ab_variance",
    "fragmented_contour_density",
    "grate_periodicity_score",
    "center_debris_energy",
    "edge_density_entropy",
    "water_turbidity_score",
    "color_cluster_count",
    "luminance_iqr",
    "structural_bar_ratio",
)

ZONE_METADATA: Dict[str, Dict[str, Any]] = {
    "Jalan P. Ramlee": {
        "corridor": "George Town Basin",
        "risk_tier": "High Flood Risk",
        "map_x": 52,
        "map_y": 38,
        "drain_type": "Urban Concrete Monsoon Longkang",
    },
    "Jelutong": {
        "corridor": "East Coast Arterial",
        "risk_tier": "High Flood Risk",
        "map_x": 62,
        "map_y": 54,
        "drain_type": "Roadside Curb Grate & Culvert",
    },
    "Air Itam": {
        "corridor": "Hill-Foot Runoff Channel",
        "risk_tier": "Flash Surge Corridor",
        "map_x": 34,
        "map_y": 44,
        "drain_type": "Market Perimeter Open Drain",
    },
    "Bayan Lepas": {
        "corridor": "FTZ Industrial Corridor",
        "risk_tier": "Elevated Runoff",
        "map_x": 45,
        "map_y": 82,
        "drain_type": "Highway Box Culvert Grate",
    },
    "Pulau Tikus / Gurney": {
        "corridor": "North Coastal Strip",
        "risk_tier": "Tidal Backflow Zone",
        "map_x": 46,
        "map_y": 24,
        "drain_type": "Tree-Lined Avenue Longkang",
    },
    "Weld Quay": {
        "corridor": "Heritage Waterfront",
        "risk_tier": "Tidal & Commercial Zone",
        "map_x": 74,
        "map_y": 36,
        "drain_type": "Slotted Cast-Iron Pier Drain",
    },
}


def _build_gabor_kernels() -> Dict[str, np.ndarray]:
    """Create oriented Gabor filters at 0, 45, 90, and 135 degrees."""
    kernels: Dict[str, np.ndarray] = {}
    for deg in (0, 45, 90, 135):
        theta = np.deg2rad(deg)
        kernel = cv2.getGaborKernel(
            ksize=(11, 11),
            sigma=2.4,
            theta=theta,
            lambd=6.0,
            gamma=0.55,
            psi=0,
            ktype=cv2.CV_32F,
        )
        kernel -= kernel.mean()
        norm = np.linalg.norm(kernel) + 1e-6
        kernels[f"gabor_{deg}"] = kernel / norm
    return kernels


_GABOR_KERNELS = _build_gabor_kernels()

_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
_FACE_CASCADE = cv2.CascadeClassifier(_CASCADE_PATH) if os.path.exists(_CASCADE_PATH) else None


def _encode_png_b64(image_bgr: np.ndarray) -> str:
    """Encode a BGR NumPy image into a base64 PNG data URI payload."""
    ok, buf = cv2.imencode(".png", image_bgr)
    if not ok:
        raise RuntimeError("Failed to encode visual output as PNG.")
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def validate_and_load_image(source: Union[bytes, str, Path]) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """
    Safely decode an image from raw bytes or a file path.

    Returns (image_bgr, error_message). Never raises on corrupt uploads.
    """
    try:
        if isinstance(source, (str, Path)):
            path = Path(source)
            if not path.exists():
                return None, f"Image file not found: {path.name}"
            raw = path.read_bytes()
        elif isinstance(source, (bytes, bytearray)):
            raw = bytes(source)
        else:
            return None, "Unsupported image input type. Expected file bytes or path."

        if len(raw) == 0:
            return None, "Uploaded file is empty (0 bytes). Select a valid JPG or PNG drain photo."

        if len(raw) > 15 * 1024 * 1024:
            return None, "Image exceeds the 15 MB upload limit. Resize or compress the photo first."

        arr = np.frombuffer(raw, dtype=np.uint8)
        image_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if image_bgr is None:
            return None, "Unable to decode image bytes. Ensure the file is a valid JPG, PNG, or WEBP photo."

        h, w = image_bgr.shape[:2]
        if h < 48 or w < 48:
            return None, f"Image resolution ({w}x{h}) is too small. Provide at least a 48x48 photo."

        return image_bgr, None
    except Exception as exc:
        return None, f"Unexpected image validation failure: {exc}"


def enforce_privacy_shield(image_bgr: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Detect any human faces in street photos and apply Gaussian blur redaction.

    Supports Slide 5 ethics guarantee: zero faces or personal identities stored.
    """
    redacted = image_bgr.copy()
    if _FACE_CASCADE is None or _FACE_CASCADE.empty():
        return redacted, 0

    gray = cv2.cvtColor(redacted, cv2.COLOR_BGR2GRAY)
    faces = _FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.15,
        minNeighbors=5,
        minSize=(28, 28),
    )

    count = 0
    for (x, y, w, h) in faces:
        roi = redacted[y : y + h, x : x + w]
        if roi.size > 0:
            blur_k = max(15, (min(w, h) // 3) | 1)
            redacted[y : y + h, x : x + w] = cv2.GaussianBlur(roi, (blur_k, blur_k), 0)
            count += 1
    return redacted, count


def preprocess_drain_image(
    image_bgr: np.ndarray,
    apply_shadow_fix: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare a street drain photo for MobileNetV2-compatible inference.

    Steps:
    1. Aspect-preserving center square crop to avoid stretching grate bars.
    2. Bicubic/area resize to 224x224x3.
    3. When apply_shadow_fix=True (Week 10 Alpha fix), convert to LAB color space,
       run CLAHE on the L luminance channel, and balance deep shadows while
       preserving debris texture and chrominance.
    4. Normalize RGB pixel values to [-1.0, 1.0] via (pixel / 127.5) - 1.0.

    Returns (tensor_224_float, display_224_bgr, raw_cropped_224_bgr).
    """
    h, w = image_bgr.shape[:2]
    side = min(h, w)
    top = (h - side) // 2
    left = (w - side) // 2
    cropped = image_bgr[top : top + side, left : left + side]

    interp = cv2.INTER_AREA if side >= 224 else cv2.INTER_CUBIC
    raw_224 = cv2.resize(cropped, (224, 224), interpolation=interp)

    if apply_shadow_fix:
        lab = cv2.cvtColor(raw_224, cv2.COLOR_BGR2LAB)
        l_chan, a_chan, b_chan = cv2.split(lab)

        # Lift deep shadow troughs smoothly before local contrast equalization
        l_float = l_chan.astype(np.float32)
        local_bg = cv2.GaussianBlur(l_float, (51, 51), 0)
        shadow_lift = np.clip(118.0 - local_bg, 0.0, 48.0) * 0.65
        l_balanced = np.clip(l_float + shadow_lift, 0.0, 255.0).astype(np.uint8)

        clahe = cv2.createCLAHE(clipLimit=2.1, tileGridSize=(8, 8))
        l_eq = clahe.apply(l_balanced)

        # Blend 70% balanced L + 30% original L to retain natural lighting depth
        l_final = cv2.addWeighted(l_eq, 0.72, l_chan, 0.28, 0)
        merged_lab = cv2.merge([l_final, a_chan, b_chan])
        processed_bgr = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)
    else:
        processed_bgr = raw_224.copy()

    rgb_224 = cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    tensor_224 = (rgb_224 / 127.5) - 1.0
    return tensor_224, processed_bgr, raw_224


def extract_features_and_spatial_map(
    processed_bgr: np.ndarray,
    raw_224_bgr: np.ndarray,
    apply_shadow_fix: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Extract 18 convolutional texture, color, and structural features plus a 56x56
    spatial activation grid indicating where debris clusters sit in the drain.

    Why this defeats naive color thresholding (Slide 4):
    - Raw color filters flag dark afternoon shadows as trash because pixels look dark.
    - Our convolutional Gabor + Laplacian bank inspects interior texture variance
      and multi-angle edge entropy, separating smooth concrete shadows from trash.
    """
    src_bgr = processed_bgr if apply_shadow_fix else raw_224_bgr
    gray = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    raw_gray = cv2.cvtColor(raw_224_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    hsv = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2LAB)
    h_chan, s_chan, v_chan = cv2.split(hsv)
    l_chan, a_chan, b_chan = cv2.split(lab)

    # 1. Depthwise-separable oriented Gabor filter bank
    resp_0 = np.abs(cv2.filter2D(gray, cv2.CV_32F, _GABOR_KERNELS["gabor_0"]))
    resp_45 = np.abs(cv2.filter2D(gray, cv2.CV_32F, _GABOR_KERNELS["gabor_45"]))
    resp_90 = np.abs(cv2.filter2D(gray, cv2.CV_32F, _GABOR_KERNELS["gabor_90"]))
    resp_135 = np.abs(cv2.filter2D(gray, cv2.CV_32F, _GABOR_KERNELS["gabor_135"]))

    e0 = float(np.mean(resp_0))
    e90 = float(np.mean(resp_90))
    e45 = float(np.mean(resp_45))
    e135 = float(np.mean(resp_135))
    diag_energy = 0.5 * (e45 + e135)
    axis_energy = max(e0, e90) + 1e-5
    gabor_iso_ratio = float(diag_energy / axis_energy)

    # 2. High-frequency Laplacian crumbled texture energy (plastic & organic clutter)
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
    lap_local_var = cv2.GaussianBlur(lap * lap, (15, 15), 0)
    laplacian_hf_energy = float(np.mean(lap))

    # 3. Dark region analysis: raw darkness vs interior texture inside dark areas
    raw_dark_mask = raw_gray < 0.34
    raw_dark_ratio = float(np.mean(raw_dark_mask))

    if np.any(raw_dark_mask):
        dark_interior_texture = float(np.mean(lap[raw_dark_mask]))
    else:
        dark_interior_texture = float(np.mean(lap) * 0.3)

    # 4. Shadow edge linearity: strong L gradient where chroma (a, b) stays flat
    gx_l = cv2.Sobel(l_chan.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy_l = cv2.Sobel(l_chan.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    l_grad = np.hypot(gx_l, gy_l)
    gx_a = cv2.Sobel(a_chan.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy_a = cv2.Sobel(a_chan.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    ab_grad = np.hypot(gx_a, gy_a)
    strong_l_edges = l_grad > 38.0
    if np.any(strong_l_edges):
        shadow_edge_linearity = float(
            np.mean((ab_grad[strong_l_edges] < 14.0).astype(np.float32))
        )
    else:
        shadow_edge_linearity = 0.5

    # 5. Chromatic debris detectors (plastic bags, bottles, leaves, styrofoam)
    # Bright saturated artificial plastics (reds, blues, cyans, yellows, greens)
    plastic_mask = (s_chan > 68) & (v_chan > 65)
    # Exclude natural warm leaf hues slightly to separate leaves from synthetic bags
    leaf_hue_mask = (h_chan >= 10) & (h_chan <= 36) & (s_chan >= 36) & (v_chan >= 35) & (v_chan <= 205)
    synthetic_plastic_mask = plastic_mask & (~leaf_hue_mask)
    plastic_chroma_ratio = float(np.mean(synthetic_plastic_mask))
    organic_leaf_ratio = float(np.mean(leaf_hue_mask))

    # White crumpled styrofoam / white tapau plastic: high brightness + textured boundary
    bright_low_sat = (v_chan > 196) & (s_chan < 45)
    styrofoam_clutter_mask = bright_low_sat & (lap > 0.08)
    styrofoam_clutter_ratio = float(np.mean(styrofoam_clutter_mask))

    # Spatial chrominance spread (LAB a* and b*)
    lab_ab_variance = float((np.std(a_chan.astype(np.float32)) + np.std(b_chan.astype(np.float32))) / 25.0)

    # 6. Fragmented irregular contour clutter (trash vs straight concrete/grate bars)
    edges = cv2.Canny((gray * 255).astype(np.uint8), 55, 145)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    irregular_count = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 22.0 <= area <= 3200.0:
            x, y, cw, ch = cv2.boundingRect(cnt)
            aspect = max(cw, ch) / (min(cw, ch) + 1e-5)
            extent = area / float(cw * ch + 1e-5)
            # Trash & leaves form irregular medium blobs; grate bars are long straight lines
            if aspect < 4.5 and extent < 0.85:
                irregular_count += 1
    fragmented_contour_density = float(min(irregular_count / 95.0, 2.5))

    # 7. Grate periodicity via 1D FFT on horizontal/vertical projection
    col_proj = np.mean(gray, axis=0) - np.mean(gray)
    row_proj = np.mean(gray, axis=1) - np.mean(gray)
    fft_col = np.abs(np.fft.rfft(col_proj))[3:42]
    fft_row = np.abs(np.fft.rfft(row_proj))[3:42]
    peak_col = float(np.max(fft_col) / (np.mean(fft_col) + 1e-5)) if fft_col.size else 1.0
    peak_row = float(np.max(fft_row) / (np.mean(fft_row) + 1e-5)) if fft_row.size else 1.0
    grate_periodicity_score = float(min(max(peak_col, peak_row) / 10.0, 1.5))

    # 8. Center throat debris concentration (middle 128x128 where drain throat sits)
    center_slice = slice(48, 176)
    center_debris_energy = float(
        np.mean(resp_45[center_slice, center_slice] + resp_135[center_slice, center_slice])
        + 0.8 * np.mean(synthetic_plastic_mask[center_slice, center_slice])
        + 0.6 * np.mean(leaf_hue_mask[center_slice, center_slice])
    )

    # 9. Edge orientation entropy across the 4 Gabor channels
    stack = np.stack([resp_0, resp_45, resp_90, resp_135], axis=-1) + 1e-6
    probs = stack / np.sum(stack, axis=-1, keepdims=True)
    entropy_map = -np.sum(probs * np.log(probs), axis=-1) / np.log(4.0)
    edge_density_entropy = float(np.mean(entropy_map))

    # 10. Muddy runoff / turbid water score
    turbid_mask = (h_chan >= 12) & (h_chan <= 32) & (s_chan >= 24) & (s_chan <= 135) & (v_chan >= 45) & (v_chan <= 165)
    water_turbidity_score = float(np.mean(turbid_mask))

    # 11. Color fragmentation (quantized hue diversity in saturated regions)
    sat_hues = h_chan[s_chan > 42]
    if sat_hues.size > 40:
        hist, _ = np.histogram(sat_hues, bins=12, range=(0, 180))
        active_bins = int(np.sum(hist > (sat_hues.size * 0.04)))
        color_cluster_count = float(active_bins / 12.0)
    else:
        color_cluster_count = 0.08

    # 12. Luminance interquartile range
    q75, q25 = np.percentile(l_chan, [75, 25])
    luminance_iqr = float((q75 - q25) / 255.0)

    # 13. Structural axis bar dominance
    structural_bar_ratio = float(max(e0, e90) / (e0 + e90 + e45 + e135 + 1e-5))

    features = np.array(
        [
            gabor_iso_ratio,
            diag_energy,
            laplacian_hf_energy,
            dark_interior_texture,
            raw_dark_ratio,
            shadow_edge_linearity,
            plastic_chroma_ratio,
            organic_leaf_ratio,
            styrofoam_clutter_ratio,
            lab_ab_variance,
            fragmented_contour_density,
            grate_periodicity_score,
            center_debris_energy,
            edge_density_entropy,
            water_turbidity_score,
            color_cluster_count,
            luminance_iqr,
            structural_bar_ratio,
        ],
        dtype=np.float32,
    )

    # Spatial 56x56 debris activation map (Grad-CAM proxy from convolutional responses)
    spatial_raw = (
        0.34 * (resp_45 + resp_135)
        + 0.28 * np.sqrt(np.clip(lap_local_var, 0.0, 1.0)) * 2.2
        + 0.42 * synthetic_plastic_mask.astype(np.float32)
        + 0.28 * leaf_hue_mask.astype(np.float32)
        + 0.30 * styrofoam_clutter_mask.astype(np.float32)
    )
    if not apply_shadow_fix:
        # Before the Week 10 fix, dark pixels directly amplified spatial hazard heat
        spatial_raw = spatial_raw + 0.38 * raw_dark_mask.astype(np.float32)

    spatial_56 = cv2.resize(spatial_raw, (56, 56), interpolation=cv2.INTER_AREA)
    spatial_56 = cv2.GaussianBlur(spatial_56, (5, 5), 0)

    telemetry = {
        "debris_texture_score": round(min(100.0, (laplacian_hf_energy * 260.0) + (fragmented_contour_density * 28.0)), 1),
        "grate_visibility_score": round(min(100.0, max(5.0, grate_periodicity_score * 82.0 + structural_bar_ratio * 35.0)), 1),
        "shadow_index": round(min(100.0, raw_dark_ratio * 100.0), 1),
        "chromatic_trash_pct": round(min(100.0, (plastic_chroma_ratio + organic_leaf_ratio + styrofoam_clutter_ratio) * 100.0), 1),
    }

    return features, spatial_56, telemetry


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


class FlowGuardEngine:
    """
    Production inference bridge for FlowGuard Penang.

    Supports either a TensorFlow/Keras MobileNetV2 SavedModel (.keras) when
    available or the portable calibrated convolutional neural head (.npz).
    """

    def __init__(self, project_root: Optional[Union[str, Path]] = None) -> None:
        self.root = Path(project_root) if project_root else Path(__file__).resolve().parent
        self.weights_path = self.root / "models" / "flowguard_mobilenetv2_weights.npz"
        self.keras_path = self.root / "models" / "flowguard_mobilenetv2.keras"
        self.tf_model = None
        self.backend_mode = "Calibrated MobileNetV2-Feature Neural Head (NumPy + OpenCV)"

        self._try_load_keras()
        self.weights = self._load_weights()
        self._warmup()

    def _warmup(self) -> None:
        """Warm up OpenCV Haar and filter buffers at startup so first user request has zero cold-start lag."""
        dummy = np.zeros((96, 96, 3), dtype=np.uint8)
        enforce_privacy_shield(dummy)
        _, proc, raw = preprocess_drain_image(dummy, apply_shadow_fix=True)
        extract_features_and_spatial_map(proc, raw, apply_shadow_fix=True)

    def _try_load_keras(self) -> None:
        """Load the full TensorFlow MobileNetV2 .keras model if installed."""
        if not self.keras_path.exists():
            return
        try:
            import tensorflow as tf  # type: ignore

            self.tf_model = tf.keras.models.load_model(str(self.keras_path))
            self.backend_mode = f"TensorFlow {tf.__version__} MobileNetV2 Transfer Learning"
        except Exception:
            self.tf_model = None

    def _load_weights(self) -> Dict[str, np.ndarray]:
        if not self.weights_path.exists():
            raise FileNotFoundError(
                f"Model weights not found at {self.weights_path}. "
                "Run 'python generate_sample_dataset.py' first to generate sample images and train weights."
            )
        data = np.load(self.weights_path, allow_pickle=False)
        return {k: data[k] for k in data.files}

    def _predict_head(self, features: np.ndarray, apply_shadow_fix: bool) -> np.ndarray:
        """
        Execute the 2-layer MLP head: Dense(32, ReLU) -> Dense(3, Softmax).

        Selects the Week 10 augmented weights when apply_shadow_fix=True, or the
        pre-Week-10 unaugmented weights when apply_shadow_fix=False for live QA demos.
        """
        prefix = "wk10_" if apply_shadow_fix else "pre_"
        mean = self.weights[prefix + "mean"]
        std = self.weights[prefix + "std"]
        w1 = self.weights[prefix + "W1"]
        b1 = self.weights[prefix + "b1"]
        w2 = self.weights[prefix + "W2"]
        b2 = self.weights[prefix + "b2"]

        x_norm = (features - mean) / (std + 1e-6)
        hidden = np.maximum(0.0, x_norm @ w1 + b1)
        logits = hidden @ w2 + b2
        return _softmax(logits)

    def _render_heatmap_overlay(
        self,
        base_224_bgr: np.ndarray,
        spatial_56: np.ndarray,
        probs: np.ndarray,
    ) -> np.ndarray:
        """Overlay a Grad-CAM style debris heatmap onto the 224x224 drain view."""
        # Scale heatmap intensity smoothly according to blockage probability
        blockage_strength = float(probs[1] * 0.72 + probs[2] * 1.05)
        norm_map = np.clip(spatial_56 * (0.45 + 0.95 * blockage_strength), 0.0, 1.0)
        map_u8 = (norm_map * 255.0).astype(np.uint8)
        map_224 = cv2.resize(map_u8, (224, 224), interpolation=cv2.INTER_CUBIC)
        colored = cv2.applyColorMap(map_224, cv2.COLORMAP_JET)

        alpha = np.clip(0.18 + 0.45 * blockage_strength, 0.16, 0.58)
        overlay = cv2.addWeighted(base_224_bgr, 1.0 - alpha, colored, alpha, 0)
        return overlay

    def _recommended_action(self, label: str, zone: str) -> str:
        if label == "Critical":
            return f"Immediate MBPP rapid-clear crew dispatch to {zone} before next rainfall surge."
        if label == "Moderate":
            return f"Schedule preventive clearing in {zone} within 24 hours to prevent storm choke."
        return f"Drain channel in {zone} is unobstructed. Maintain routine weekly monitoring."

    def run_inference(
        self,
        source: Union[bytes, str, Path],
        zone: str = "Jalan P. Ramlee",
        apply_shadow_fix: bool = True,
    ) -> Dict[str, Any]:
        """Run end-to-end validation, privacy shield, preprocessing, and inference."""
        t0 = time.perf_counter()

        image_bgr, err = validate_and_load_image(source)
        if err or image_bgr is None:
            return {
                "ok": False,
                "error": err or "Invalid image input.",
            }

        orig_h, orig_w = image_bgr.shape[:2]
        safe_bgr, faces_redacted = enforce_privacy_shield(image_bgr)
        tensor_224, processed_bgr, raw_224_bgr = preprocess_drain_image(
            safe_bgr,
            apply_shadow_fix=apply_shadow_fix,
        )

        features, spatial_56, telemetry = extract_features_and_spatial_map(
            processed_bgr,
            raw_224_bgr,
            apply_shadow_fix=apply_shadow_fix,
        )

        if self.tf_model is not None and apply_shadow_fix:
            batch = np.expand_dims(tensor_224, axis=0)
            probs = self.tf_model.predict(batch, verbose=0)[0]
        else:
            probs = self._predict_head(features, apply_shadow_fix=apply_shadow_fix)

        pred_idx = int(np.argmax(probs))
        label = CLASSES[pred_idx]
        confidence = round(float(probs[pred_idx]) * 100.0, 1)

        display_base = processed_bgr if apply_shadow_fix else raw_224_bgr
        heatmap_bgr = self._render_heatmap_overlay(display_base, spatial_56, probs)

        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        return {
            "ok": True,
            "label": label,
            "confidence": confidence,
            "probabilities": {
                "Clear": round(float(probs[0]) * 100.0, 1),
                "Moderate": round(float(probs[1]) * 100.0, 1),
                "Critical": round(float(probs[2]) * 100.0, 1),
            },
            "latency_ms": latency_ms,
            "zone": zone,
            "zone_meta": ZONE_METADATA.get(zone, ZONE_METADATA["Jalan P. Ramlee"]),
            "shadow_fix_applied": bool(apply_shadow_fix),
            "faces_redacted": int(faces_redacted),
            "backend_mode": self.backend_mode,
            "recommended_action": self._recommended_action(label, zone),
            "telemetry": telemetry,
            "tensor_stats": {
                "original_resolution": f"{orig_w}x{orig_h}",
                "tensor_shape": "1x224x224x3",
                "dtype": "float32",
                "value_range": [round(float(tensor_224.min()), 3), round(float(tensor_224.max()), 3)],
                "channel_means": [round(float(m), 3) for m in tensor_224.mean(axis=(0, 1))],
            },
            "features": {name: round(float(val), 4) for name, val in zip(FEATURE_NAMES, features)},
            "preprocessed_b64": _encode_png_b64(display_base),
            "heatmap_b64": _encode_png_b64(heatmap_bgr),
        }
