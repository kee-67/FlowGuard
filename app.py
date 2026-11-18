"""
FlowGuard Penang: Streamlit Municipal Inspection Prototype.

Implements Slide 4 technical architecture:
- @st.cache_resource caches FlowGuardEngine in memory for zero-lag inference.
- 3-Step User Flow:
  01. Snap & Upload (or pick a Penang sample drain photo) + select zone
  02. MobileNetV2-calibrated CNN classification + Grad-CAM Debris Heatmap
  03. Priority Dispatch to live municipal hazard log (data/hazard_log.json)
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st

from model_engine import FlowGuardEngine, ZONE_METADATA

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
SAMPLES_DIR = ROOT / "sample_drains"
HAZARD_LOG_PATH = DATA_DIR / "hazard_log.json"
MANIFEST_PATH = DATA_DIR / "samples_manifest.json"
METRICS_PATH = DATA_DIR / "validation_metrics.json"


@st.cache_resource
def get_flowguard_engine() -> FlowGuardEngine:
    """
    Cache the AI engine across Streamlit reruns using @st.cache_resource.

    Prevents reloading neural weights and Haar privacy cascades on every button
    click, keeping live inference latency under 50 ms on standard hardware.
    """
    return FlowGuardEngine(ROOT)


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def append_hazard_record(record: Dict[str, Any]) -> None:
    rows: List[Dict[str, Any]] = load_json(HAZARD_LOG_PATH, [])
    rows.insert(0, record)
    HAZARD_LOG_PATH.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def main() -> None:
    st.set_page_config(
        page_title="FlowGuard Penang | Storm Drain Inspection",
        layout="wide",
    )

    st.title("FlowGuard Penang: Proactive Storm Drain Blockage Detection")
    st.caption("Han Chiang High School Makers Club | B2G Municipal Inspection Prototype (MBPP)")

    engine = get_flowguard_engine()
    samples = load_json(MANIFEST_PATH, [])
    metrics = load_json(METRICS_PATH, {})

    with st.sidebar:
        st.header("Inspection Controls")
        zone = st.selectbox("Penang Flood Corridor Zone", list(ZONE_METADATA.keys()))
        apply_shadow_fix = st.toggle(
            "Week 10 Shadow Fix (LAB CLAHE + Augmented Head)",
            value=True,
            help="Toggle off to demonstrate the Pre-Week-10 bug where harsh afternoon shadows triggered false alarms.",
        )
        st.divider()
        if metrics:
            st.metric("Validation Accuracy (Week 10)", f"{metrics.get('week10_validation_accuracy', 0)}%")
            st.metric("Average Inference Latency", f"{metrics.get('avg_latency_ms', 0)} ms")
            st.caption(f"Engine: {engine.backend_mode}")

    col_left, col_right = st.columns([5, 7], gap="large")

    with col_left:
        st.subheader("01. Snap & Upload Drain Photo")
        uploaded = st.file_uploader(
            "Upload storm drain photo (JPG, PNG, WEBP)",
            type=["jpg", "jpeg", "png", "webp"],
        )

        sample_titles = {s["title"]: s for s in samples}
        selected_title = st.selectbox(
            "Or select a Penang field sample",
            list(sample_titles.keys()) if sample_titles else ["No samples found"],
        )

        notes = st.text_input("Field observation note (optional)", placeholder="Near corner culvert by hawker stalls")
        run_btn = st.button("Analyze Blockage Severity", type="primary", use_container_width=True)

    source_bytes: Any = None
    source_name = ""
    if uploaded is not None:
        source_bytes = uploaded.getvalue()
        source_name = uploaded.name
    elif sample_titles and selected_title in sample_titles:
        sample_item = sample_titles[selected_title]
        source_bytes = SAMPLES_DIR / sample_item["filename"]
        source_name = sample_item["filename"]

    with col_right:
        st.subheader("02. AI Classification & Visual Proof")
        if source_bytes is None:
            st.info("Upload a photo or select a Penang sample drain on the left to inspect.")
        elif run_btn or source_bytes is not None:
            result = engine.run_inference(
                source_bytes,
                zone=zone,
                apply_shadow_fix=apply_shadow_fix,
            )
            if not result.get("ok"):
                st.error(result.get("error", "Inference failed."))
            else:
                label = result["label"]
                conf = result["confidence"]
                lat = result["latency_ms"]

                m1, m2, m3 = st.columns(3)
                m1.metric("Severity Verdict", label)
                m2.metric("Model Confidence", f"{conf}%")
                m3.metric("Inference Latency", f"{lat} ms")

                img_a, img_b = st.columns(2)
                img_a.image(result["preprocessed_b64"], caption="224x224 Preprocessed Input", use_container_width=True)
                img_b.image(result["heatmap_b64"], caption="CNN Debris Activation Heatmap", use_container_width=True)

                st.write("Class Probabilities:")
                for cls_name, pct in result["probabilities"].items():
                    st.progress(int(round(pct)), text=f"{cls_name}: {pct}%")

                st.info(result["recommended_action"])

                if st.button("03. Dispatch to MBPP Live Hazard Log", use_container_width=True):
                    existing = load_json(HAZARD_LOG_PATH, [])
                    new_id = f"FG-2026-{101 + len(existing)}"
                    append_hazard_record(
                        {
                            "id": new_id,
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M MYT"),
                            "zone": zone,
                            "severity": label,
                            "confidence": conf,
                            "latency_ms": lat,
                            "status": "Pending Dispatch" if label != "Clear" else "Cleared",
                            "source_photo": source_name,
                            "notes": notes or result["recommended_action"],
                        }
                    )
                    st.success(f"Logged {new_id} ({zone}: {label}) to live municipal dashboard.")

    st.divider()
    st.subheader("03. Live Municipal Priority Dispatch Log")
    hazards = load_json(HAZARD_LOG_PATH, [])
    if hazards:
        st.dataframe(hazards, use_container_width=True)
    else:
        st.info("No hazard records logged yet. Run an inspection and click Dispatch.")


if __name__ == "__main__":
    main()
