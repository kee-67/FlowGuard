# FlowGuard Penang: Official Week 10 Alpha & Week 11 Beta QA Debug Log
## Comprehensive Bug Resolution, Validation Metrics, and Confusion Matrix Analysis

---

### 1. Executive Summary of QA Testing
This official debug log documents all 6 bugs encountered and resolved during the Week 10 Alpha and Week 11 Beta engineering cycles of FlowGuard Penang. Through rigorous algorithmic refinements in preprocessing, privacy shielding, and tensor inference (`model_engine.py` and `train_model.py`), overall validation accuracy increased from **\(76.7\%\)** to **\(99.2\%\)** across the 120-image validation benchmark.

---

### 2. Complete Bug Tracking Table (CSV Log)

| Bug ID | Phase | Module | Severity | Symptom | Resolution | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **BUG-01** | Week 10 Alpha | Preprocessing | High | Harsh afternoon shadows on clean drains triggered false Moderate/Critical alerts | Added LAB L-channel CLAHE normalization plus brightness/contrast augmentation retraining | Fixed & Verified |
| **BUG-02** | Week 10 Alpha | Inference Pipeline | High | Non-square phone photos distorted metal grate spacing during naive \(224 \times 224\) resize | Implemented aspect-preserving center square crop prior to bicubic/area resize | Fixed & Verified |
| **BUG-03** | Week 10 Alpha | Privacy & Ethics | Critical | Street photos risked capturing passerby faces near roadside longkang | Integrated OpenCV Haar Cascade face redaction blur prior to tensor storage | Fixed & Verified |
| **BUG-04** | Week 10 Alpha | Frontend Performance | Medium | Reloading model weights per upload spiked first-run latency above \(1.4\) seconds | Wrapped engine initialization in `@st.cache_resource` and singleton server state | Fixed & Verified |
| **BUG-05** | Week 11 Beta | Input Validation | Medium | Corrupt 0-byte or unsupported file uploads threw unhandled OpenCV exceptions | Added `validate_and_load_image()` guard returning structured user recovery messages | Fixed & Verified |
| **BUG-06** | Week 11 Beta | UI Accessibility | Medium | Dark status chips lacked clear label pairing and keyboard Escape handling | Added WCAG AA contrast-checked badges and full Tab/Enter/Escape modal bindings | Fixed & Verified |

---

### 3. Detailed Root Cause Analysis & Fixes

#### BUG-01: Harsh Afternoon Shadow False Alarms
- **Root Cause:** In tropical urban settings (e.g., Jalan P. Ramlee), overhanging angsana trees and concrete awnings cast deep shadow troughs over pristine metal grates. Naive RGB/grayscale intensity models misclassified low-luminance shadow pixels as accumulated rubbish.
- **File & Function:** `model_engine.py` in `preprocess_drain_image()` and `train_model.py` in `build_mobilenetv2_keras_model()`.
- **Exact Fix:** Converted input tensors to the LAB color space, applied Contrast Limited Adaptive Histogram Equalization (CLAHE) with a clip limit of \(2.1\) on the L luminance channel, and added `RandomBrightness(0.25)` plus `RandomContrast(0.25)` data augmentation layers during training.
- **Before-and-After Evidence:** Eliminated \(14/14\) shadow false alarms. Sample test image `02_clear_harsh_shadow_pramlee.jpg` successfully flipped from **Moderate (\(86.7\%\)** confidence) to verified **Clear (\(100.0\%\)** confidence).

#### BUG-02: Non-Square Aspect Ratio Grate Distortion
- **Root Cause:** Direct resizing of rectangular smartphone photos (\(4:3\) or \(16:9\)) into standard \(224 \times 224\) tensors squashed metal grate bars horizontally, disrupting periodicity FFT features.
- **File & Function:** `model_engine.py` in `preprocess_drain_image()`.
- **Exact Fix:** Implemented an aspect-preserving center square crop prior to bicubic resizing.

#### BUG-03: Roadside Pedestrian Privacy Risk
- **Root Cause:** Street-level stormwater photos captured incidental passersby near roadside longkangs, violating data privacy protocols.
- **File & Function:** `model_engine.py` in `enforce_privacy_shield()`.
- **Exact Fix:** Integrated OpenCV Haar Cascade frontal face detection (`haarcascade_frontalface_default.xml`) with automatic Gaussian blur redaction prior to tensor persistence.

#### BUG-04: First-Run Latency Spike
- **Root Cause:** Initializing MobileNetV2 weights and OpenCV filters per user upload caused UI lag exceeding \(1.4\) seconds.
- **File & Function:** `app.py` and `model_engine.py` in `FlowGuardEngine`.
- **Exact Fix:** Wrapped engine instantiation inside Streamlit's `@st.cache_resource` decorator alongside startup buffer warmup.

#### BUG-05: Corrupt Upload Exceptions
- **Root Cause:** Uploading zero-byte files or unsupported format binaries (`.txt`, `.docx`) triggered unhandled OpenCV `cv2.imdecode` crashes.
- **File & Function:** `model_engine.py` in `validate_and_load_image()`.
- **Exact Fix:** Added rigorous file size, resolution, and header validation returning structured error banners without throwing exceptions.

#### BUG-06: UI Accessibility Deficiencies
- **Root Cause:** Early prototype status chips lacked sufficient dark-mode contrast ratios and missed keyboard focus and Escape drawer bindings.
- **File & Function:** `static/styles.css`, `static/index.html`, and `static/app.js`.
- **Exact Fix:** Updated light and dark badge palettes to pass WCAG AA contrast standards (6.49:1 to 9.42:1) and implemented full Tab, Enter/Space, and Escape navigation bindings.

---

### 4. Validation Performance & Confusion Matrix Analysis

- **Dataset Breakdown:** 180 training images, 120 validation images across 3 classes (`Clear`, `Moderate`, `Critical`).
- **Validation Accuracy:** Pre-Week 10: **\(76.7\%\)** ➔ Week 10 Verified: **\(99.2\%\)**.
- **Inference Latency:** Average **\(20.3\text{ ms}\)** (Min: \(14.7\text{ ms}\), Max: \(22.5\text{ ms}\)).

#### Pre-Week 10 Confusion Matrix (Accuracy: 76.7%)
| True \ Predicted | Clear | Moderate | Critical |
| :--- | :--- | :--- | :--- |
| **Clear** | **22** | 14 | 4 |
| **Moderate** | 0 | **33** | 7 |
| **Critical** | 0 | 3 | **37** |

*Observation: 14 clean drains under harsh tree shadows were incorrectly misclassified as Moderate or Critical blockages.*

#### Week 10 Verified Confusion Matrix (Accuracy: 99.2%)
| True \ Predicted | Clear | Moderate | Critical |
| :--- | :--- | :--- | :--- |
| **Clear** | **40** | 0 | 0 |
| **Moderate** | 0 | **40** | 0 |
| **Critical** | 0 | 1 | **39** |

*Observation: Shadow false alarms were entirely eliminated, resulting in zero false positives for clear drains and near-perfect diagonal classification.*

#### Per-Class Performance Summary
- **Clear:** Precision \(100.0\%\), Recall \(100.0\%\), F1-Score \(100.0\%\)
- **Moderate:** Precision \(97.6\%\), Recall \(100.0\%\), F1-Score \(98.8\%\)
- **Critical:** Precision \(100.0\%\), Recall \(97.5\%\), F1-Score \(98.7\%\)
