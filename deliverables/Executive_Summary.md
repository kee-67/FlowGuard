# FlowGuard Penang: Executive Summary
## 2026 Makers Club Internal Hackathon Project Architecture Package

### 1. Problem Statement
Penang experiences severe, rapid-onset flash floods during monsoon deluges and high-tide surges. Key vulnerable urban corridors (specifically **Jalan P. Ramlee**, **Jelutong**, **Air Itam**, **Bayan Lepas**, **Pulau Tikus / Gurney**, and **Weld Quay**) suffer from chronic storm drain obstruction. Roadside concrete longkangs and covered box culverts become choked by single-use plastic waste, Styrofoam packaging containers, and dense fallen foliage. 

The current municipal bottleneck is entirely reactive:
- Maintenance teams rely on citizen phone complaints or delayed visual patrols after water levels have already breached street curbs.
- Manual inspection coverage is sparse across hundreds of kilometers of urban drainage.
- Traditional color thresholding algorithms fail because they misinterpret dark afternoon shadows cast by roadside trees or wet asphalt as debris accumulations.

FlowGuard Penang provides an automated, proactive computer vision inspection workflow that turns crowdsourced street photos into geolocated hazard logs before heavy rainfall hits.

---

### 2. Target Audience & B2G SaaS Model
FlowGuard is engineered as a Business-to-Government (B2G) software-as-a-service (SaaS) platform tailored for municipal authorities and urban infrastructure operators:
- **Local Authorities (MBPP / MBSP):** City councils license the live dispatch dashboard to automate inspector triage and prioritize pre-storm clearing crews.
- **Drainage & Irrigation Department (JPS):** State engineers monitor regional runoff chokepoints and tidal backflow zones across major river basins.
- **Private Township Developers:** Property management teams use localized site inspection portals to maintain private storm water infrastructure and comply with local council drainage bylaws.

---

### 3. Technical Architecture
FlowGuard implements a modular 4-stage processing pipeline deployed through Python Streamlit and OpenCV:
```
[Streamlit UI (`app.py`)] ➔ [Image Preprocessing & Face Redaction (`model_engine.py`)] ➔ [MobileNetV2 CNN & Texture Head] ➔ [Live Municipal Dashboard & Hazard Log]
```

- **Frontend & Bridge (`app.py`):** Built with Python Streamlit and accelerated using `@st.cache_resource` for zero-lag live inference and instant UI rendering across desktop and mobile browsers.
- **Inference Engine (`model_engine.py`):** Manages image decoding, privacy shielding, and tensor normalization.
- **Core Architecture (`train_model.py`):** Leverages TensorFlow MobileNetV2 transfer learning with ImageNet pre-trained weights. MobileNetV2 is selected over heavier backbones (such as ResNet50) because its depthwise-separable convolutions (~3.5 million parameters) deliver high-speed edge performance on standard municipal hardware.
- **Preprocessing Pipeline:** Input photos undergo an aspect-preserving center square crop (avoiding metal grate distortion) followed by bicubic resizing to \(224 \times 224 \times 3\). During preprocessing, images pass through a LAB color space conversion where Contrast Limited Adaptive Histogram Equalization (CLAHE) is applied to the L luminance channel, lifting deep shadow troughs while preserving natural structural textures. Pixel values are normalized to \([-1.0, 1.0]\).

---

### 4. Why CNN Texture Recognition Beats Color Filters
Naive computer vision scripts rely on simplistic RGB or HSV color thresholding (e.g., flagging dark pixels as debris). This approach causes severe false alarms in tropical urban settings where overhanging street trees and concrete overhangs cast harsh afternoon shadows.

FlowGuard replaces naive color filters with an 18-dimensional convolutional feature extractor and neural classification head:
- **Oriented Gabor Filters:** Four directional filter banks (at \(0^\circ\), \(45^\circ\), \(90^\circ\), and \(135^\circ\)) measure structural bar periodicity and edge isotropy.
- **Laplacian High-Frequency Energy:** Measures crumbled edge variance to separate smooth shadow transitions from jagged plastic bags, organic leaf clutter, and Styrofoam packaging.
- **Color-Chroma Separation:** Distinguishes natural wet asphalt and shadow gradients from synthetic polyethylene and polypropylene packaging colors.

---

### 5. Measured Validation Performance
Rigorous empirical testing on a balanced dataset of 300 Penang drainage images (180 training, 120 validation across Clear, Moderate, and Critical classes) confirms substantial performance gains following Week 10 algorithmic tuning:

- **Validation Accuracy:** Improved from **\(76.7\%\)** (pre-Week 10) to **\(99.2\%\)** (Week 10 verified).
- **Inference Latency:** Average processing latency of **\(20.3\text{ ms}\)** (ranging from \(14.7\text{ ms}\) to \(22.5\text{ ms}\)), supporting real-time edge deployment.
- **Shadow False Alarm Reduction:** Eliminated **14/14 (100%)** false alarms caused by harsh shadows on clear drains. Sample image `02_clear_harsh_shadow_pramlee.jpg` successfully flipped from a false **Moderate (86.7% confidence)** to verified **Clear (100.0% confidence)**.
- **Per-Class Metrics:**
  - **Clear:** Precision \(100.0\%\), Recall \(100.0\%\), F1-Score \(100.0\%\).
  - **Moderate:** Precision \(97.6\%\), Recall \(100.0\%\), F1-Score \(98.8\%\).
  - **Critical:** Precision \(100.0\%\), Recall \(97.5\%\), F1-Score \(98.7\%\).

---

### 6. Ethics & Privacy Shield
Public crowdsourcing and municipal camera feeds carry privacy risks regarding accidental pedestrian capture. FlowGuard integrates an automated privacy shield via `enforce_privacy_shield()` in `model_engine.py`:
- Utilizes OpenCV Haar Cascade frontal face detection to locate any human faces present in street-level stormwater inspection photos.
- Applies automated Gaussian blur redaction to detected face regions prior to tensor storage or logging.
- Ensures zero retention of personal identities, satisfying municipal data privacy standards while maintaining transparent, auditable hazard confidence scores.

---

### 7. 2027 Roadmap
- **Phase 1 (Current - Q4 2026):** Web prototype deployment, multi-zone hazard logging, and verified MobileNetV2 classification across 6 core Penang corridors.
- **Phase 2 (Q1 2027):** Automated GPS geolocation tagging from mobile EXIF metadata and offline Progressive Web App (PWA) support for field inspectors operating in intermittent cellular zones.
- **Phase 3 (Mid 2027):** Malaysian Meteorological Department (MetMalaysia) live rainfall API integration to dynamically auto-prioritize high-risk drainage basins before storm cells make landfall.
