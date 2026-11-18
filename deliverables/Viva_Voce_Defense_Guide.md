# FlowGuard Penang: Complete 8-Minute Viva-Voce Stage Playbook

## Executive Summary & Panel Context
- **Target Panel**: Chief Judge Yap Chin Pok and judging panel.
- **Total Time**: 8 Minutes (3:00 Pitch Deck, 3:00 Live Software Demo, 2:00 Rapid Q&A).
- **Core Principle**: Full transparency, zero black boxes, absolute technical defensibility across all team members following the 80/20 Mentor-Mentee rule (Beginners and Intermediates present core modules and handle live QA questions).

---

## Part 1: 3-Minute Pitch Script & Slide Transitions
- **Time 0:00 - 0:45 (Slides 1-2)**: Speaker [Name 3 - Intermediate] introduces FlowGuard Penang, addressing rapid urban flooding in Jalan P. Ramlee and Jelutong caused by blocked roadside storm drains.
- **Time 0:45 - 1:30 (Slide 3)**: Speaker [Name 4 - Intermediate] walks through the 3-step user flow: snap photo, AI classification in milliseconds, and priority municipal dashboard dispatch.
- **Time 1:30 - 2:15 (Slide 4)**: Mentor/Expert [Name 1 - Expert] explains the technical pipeline from Streamlit UI to MobileNetV2 feature extraction, proving why color filters fail against shadows while our CNN succeeds.
- **Time 2:15 - 3:00 (Slides 5-6)**: Speaker [Name 6 - Beginner] shares Week 10 QA metrics (99.2% accuracy, 20.3 ms latency) and privacy ethics, handing over to Speaker [Name 2 - Expert] for the B2G roadmap before transitioning to the live demo.

---

## Part 2: 3-Minute Live Demo Choreography
- **Setup**: Open the live web console (`python server.py` at `http://127.0.0.1:8501`) or the Streamlit frontend (`streamlit run app.py`).
- **Step 1: Shadow False Alarm Resolution Demonstration**:
  - Select sample image `02_clear_harsh_shadow_pramlee.jpg`.
  - Toggle **Week 10 Shadow-Invariance Fix** to **OFF**.
  - *Observe*: System misclassifies the harsh afternoon shadow as "Moderate" blockage with 86.7% confidence (pre-Week-10 behavior, demonstrating past 14/14 false alarms).
  - Toggle **Week 10 Shadow-Invariance Fix** to **ON**.
  - *Observe*: System instantly re-evaluates via `preprocess_drain_image()` with LAB CLAHE luminance balancing, outputting "Clear" with 100.0% confidence.
- **Step 2: Critical Hazard Dispatch Simulation**:
  - Select sample image `05_critical_plastic_jalan_pramlee.jpg`.
  - *Observe*: System detects heavy plastic blockage, outputting "Critical" classification (99.3% confidence, 22.5 ms latency).
  - Click **Dispatch to MBPP Hazard Log**.
  - *Observe*: Live municipal dashboard updates immediately, adding Jalan P. Ramlee to active dispatch queues on the Penang Corridor Map alongside the Grad-CAM spatial heatmap overlay.

---

## Part 3: 8 Tough Anti-Black Box Technical Q&A Cards

### Card 1: Beginner / Intermediate Q&A (Privacy & Data Collection)
- **Question**: "How do you ensure pedestrian privacy when citizens upload photos of public streets?"
- **Answered By**: [Name 5 - Beginner] or [Name 6 - Beginner].
- **Defense Script**: "We enforce an automatic privacy shield using OpenCV Haar Cascades (`_FACE_CASCADE`) in `model_engine.py` via `enforce_privacy_shield()`. Any detected human face is instantly blurred with a Gaussian filter before feature extraction, ensuring zero personal images or identities are stored."

### Card 2: Intermediate Q&A (Shadow Handling & Preprocessing)
- **Question**: "Why did afternoon shadows trigger false alarms in earlier testing, and how did you fix it?"
- **Answered By**: [Name 3 - Intermediate].
- **Defense Script**: "In pre-Week-10 testing, 14 out of 14 clear drains with deep shadows were misclassified as blocked because dark pixel counts spiked. We resolved this in Week 10 by introducing CLAHE luminance balancing and local shadow lifting in `preprocess_drain_image()` within `model_engine.py`. This normalizes lighting while preserving actual debris texture."

### Card 3: Intermediate Q&A (Inference Latency & Edge Performance)
- **Question**: "How fast is your model, and can municipal workers run it on ordinary mobile phones?"
- **Answered By**: [Name 4 - Intermediate].
- **Defense Script**: "Our average inference latency is 20.3 ms (ranging from 14.7 ms to 22.5 ms) as verified in `validation_metrics.json`. By using MobileNetV2 transfer learning with 224x224 tensor inputs, the model executes smoothly on standard smartphones and lightweight web servers without requiring expensive GPU clusters."

### Card 4: Expert Q&A (Feature Engineering & CNN vs Color Filters)
- **Question**: "Why not just use simple RGB color thresholding instead of a Convolutional Neural Network?"
- **Answered By**: [Name 1 - Expert].
- **Defense Script**: "Simple color filters fail because dark wet concrete, oily road patches, and tree shadows share dark RGB values with organic silt. Our `extract_features_and_spatial_map()` function in `model_engine.py` computes an 18-dimensional feature vector combining oriented Gabor filter responses and Laplacian high-frequency edge energy, allowing the model to distinguish true debris textures from shadows."

### Card 5: Expert Q&A (Model Training & Loss Function)
- **Question**: "How was the MobileNetV2 classification head trained, and what dataset size was used?"
- **Answered By**: [Name 2 - Expert].
- **Defense Script**: "The model was trained on 180 training images and validated on 120 held-out images across three severity classes (Clear, Moderate, Critical), achieving 99.2% validation accuracy. Training scripts in `train_model.py` optimize cross-entropy loss with data augmentation covering rotations, brightness jitter, and shadow simulations."

### Card 6: Intermediate Q&A (Spatial Heatmap & Grad-CAM Verification)
- **Question**: "How do you prove the AI is looking at the actual debris rather than guessing randomly?"
- **Answered By**: [Name 3 - Intermediate].
- **Defense Script**: "We generate a 56x56 spatial activation debris heatmap in `model_engine.py` using convolutional responses from Gabor and Laplacian layers. This acts as a Grad-CAM style visual proof overlaid on the Streamlit dashboard, letting judges and municipal workers visually verify where the model detected blockage."

### Card 7: Expert Q&A (System Architecture & Fallback Robustness)
- **Question**: "What happens if TensorFlow fails to load or the server runs in an offline environment?"
- **Answered By**: [Name 1 - Expert].
- **Defense Script**: "Our `FlowGuardEngine` class implements a dual-mode fallback architecture. If the Keras MobileNetV2 model file is absent, it automatically falls back to our calibrated depthwise-separable convolutional feature bank and NumPy neural head (`flowguard_mobilenetv2_weights.npz`), guaranteeing zero downtime during field demos."

### Card 8: Beginner / Intermediate Q&A (Deployment & Scalability)
- **Question**: "How does the system integrate with municipal maintenance workflows like MBPP?"
- **Answered By**: [Name 5 - Beginner].
- **Defense Script**: "Both `app.py` and `server.py` map submissions to specific Penang corridors defined in `ZONE_METADATA` (such as Jalan P. Ramlee and Jelutong, with schematic map coordinates `map_x`, `map_y`). When a critical hazard is confirmed, it logs the zone, confidence score, latency, and timestamp into `data/hazard_log.json` for crew routing (with auto-GPS EXIF tagging scheduled for Phase 2 in Q1 2027)."
