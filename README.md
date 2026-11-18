# FlowGuard Penang

Proactive Storm Drain Blockage Detection Before the Rain Falls.
Built for the 2026 Han Chiang High School Makers Club Internal Hackathon (The AI Vanguard Challenge).

## Project Structure

- `model_engine.py`: Core computer vision and neural inference engine (`FlowGuardEngine`, LAB CLAHE shadow normalization, Haar face privacy shield, Gabor/Laplacian/FFT feature bank, and Grad-CAM style debris heatmap).
- `train_model.py`: TensorFlow `MobileNetV2` transfer learning pipeline with `RandomBrightness(0.25)` and `RandomContrast(0.25)` Week 10 shadow augmentation, plus portable fallback trainer.
- `generate_sample_dataset.py`: Generates the 6 Penang showcase drain photos in `sample_drains/`, trains Pre-Week-10 and Week-10 weights (`models/flowguard_mobilenetv2_weights.npz`), and records validation metrics in `data/`.
- `app.py`: Streamlit frontend (`@st.cache_resource`) matching Slide 4 architecture.
- `server.py` + `static/`: Zero-dependency HTTP + JSON API server and responsive civic inspection UI (Inspect Studio, Dispatch Map, Anti-Black-Box Code Inspector, Week 10 QA Dashboard, 6-Slide Pitch Deck & 8:00 Stage Timer).
- `deliverables/`: Complete hackathon submission package (`Canva_Pitch_Deck_6_Slides.md`, `Executive_Summary.md`, `QA_Debug_Log_Week10.md`, `Viva_Voce_Defense_Guide.md`).

## Quick Start

### Option A: Zero-Dependency Local Server (Recommended for Offline / Live Stage Demo)
```bash
python server.py
```
Open `http://127.0.0.1:8501` in any browser.

### Option B: Streamlit Runtime (When `streamlit` is installed)
```bash
streamlit run app.py
```

### Regenerate Sample Images & Re-run Validation Calibration
```bash
python generate_sample_dataset.py
```
