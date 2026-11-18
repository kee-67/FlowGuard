# FlowGuard Penang Design Direction

> Reading this as: Municipal flood-prevention AI inspection studio and dispatch console for Penang city engineers (MBPP), residents, and hackathon judges, in a civic-hydrological industrial style, dial ENERGY 2 / RHYTHM 2 / MOTION 1.

Dial: ENERGY 2 / RHYTHM 2 / MOTION 1

## 1. Visual Identity & Craftsmanship
FlowGuard Penang is an engineering inspection and municipal dispatch console for roadside storm drains (*longkang*) across Penang Island flood corridors (Jalan P. Ramlee, Jelutong, Air Itam, Bayan Lepas, Pulau Tikus / Gurney, Weld Quay). Its aesthetic combines civic hydrological mapping with field-ready technical legibility rather than generic dark SaaS templates.

- **Identity Motif**: Calibrated drainage-grate divider ticks and sharp `2px`/`6px` architectural corner geometry paired with tabular monospace telemetry (`224x224x3`, `ms` latency, confidence percentages).
- **Focal Point Per View**: Each workspace view isolates one decision:
  - Inspection Studio: Upload or select a drain photo, verify the Grad-CAM debris activation heatmap, and dispatch if severity is Moderate or Critical.
  - Dispatch Map & Log: Pinpoint active Critical/Moderate hazards on the Penang corridor map and advance crew status.
  - Anti-Black-Box Architecture: Trace a single `224x224` drain image through the four pipeline stages and read the exact Python logic.
  - Week 10 QA & Pitch Stage: Inspect real validation metrics, run live latency benchmarks, and pace the 8-minute Viva-Voce presentation.

## 2. Color Palette (WCAG AA Verified, Max 3 Core + 1 Accent)
All text pairings pass WCAG AA normal text (4.5:1+) and interactive borders pass non-text contrast (3.0:1+):

### Light Mode (Default Field & Stage Daylight Mode)
- **Neutral Base**: Warm Concrete `#F4F3EE` (background), Pure Field Paper `#FFFFFF` (cards and inspection panels), Subtle Card Divider `#D8D6CE`, Interactive Input Border `#75818A` (3.99:1 on white).
- **Primary Ink**: Deep Asphalt `#182229` (14.54:1 on `#F4F3EE`, 16.16:1 on `#FFFFFF`).
- **Secondary Ink**: Slate Silt `#465560` (6.93:1 on `#F4F3EE`, 7.70:1 on `#FFFFFF`).
- **Core Brand Color**: Penang Strait Teal `#124559` (10.39:1 with white text) used for civic navigation chrome, active pipeline block borders, and focus rings.
- **Single Deliberate Accent**: Municipal Safety Amber `#B6450D` (5.46:1 with white text) reserved for the primary Analyze / Priority Dispatch CTA and highlighted active map pin.

### Dark Mode (Monsoon Operations Control Toggle)
- **Neutral Base**: Monsoon Slate `#0D151C` (background), Drainage Steel `#15222D` (cards and panels), Subtle Divider `#253746`, Interactive Border `#5B778D` (3.44:1 on `#15222D`).
- **Primary Ink**: Mist Chalk `#ECF1F4` (16.17:1 on `#0D151C`, 14.21:1 on `#15222D`).
- **Secondary Ink**: Harbor Mist `#A8BAC7` (9.21:1 on `#0D151C`, 8.10:1 on `#15222D`).
- **Core Brand Color**: Strait Cyan-Teal `#58A6C1` (6.69:1 with `#0D151C` text, 5.88:1 on `#15222D`).
- **Single Deliberate Accent**: High-Vis Beacon `#F08C4B` (7.50:1 with `#0D151C` text).

### Functional Severity Badges (Used Only for Real Inference Status)
- **Clear**: `#166534` on `#DCFCE7` (Light, 6.49:1) / `#86EFAC` on `#0F3D22` (Dark, 8.74:1).
- **Moderate**: `#9A3412` on `#FEF3C7` (Light, 6.56:1) / `#FDE68A` on `#5C2A07` (Dark, 9.42:1).
- **Critical**: `#991B1B` on `#FEE2E2` (Light, 6.80:1) / `#FCA5A5` on `#631717` (Dark, 6.64:1).

## 3. Typography & Hierarchy
- **Primary Sans**: `Plus Jakarta Sans`, `Segoe UI`, sans-serif. Chosen for clear civic signage geometry and unambiguous numerals in field reports.
- **Technical Monospace**: `IBM Plex Mono`, `Consolas`, monospace. Used strictly for tensor dimensions (`224x224x3`), inference latency (`ms`), probability vectors, and inspected Python source blocks (R-06).

## 4. Documented Design Reasons (R-31)
- **Color**: Warm Concrete and Penang Strait Teal reflect municipal civil engineering drawings and Penang coastal hydrology instead of generic AI blue-purple gradients (R-01).
- **Layout**: Asymmetric 5:7 studio split on Inspection (controls left, tensor + Grad-CAM visual proof right) and tabular dispatch view on Municipal Log ensure every screen centers on a single engineering decision (R-05, R-20).
- **Typography**: Pairing geometric civic sans with monospace telemetry separates human action labels from verifiable machine measurements (R-06).
- **Spacing & Radius**: Crisp `4px` and `8px` architectural radii differentiate buttons/inputs from cards without turning every container into a pill (R-11).
- **Icons**: Custom inline SVG icons (drain grate, hydrological contour pin, camera shutter, neural filter grid) represent real civil-engineering concepts with zero generic AI sparkles or decorative emojis (R-04).
- **Shadows & Motion**: Subtle 1px elevation borders keep the interface flat and crisp (`MOTION 1`: hover and active focus transitions only, no perpetual pulses) (R-12, R-19).
