# Meghdrishti — GenAI Satellite Cloud Removal Frontend

Frontend web interface for **Meghdrishti** (ISRO Bharatiya Antariksh Hackathon 2026).

## Environment & Zero-Global-Install Guarantee
This project relies **exclusively** on the existing Python virtual environment (`/venv`) located in the root project directory. No global Python or npm packages are installed on your system.

---

## How to Run Locally

### Step 1: Ensure Backend Server is Running
Start the FastAPI backend server using the project's root `venv`:

```powershell
# From project root (cloud_removal_bah2026)
.\venv\Scripts\uvicorn serve.app:app --host 0.0.0.0 --port 8000
```

Verify backend health at: `http://localhost:8000/health` (returns `{"status": "ok", "model": "SpA-GAN-BAH2026"}`).

### Step 2: Serve the Frontend

#### Option A: Direct Browser File (Simplest)
Open `frontend/index.html` directly in your web browser (Double-click or drag into Chrome/Edge/Firefox).

#### Option B: HTTP Server via existing Root `venv` (Recommended)
Run Python's built-in HTTP server using the existing `venv`:

```powershell
# From project root
.\venv\Scripts\python.exe -m http.server 3000 --directory frontend
```

Then open `http://localhost:3000` in your web browser.

---

## Refactoring Summary & Features
1. **Site-Wide Scroll Entrance Animations**: Unified single-trigger `IntersectionObserver` fading & sliding up (`24px` → `0px`) all stat cards, flowchart nodes, detail breakdown cards, loss terms, table rows, and demo cards with staggered index delays (`60ms`).
2. **Prominent Hero Background Video**: Increased video element opacity to `0.88` with a balanced gradient overlay (`rgba(5, 9, 17, 0.35)` to `0.85`), delivering high visual video prominence while preserving sharp text contrast.
3. **Continuous Background Image (`background.jpg`)**: Relocated `background.jpg` to `assets/background.jpg` and deployed as a continuous fixed background (`background-attachment: fixed`, `background-size: cover`) across all sections below the hero with zero hard cuts.
4. **Symmetric Curtain Reveal Slider Labels**: Programmatic symmetric fading and scale curves for both `"ORIGINAL (CLOUDY INPUT)"` (left) and `"RECONSTRUCTED (CLOUD-FREE)"` (right) badges as the handle approaches within 25% of either edge.
5. **Glassmorphism Engineering Dashboard UI**: Transformed all cards, flowchart nodes, Stage 3 & 4 detail drawers, loss cards, comparison table, tech badges, and upload dropzone into translucent glassmorphic components with `backdrop-filter: blur(14px)`.
