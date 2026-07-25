# Meghdrishti — GenAI Satellite Cloud Removal System

[![ISRO BAH 2026](https://img.shields.io/badge/ISRO-Bharatiya%20Antariksh%20Hackathon%202026-saffron?style=for-the-badge)](https://hackathon.iirs.gov.in/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)](LICENSE)

**Meghdrishti** is a generative AI-based cloud removal and atmospheric reconstruction system designed specifically for optical satellite imagery, including ISRO's LISS-IV sensor onboard Resourcesat-2/2A. The system integrates a hybrid U-Net generator with Swin Transformer bottleneck blocks, Spatial Attention Modules (SAM), an automated Cloud Detection Head, and a Multi-Scale PatchGAN Discriminator to perform end-to-end cloud removal, thin-haze suppression, and surface texture recovery.

---

## Problem Statement

Optical satellite remote sensing over tropical and mountainous geographies—such as Northeast India—suffers from persistent cloud cover and atmospheric haze. For ISRO's high-resolution LISS-IV optical sensor, cloud degradation obscures ground reflectances, impeding critical applications in agricultural monitoring, disaster management, land-use classification, and urban planning.

Traditional cloud removal techniques rely on temporal compositing or spatial interpolation, which either suffer from temporal misalignments or produce blurry ground textures. Standard Generative Adversarial Networks (GANs) often introduce hallucinated artifacts or struggle with fine-grained spectral fidelity across non-cloud terrain. **Meghdrishti** addresses these challenges through attention-gated feature reconstruction and multi-objective loss optimization.

---

## System Architecture

```
Cloudy Image (X) ──► Preprocessing ──► Cloud Detection Head ──► Predicted Mask (Mc)
                                             │
                                             ▼
  SPANet Generator (Encoder ──► Swin Bottleneck ──► Attention-Gated Decoder) ──► Reconstructed Image (Y_hat)
                                             │
                                             ▼
                        Multi-Scale PatchGAN Discriminator (D256, D128)
                                             │
                                             ▼
                                Combined Multi-Objective Loss
```

### Architectural Specifications

1. **Preprocessing & Automated Cloud Detection Head**:
   - Accepts raw RGB/multispectral bands $X \in \mathbb{R}^{3 \times H \times W}$.
   - The auxiliary Cloud Detection Head generates a continuous spatial cloud confidence map $M_c \in [0, 1]^{1 \times H \times W}$ to guide the spatial attention gating mechanism.

2. **SPANet Generator Architecture**:
   - **Encoder**: 5-stage convolutional feature extractor with channel progression:
     $$\text{Encoder Channels}: 64 \longrightarrow 128 \longrightarrow 256 \longrightarrow 512 \longrightarrow 512$$
   - **Swin Transformer Bottleneck**: 2 Swin Transformer blocks operating at the lowest feature resolution with feature dimension $C = 512$, window size $W = 8 \times 8$, and multi-head self-attention with $H = 8$ heads.
   - **Attention-Gated Decoder**: Symmetric decoder featuring Spatial Attention Modules (SAM) at each skip connection. SAM gates selectively pass non-cloud ground features while re-synthesizing cloud-occluded spatial regions.

3. **Multi-Scale PatchGAN Discriminator**:
   - Dual-branch discriminator operating concurrently across two spatial resolutions: $256 \times 256$ (full resolution) and $128 \times 128$ (downsampled resolution).
   - Discriminates local texture realism ($70 \times 70$ receptive field patches) as well as global structural coherence.

---

## Mathematical Formulation & Loss Function

The network is optimized end-to-end using a composite 5-term loss function balancing pixel reconstruction, structural realism, spatial attention accuracy, perceptual feature matching, and spectral integrity:

$$L_{\text{total}} = L_1 + L_{\text{GAN}} + 0.10 \cdot L_{\text{attention}} + 0.10 \cdot L_{\text{perceptual}} + 0.05 \cdot L_{\text{spectral}}$$

### Term Definitions

1. **Pixel-Level Reconstruction Loss ($L_1$)**:
   Measures absolute spatial discrepancy between the reconstructed image $\hat{Y}$ and ground-truth cloud-free image $Y$:
   $$L_1 = \|\hat{Y} - Y\|_1$$

2. **Adversarial Multi-Scale GAN Loss ($L_{\text{GAN}}$)**:
   Encourages realistic surface texturing via min-max optimization with the dual PatchGAN discriminators ($D_{256}, D_{128}$):
   $$L_{\text{GAN}} = \sum_{s \in \{256, 128\}} \mathbb{E}_{X, Y}\left[\log D_s(X, Y)\right] + \mathbb{E}_{X}\left[\log (1 - D_s(X, \hat{Y}))\right]$$

3. **Spatial Attention Consistency Loss ($L_{\text{attention}}$)**:
   Constrains predicted SAM attention maps $A$ against ground-truth cloud masks $M_c$:
   $$L_{\text{attention}} = \|A - M_c\|_2^2$$

4. **Perceptual Feature Matching Loss ($L_{\text{perceptual}}$)**:
   Evaluates high-level feature similarity using pre-trained VGG-16 activations at layers `conv2_2` ($\phi_2$) and `conv3_3` ($\phi_3$):
   $$L_{\text{perceptual}} = \sum_{l \in \{2, 3\}} \|\phi_l(\hat{Y}) - \phi_l(Y)\|_1$$

5. **Non-Cloud Spectral Preservation Loss ($L_{\text{spectral}}$)**:
   Enforces channel-wise mean and variance equality across non-cloud ground regions $(1 - M_c)$:
   $$L_{\text{spectral}} = \sum_{c \in \{R,G,B\}} \left| \mu(\hat{Y}_c \odot (1-M_c)) - \mu(Y_c \odot (1-M_c)) \right| + \left| \sigma(\hat{Y}_c \odot (1-M_c)) - \sigma(Y_c \odot (1-M_c)) \right|$$

---

## Datasets

The model was trained and validated on the standard **RICE (Remote Sensing Image Cloud Removing)** benchmark datasets:
- **RICE1**: 500 pairs of cloud-contaminated and cloud-free satellite images.
- **RICE2**: 736 pairs of cloud-contaminated images with explicit ground-truth cloud masks.

📥 **Dataset Download Link**: [Google Drive — RICE Benchmark Dataset](https://drive.google.com/file/d/1Tsm9qEugNyDKLe4bu06e-2IqEhENu64D/view)

---

## Benchmark Quantitative Results

Evaluated on test benchmark images over **100 training epochs**:

| Metric | Score | Description |
| :--- | :---: | :--- |
| **PSNR** | **22.70 dB** | Peak Signal-to-Noise Ratio (higher is better) |
| **SSIM** | **0.843** | Structural Similarity Index Measure (scale 0–1, higher is better) |
| **MAE** | **0.040** | Mean Absolute Error (lower is better) |
| **SAM** | **0.071 rad** | Spectral Angle Mapper (lower is better) |

---

## What We Improved (Upgrades Over Baseline)

| Architectural Component | Original SpA-GAN (Pan et al., 2020) | Meghdrishti (BAH 2026 System) |
| :--- | :--- | :--- |
| **Generator Architecture** | Standard Convolutional Encoder-Decoder | Hybrid U-Net + Swin Transformer Bottleneck |
| **Discriminator Setup** | Single-Resolution Discriminator | Multi-Scale PatchGAN ($256 \times 256$ & $128 \times 128$) |
| **Cloud Masking** | Manual / Pre-computed Input Masks | Automated Internal Cloud Detection Head |
| **Loss Signal** | 3 Losses ($L_1$, $L_{\text{att}}$, $L_{\text{GAN}}$) | 5 Combined Multi-Objective Losses |
| **Serving & Deployment** | CLI Research Code Scripts Only | Production FastAPI Backend + Dockerized Microservice |
| **Target Remote Sensing Data** | Generic Google Earth RGB Images | ISRO LISS-IV & Sentinel-2 Satellite Imagery |
| **Interactive User Interface** | None | Responsive Vercel-Ready Web Studio Interface |

---

## Tech Stack

- **Core Framework**: Python 3.11, PyTorch (CUDA acceleration), torchvision
- **Transformer & Vision Backbones**: `timm` (Swin Transformer modules)
- **Geospatial & Image Processing**: `rasterio`, `OpenCV`, `Pillow`, `NumPy`
- **Model Serving & API**: FastAPI, Uvicorn, Docker
- **Frontend & Deployment**: HTML5, Vanilla CSS3, JavaScript (ES6+), Vercel
- **Version Control**: Git, GitHub Actions

---

## Project Structure Overview

```text
Meghdrishti/
├── models/                  # PyTorch model definitions
│   ├── cloud_head.py        # Automated Cloud Detection Head
│   ├── layers.py            # Spatial Attention Modules (SAM) & Building Blocks
│   ├── models_utils.py      # Weight initializers and helpers
│   ├── dis/                 # Multi-Scale PatchGAN Discriminator
│   └── gen/                 # SPANet Generator with Swin Transformer bottleneck
├── serve/                   # Microservice & Serving directory
│   ├── app.py               # FastAPI server REST endpoints
│   ├── Dockerfile           # Docker container configuration
│   └── requirements.txt     # Serving dependency list
├── frontend/                # Standalone Web Application (Vercel Ready)
│   ├── index.html           # Interactive Cloud Removal Studio
│   ├── css/styles.css       # Custom Glassmorphism UI styling
│   ├── js/app.js            # Frontend API integration & slider controller
│   └── vercel.json          # Vercel deployment configuration
├── readme_images/           # Architecture diagrams and loss visualizers
├── train.py                 # PyTorch training pipeline with multi-loss calculation
├── evaluate.py              # Quantitative evaluation script (PSNR, SSIM, MAE, SAM)
├── data_manager.py          # Dataset loaders for RICE1 and RICE2
├── predict.py               # Standalone inference script for satellite rasters
├── config.yml               # Hyperparameters and model configuration
├── vercel.json              # Root Vercel deployment routing
└── LICENSE                  # MIT License
```

---

## Installation & Setup Guide

### Prerequisites
- Python 3.10+
- NVIDIA GPU with CUDA support (recommended for inference/training)

### 1. Environment Setup
```bash
# Clone repository
git clone https://github.com/arpbiswas04-coder/Meghdrishti.git
cd Meghdrishti

# Create virtual environment
python -m venv venv

# Activate virtual environment (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Install PyTorch and dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r serve/requirements.txt
```

### 2. Running the FastAPI Server Locally
```bash
# Start FastAPI server on port 8000
uvicorn serve.app:app --host 0.0.0.0 --port 8000 --reload
```
Test backend health by navigating to `http://localhost:8000/health`.

---

## API Documentation

### `GET /health`
- **Description**: Returns backend service health status and model state.
- **Response Format**:
  ```json
  {
    "status": "ok",
    "model": "SpA-GAN-BAH2026",
    "version": "1.0.0"
  }
  ```

### `POST /predict`
- **Description**: Upload a cloudy satellite image (`.png`, `.jpg`, `.tif`) to receive cloud removal reconstruction.
- **Request Payload**: `multipart/form-data` with key `file`.
- **Response Format**: Returns PNG image stream of the reconstructed cloud-free raster.

---

## Frontend Web Studio & Vercel Deployment

The `/frontend` directory contains an interactive, single-page web studio featuring a before-after comparison slider, real-time health indicator, and drag-and-drop raster uploader.

### Deploying Frontend to Vercel
1. Import the `Meghdrishti` repository in your Vercel Dashboard.
2. Set **Root Directory** to `frontend` (or keep root with existing `vercel.json`).
3. Add Environment Variable:
   - `MEGHDRISHTI_API_URL`: URL of your deployed FastAPI backend (e.g., `https://your-api.onrender.com`).
4. Click **Deploy**.

---

## Team Credits

Built for **ISRO's Bharatiya Antariksh Hackathon 2026** by **Team Trinova**:

- **Arpan Biswas** (Team Leader) — [GitHub](https://github.com/arpbiswas04-coder)
- **Ayan Kumar Mondal**
- **Debasrit Sahoo**

*Institution*: Haldia Institute of Technology, West Bengal, India.

---

## License

This project is licensed under the [MIT License](LICENSE).
