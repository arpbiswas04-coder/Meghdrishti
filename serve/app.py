import os
import io
import base64
import requests
import torch
import numpy as np
import rasterio
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image

from models.gen.SPANet import Generator
from models.cloud_head import CloudDetectionHead

app = FastAPI(title="SpA-GAN BAH 2026")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

gen_model = None
head_model = None
in_ch_global = 4

def download_checkpoint_if_needed(ckpt_path: str):
    """
    Downloads the trained model checkpoint if it does not exist locally.
    Uses the CHECKPOINT_URL environment variable (Hugging Face / Google Drive / Direct link).
    """
    if not os.path.exists(ckpt_path):
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        checkpoint_url = os.getenv("CHECKPOINT_URL")
        if checkpoint_url:
            print(f"Downloading checkpoint from '{checkpoint_url}' to '{ckpt_path}'...")
            try:
                response = requests.get(checkpoint_url, stream=True, timeout=300)
                response.raise_for_status()
                with open(ckpt_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                print("Checkpoint downloaded successfully.")
            except Exception as e:
                print(f"Error downloading checkpoint from {checkpoint_url}: {e}")
        else:
            print(f"Warning: Checkpoint file '{ckpt_path}' not found and CHECKPOINT_URL env var is not set.")

@app.on_event("startup")
def load_models():
    global gen_model, head_model, in_ch_global
    
    ckpt_path = "checkpoints/best_psnr.pth"
    download_checkpoint_if_needed(ckpt_path)
    
    in_ch = 4
    state_dict = None

    if os.path.exists(ckpt_path):
        state_dict = torch.load(ckpt_path, map_location=device)
        first_layer_weight = state_dict.get('gen.gen.inc.0.weight', None)
        if first_layer_weight is not None:
            in_ch = first_layer_weight.shape[1]

    in_ch_global = in_ch

    gen_model = Generator(gpu_ids=[], in_ch=in_ch).to(device)
    head_model = CloudDetectionHead(in_channels=in_ch).to(device)

    if state_dict is not None:
        model_state = gen_model.state_dict()
        filtered = {
            k: v for k, v in state_dict.items()
            if k in model_state and v.shape == model_state[k].shape
        }
        model_state.update(filtered)
        gen_model.load_state_dict(model_state)
        print(f"Generator: loaded {len(filtered)}/{len(state_dict)} layers")

    gen_model.eval()
    head_model.eval()
    print(f"Models ready on {device} | in_ch={in_ch}")

@app.get("/")
def serve_demo():
    return FileResponse("demo.html")

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "SpA-GAN-BAH2026",
        "device": str(device),
        "checkpoint_exists": os.path.exists("checkpoints/best_psnr.pth")
    }

@app.post("/predict")
async def predict(
    cloudy_image: UploadFile = File(...),
    sar_image: UploadFile = File(None)
):
    cloudy_bytes = await cloudy_image.read()
    
    try:
        with rasterio.io.MemoryFile(cloudy_bytes) as memfile:
            with memfile.open() as src:
                bands = src.count
                if bands >= 4:
                    c_img = src.read([1, 2, 3, 4]).astype(np.float32)
                else:
                    c_img = src.read().astype(np.float32)
                    if c_img.shape[0] == 3:
                        c_img = np.concatenate([c_img, c_img[0:1]], axis=0)
    except Exception:
        img = Image.open(io.BytesIO(cloudy_bytes)).convert('RGB')
        c_img = np.array(img).transpose(2, 0, 1).astype(np.float32)
        c_img = np.concatenate([c_img, c_img[0:1]], axis=0)

    if c_img.max() > 1.0:
        c_img = c_img / 127.5 - 1.0
    else:
        c_img = c_img * 2.0 - 1.0

    c_img_tensor = torch.from_numpy(c_img).unsqueeze(0).to(device)
    input_tensor = c_img_tensor

    if in_ch_global == 6:
        sar_img = np.zeros((2, c_img.shape[1], c_img.shape[2]), dtype=np.float32)
        if sar_image is not None:
            sar_bytes = await sar_image.read()
            with rasterio.io.MemoryFile(sar_bytes) as memfile:
                with memfile.open() as src:
                    sar_img = src.read([1, 2]).astype(np.float32)
        sar_img_tensor = torch.from_numpy(sar_img).unsqueeze(0).to(device)
        input_tensor = torch.cat([c_img_tensor, sar_img_tensor], dim=1)

    with torch.no_grad():
        output = gen_model(input_tensor)
        if isinstance(output, (tuple, list)):
            _, out_tensor = output
        else:
            out_tensor = output

        mask_logits = head_model(input_tensor)
        mask_pred = (torch.sigmoid(mask_logits) > 0.5).float()

    cloud_coverage_pct = round(mask_pred.mean().item() * 100.0, 2)

    out_np = out_tensor.squeeze(0).cpu().numpy()
    out_np = np.clip((out_np + 1) / 2.0, 0, 1)
    out_rgb = (out_np[:3][::-1].transpose(1, 2, 0) * 255).astype(np.uint8)

    pil_img = Image.fromarray(out_rgb)
    buffered = io.BytesIO()
    pil_img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return JSONResponse(content={
        "result_image": img_str,
        "cloud_coverage_pct": cloud_coverage_pct,
        "psnr": None,
        "ssim": None
    })