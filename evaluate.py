import os
import argparse
import torch
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from numpy import mean as mae

from models.gen.SPANet import Generator
from data_manager import TrainDataset
from torch.utils.data import DataLoader

def calculate_sam(pred, target):
    dot = np.sum(pred * target, axis=0)
    norm_p = np.linalg.norm(pred, axis=0)
    norm_t = np.linalg.norm(target, axis=0)
    denom = norm_p * norm_t + 1e-8
    cos_theta = np.clip(dot / denom, -1.0, 1.0)
    sam = np.arccos(cos_theta)
    return np.mean(sam)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--dataroot', type=str, required=True)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    state_dict = torch.load(args.checkpoint, map_location=device)
    first_layer_weight = state_dict.get('gen.gen.inc.0.weight', None)
    in_ch = first_layer_weight.shape[1] if first_layer_weight is not None else 4

    # Detect out channels from checkpoint
    final_conv_weight = state_dict.get('gen.gen.final_conv.0.weight', None)
    out_ch = final_conv_weight.shape[0] if final_conv_weight is not None else 3

    gen = Generator(gpu_ids=[], in_ch=in_ch).to(device)

# Filter only keys that match current model shape exactly
    model_state = gen.state_dict()
    filtered = {
    k: v for k, v in state_dict.items()
    if k in model_state and v.shape == model_state[k].shape
    }
    model_state.update(filtered)
    gen.load_state_dict(model_state)
    print(f"Loaded {len(filtered)}/{len(state_dict)} layers from checkpoint")
    gen.eval()

    # Use RICE dataset loader instead of SEN12MSCRDataset
    class SimpleConfig:
        datasets_dir = os.path.join(args.dataroot, 'RICE1')
        image_size = 256

    dataset = TrainDataset(SimpleConfig())
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    os.makedirs('results/images', exist_ok=True)

    results = []
    print(f"Evaluating {len(dataset)} samples...")

    for i, batch in enumerate(loader):
        # Handle both dict and tuple/list returns from dataloader
        if isinstance(batch, dict):
            c_img = batch['input'].to(device)
            t_img = batch['target'].to(device)
        else:
            c_img = batch[0].to(device)
            t_img = batch[1].to(device)

        input_tensor = c_img

        with torch.no_grad():
            output = gen(input_tensor)
            # Handle both (attention, image) tuple and plain tensor output
            if isinstance(output, (tuple, list)):
                pred = output[1]
            else:
                pred = output

        p_np = pred.squeeze(0).cpu().numpy()
        t_np = t_img.squeeze(0).cpu().numpy()
        c_np = c_img.squeeze(0).cpu().numpy()

        p_np = np.clip((p_np + 1) / 2.0, 0, 1)
        t_np = np.clip((t_np + 1) / 2.0, 0, 1)
        c_np = np.clip((c_np + 1) / 2.0, 0, 1)

        min_ch = min(p_np.shape[0], t_np.shape[0])
        p_np = p_np[:min_ch]
        t_np = t_np[:min_ch]
        c_np = c_np[:min_ch]

        p_val = psnr(t_np, p_np, data_range=1.0)
        s_val = ssim(t_np, p_np, data_range=1.0, channel_axis=0)
        m_val = float(np.mean(np.abs(t_np - p_np)))
        sam_val = calculate_sam(p_np, t_np)

        results.append({
            'sample_id': i,
            'psnr': round(p_val, 4),
            'ssim': round(s_val, 4),
            'mae': round(m_val, 4),
            'sam': round(sam_val, 4)
        })

        # Use first 3 channels for RGB visualization
        c_rgb = (c_np[:3][::-1].transpose(1, 2, 0) * 255).astype(np.uint8)
        p_rgb = (p_np[:3][::-1].transpose(1, 2, 0) * 255).astype(np.uint8)
        t_rgb = (t_np[:3][::-1].transpose(1, 2, 0) * 255).astype(np.uint8)

        w, h = c_rgb.shape[1], c_rgb.shape[0]
        combined = Image.new('RGB', (w * 3, h))
        combined.paste(Image.fromarray(c_rgb), (0, 0))
        combined.paste(Image.fromarray(p_rgb), (w, 0))
        combined.paste(Image.fromarray(t_rgb), (w * 2, 0))

        draw = ImageDraw.Draw(combined)
        draw.text((5, 5),       "Cloudy Input",   fill=(255, 255, 0))
        draw.text((w + 5, 5),   "Model Output",   fill=(255, 255, 0))
        draw.text((w * 2 + 5, 5), "Ground Truth", fill=(255, 255, 0))
        draw.text((w + 5, 20),
                  f"PSNR: {p_val:.2f} dB  SSIM: {s_val:.3f}",
                  fill=(255, 0, 0))

        combined.save(f"results/images/sample_{i:04d}.png")
        print(f"  [{i+1}/{len(dataset)}] PSNR: {p_val:.2f} dB | SSIM: {s_val:.3f}")

    df = pd.DataFrame(results)
    df.to_csv("results/metrics.csv", index=False)

    print("\n====== FINAL RESULTS ======")
    print(f"Mean PSNR : {df['psnr'].mean():.4f} dB")
    print(f"Mean SSIM : {df['ssim'].mean():.4f}")
    print(f"Mean MAE  : {df['mae'].mean():.4f}")
    print(f"Mean SAM  : {df['sam'].mean():.4f}")
    print("===========================")
    print("Images saved to results/images/")
    print("CSV saved to results/metrics.csv")

if __name__ == '__main__':
    main()