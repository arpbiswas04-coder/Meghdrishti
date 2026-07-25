import os
import subprocess
import tarfile
import numpy as np
import pandas as pd
from pathlib import Path

DATA_ROOT = Path("./data")
DATA_ROOT.mkdir(exist_ok=True)

# ── 1. RICE dataset check ─────────────────────────────────────────────────────
print("Checking RICE dataset...")
rice_path = DATA_ROOT / "RICE"
if rice_path.exists():
    print(f"  RICE found at {rice_path}. OK.")
else:
    print("  RICE not found. Check the original repo README for download link.")

# ── 2. SEN12MS-CR — Asia East spring only (~10 GB total) ─────────────────────
print("\nDownloading SEN12MS-CR (Asia East, spring season only)...")
sen12_path = DATA_ROOT / "SEN12MSCR"
sen12_path.mkdir(exist_ok=True)

# Only downloading Asia East + spring to stay within storage budget.
# Credentials are public: username = m1554803, password = m1554803
sen12_files = {
    "s2_cloudy_asiaEast":    "ftp://m1554803:m1554803@dataserv.ub.tum.de/ROIs1158_spring_s2_asiaEast.tar.gz",
    "s2_cloudfree_asiaEast": "ftp://m1554803:m1554803@dataserv.ub.tum.de/ROIs1158_spring_s2_cloudfree_asiaEast.tar.gz",
    "s1_asiaEast":           "ftp://m1554803:m1554803@dataserv.ub.tum.de/ROIs1158_spring_s1_asiaEast.tar.gz",
}

for name, url in sen12_files.items():
    out_tar = sen12_path / f"{name}.tar.gz"
    extracted_flag = sen12_path / f"{name}.extracted"

    if extracted_flag.exists():
        print(f"  {name} already extracted. Skipping.")
        continue

    if not out_tar.exists():
        print(f"  Downloading {name}...")
        ret = subprocess.run(["wget", "-q", "--show-progress", "-O", str(out_tar), url])
        if ret.returncode != 0:
            print(f"  wget failed, trying curl...")
            subprocess.run(["curl", "-L", "-o", str(out_tar), url])

    if out_tar.exists():
        print(f"  Extracting {name}...")
        with tarfile.open(out_tar, "r:gz") as tar:
            tar.extractall(path=sen12_path)
        out_tar.unlink()  # delete tar immediately to free space
        extracted_flag.touch()
        print(f"  Done. Tar deleted to save space.")
    else:
        print(f"  WARNING: Could not download {name}. Skipping.")

# ── 3. CloudSEN12+ — 2000 train + 500 val patches only (~3 GB) ───────────────
print("\nDownloading CloudSEN12+ subset (2500 high-quality patches only)...")
cloudsen_path = DATA_ROOT / "CloudSEN12"
cloudsen_path.mkdir(exist_ok=True)
done_flag = cloudsen_path / "download.done"

if done_flag.exists():
    print("  CloudSEN12+ already downloaded. Skipping.")
else:
    try:
        import tacoreader, mlstac
        dataset = tacoreader.load("tacofoundation:cloudsen12-l2a")
        meta = dataset.metadata
        high = meta[meta["label_type"] == "high"]
        train_sub = high[high["split"] == "train"].head(2000)
        val_sub   = high[high["split"] == "val"].head(500)
        subset    = pd.concat([train_sub, val_sub])
        print(f"  Fetching {len(subset)} patches...")
        datacube = mlstac.get_data(dataset=subset)
        np.save(cloudsen_path / "images.npy", datacube[:, :13])
        np.save(cloudsen_path / "masks.npy",  datacube[:, 13:14])
        np.save(cloudsen_path / "splits.npy", np.array([0]*2000 + [1]*500))
        done_flag.touch()
        print(f"  Saved to {cloudsen_path}")
    except Exception as e:
        print(f"  CloudSEN12+ download failed: {e}")
        print("  Will use pseudo-labels from SEN12MS-CR pairs instead (auto-handled in training).")

# ── 4. Storage report ─────────────────────────────────────────────────────────
total = sum(f.stat().st_size for f in DATA_ROOT.rglob("*") if f.is_file())
print(f"\nTotal data on disk: {total / 1e9:.2f} GB")
print("All done. Run: python train.py --dataset sen12mscr --use_sar --name bah2026")
