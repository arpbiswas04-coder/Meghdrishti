import os
import numpy as np
from PIL import Image

data_root = "data/RICE_DATASET"

# Create dummy images for smoke testing since download failed
for ds in ["RICE1", "RICE2"]:
    c_dir = os.path.join(data_root, ds, "cloudy")
    g_dir = os.path.join(data_root, ds, "ground_truth")
    os.makedirs(c_dir, exist_ok=True)
    os.makedirs(g_dir, exist_ok=True)
    
    with open(os.path.join(data_root, ds, "train_list.txt"), "w") as ft, \
         open(os.path.join(data_root, ds, "test_list.txt"), "w") as fv:
        
        for i in range(50):
            name = f"{i}.png"
            c_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            g_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            
            Image.fromarray(c_img).save(os.path.join(c_dir, name))
            Image.fromarray(g_img).save(os.path.join(g_dir, name))
            
            # Split train/test
            if i < 40:
                ft.write(name + "\n")
            else:
                fv.write(name + "\n")
                
print("Synthetic images generated!")
