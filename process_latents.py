# preprocess_latents.py
import os
import torch
from PIL import Image
from torchvision import transforms
from diffusers import AutoencoderKL
import pandas as pd
from tqdm import tqdm

MODEL_ID = "runwayml/stable-diffusion-v1-5"
TARGET_DIR = './dataset/arrow_dataset/target_images'
JSON_PATH = './dataset/arrow_dataset/annotations.json'
OUTPUT_LATENT_DIR = './dataset/arrow_dataset/latents'  # 作成する
IMG_SIZE = 512
BATCH = 8
device = "cuda"

os.makedirs(OUTPUT_LATENT_DIR, exist_ok=True)

vae = AutoencoderKL.from_pretrained(MODEL_ID, subfolder="vae").to(device)
vae.eval()

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

df = pd.read_json(JSON_PATH, encoding='utf-8')

# バッチ処理で高速に latent を生成して保存
def image_batch_generator(filepaths, batch_size):
    for i in range(0, len(filepaths), batch_size):
        batch_paths = filepaths[i:i+batch_size]
        imgs = []
        ids = []
        for p in batch_paths:
            img = Image.open(p).convert("RGB")
            img_t = transform(img)
            imgs.append(img_t)
            ids.append(os.path.basename(p))
        imgs = torch.stack(imgs).to(device)
        yield imgs, ids

filepaths = []
for idx, row in df.iterrows():
    target_fn = f"{row['id']:08d}.jpg"
    filepaths.append(os.path.join(TARGET_DIR, target_fn))

for imgs, ids in tqdm(image_batch_generator(filepaths, BATCH), total=(len(filepaths)//BATCH)+1):
    with torch.no_grad():
        latents = vae.encode(imgs).latent_dist.sample() * vae.config.scaling_factor
    latents = latents.cpu()
    # 各 latent を個別ファイルで保存（または一つの memmap）
    for i, idpath in enumerate(ids):
        # 例: '000123.jpg' -> '000123.pt'
        fname = os.path.splitext(os.path.basename(idpath))[0] + ".pt"
        torch.save(latents[i], os.path.join(OUTPUT_LATENT_DIR, fname))
