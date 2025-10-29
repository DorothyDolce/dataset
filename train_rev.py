import os
import json
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from diffusers import DDPMScheduler, UNet2DConditionModel, ControlNetModel
from transformers import CLIPTextModel, CLIPTokenizer
from accelerate import Accelerator
from accelerate.utils import set_seed


# ============================================================
# Dataset
# ============================================================
class ArrowControlNetDataset(Dataset):
    def __init__(self, df, input_dir, latent_dir, input_transform=None):
        self.df = df
        self.input_dir = input_dir
        self.latent_dir = latent_dir
        self.input_transform = input_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        source_filename = f"{row['source_file']:08d}.jpg"
        input_img_path = os.path.join(self.input_dir, source_filename)
        conditioning_image = Image.open(input_img_path).convert("RGB")
        if self.input_transform:
            conditioning_image = self.input_transform(conditioning_image)
        latent_filename = f"{row['id']:08d}.pt"
        latent_path = os.path.join(self.latent_dir, latent_filename)
        latents = torch.load(latent_path)
        text_input_ids = torch.tensor(row["text_input_ids"], dtype=torch.long)
        return {
            "latents": latents,
            "conditioning_image": conditioning_image,
            "text_input_ids": text_input_ids
        }


# ============================================================
# Training
# ============================================================
def main():
    # --- Paths ---
    INPUT_IMAGE_DIR = "./dataset/arrow_dataset/input_images_openpose"
    LATENT_DIR = "./dataset/arrow_dataset/latents"
    JSON_PATH = "./dataset/arrow_dataset/annotations_with_tokens.json"
    OUTPUT_DIR = "./arrow_controlnet_model_openpose_1"

    # --- Hyperparams ---
    IMG_SIZE = 512
    BATCH_SIZE = 8
    EPOCHS = 20
    LEARNING_RATE = 6e-5
    SEED = 42
    VALIDATION_SPLIT_RATIO = 0.1
    TEST_SPLIT_RATIO = 0.1

    set_seed(SEED)
    accelerator = Accelerator(gradient_accumulation_steps=1, mixed_precision="fp16")

    # --- モデル読み込み ---
    model_id = "runwayml/stable-diffusion-v1-5"
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
    controlnet = ControlNetModel.from_unet(unet)
    noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)

    try:
        unet.enable_attention_slicing()
        controlnet.enable_attention_slicing()
    except Exception:
        pass
    try:
        unet.enable_xformers_memory_efficient_attention()
        controlnet.enable_xformers_memory_efficient_attention()
    except Exception:
        pass

    # --- Dataset / DataLoader ---
    df = pd.read_json(JSON_PATH, encoding="utf-8")
    
    # 1. まず 訓練+検証データ (90%) と テストデータ (10%) に分割
    train_val_df, test_df = train_test_split(
        df, 
        test_size=TEST_SPLIT_RATIO, 
        random_state=SEED, 
        shuffle=True
    )
    
    # 2. 訓練+検証データを 訓練データ と 検証データ に分割
    val_split_ratio_adjusted = VALIDATION_SPLIT_RATIO / (1.0 - TEST_SPLIT_RATIO)
    train_df, val_df = train_test_split(
        train_val_df, 
        test_size=val_split_ratio_adjusted, 
        random_state=SEED, # 乱数を固定
        shuffle=True
    )
    
    print(f"Total data: {len(df)}")
    print(f"Training data: {len(train_df)}")
    print(f"Validation data: {len(val_df)}")
    print(f"Test data: {len(test_df)}") 

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    num_workers = min(8, max(1, os.cpu_count() - 2))

    train_dataset = ArrowControlNetDataset(train_df, INPUT_IMAGE_DIR, LATENT_DIR, transform)
    val_dataset = ArrowControlNetDataset(val_df, INPUT_IMAGE_DIR, LATENT_DIR, transform)
    test_dataset = ArrowControlNetDataset(test_df, INPUT_IMAGE_DIR, LATENT_DIR, transform) 

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=num_workers, pin_memory=True, persistent_workers=True, prefetch_factor=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    optimizer = torch.optim.AdamW(controlnet.parameters(), lr=LEARNING_RATE)

    
    controlnet, unet, text_encoder, optimizer, train_loader, val_loader, test_loader = accelerator.prepare(
        controlnet, unet, text_encoder, optimizer, train_loader, val_loader, test_loader
    )

    # ============================================================
    # Training Loop
    # ============================================================
    train_epoch_losses = []
    val_epoch_losses = []
    
    for epoch in range(EPOCHS):
        
        # --- Training Phase ---
        controlnet.train()
        total_train_loss = 0
        for step, batch in enumerate(train_loader):
            with accelerator.accumulate(controlnet):
                latents = batch["latents"].to(accelerator.device)
                conditioning_images = batch["conditioning_image"].to(accelerator.device)
                text_input_ids = batch["text_input_ids"].to(accelerator.device)
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device
                ).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                encoder_hidden_states = text_encoder(text_input_ids, return_dict=False)[0]

                down_block_res_samples, mid_block_res_sample = controlnet(
                    noisy_latents, timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    controlnet_cond=conditioning_images, return_dict=False
                )
                model_pred = unet(
                    noisy_latents, timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample
                ).sample

                loss = F.mse_loss(model_pred, noise)
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
                total_train_loss += loss.item()
            
            if (step + 1) % 20 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}], Step [{step+1}/{len(train_loader)}], Train Loss: {loss.item():.4f}")

        avg_train_loss = total_train_loss / len(train_loader)
        train_epoch_losses.append(avg_train_loss)
        print(f"Epoch [{epoch+1}/{EPOCHS}] finished. Average Training Loss: {avg_train_loss:.4f}")

        # --- Validation Phase ---
        controlnet.eval()
        total_val_loss = 0
        with torch.no_grad():
            for step, batch in enumerate(val_loader):
                latents = batch["latents"].to(accelerator.device)
                conditioning_images = batch["conditioning_image"].to(accelerator.device)
                text_input_ids = batch["text_input_ids"].to(accelerator.device)
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device
                ).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                encoder_hidden_states = text_encoder(text_input_ids, return_dict=False)[0]

                down_block_res_samples, mid_block_res_sample = controlnet(
                    noisy_latents, timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    controlnet_cond=conditioning_images, return_dict=False
                )
                model_pred = unet(
                    noisy_latents, timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample
                ).sample
                
                loss = F.mse_loss(model_pred, noise)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_epoch_losses.append(avg_val_loss)
        print(f"Epoch [{epoch+1}/{EPOCHS}] finished. Average Validation Loss: {avg_val_loss:.4f}")

    # ============================================================
    # Save Model & Plot Loss
    # ============================================================
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        plt.figure(figsize=(16, 9))
        epochs_range = range(1, EPOCHS + 1)
        
        plt.plot(epochs_range, train_epoch_losses, marker="o", label="Training Loss")
        plt.plot(epochs_range, val_epoch_losses, marker="s", label="Validation Loss")
        
        plt.title('Training and Validation Loss per Epoch')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.grid(True)
        plt.legend()
        plt.xticks(epochs_range)
        
        plot_path = os.path.join(OUTPUT_DIR, 'loss_plot.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"✅ Loss plot saved to {plot_path}")

    unwrapped_controlnet = accelerator.unwrap_model(controlnet)
    unwrapped_controlnet.save_pretrained(OUTPUT_DIR)
    print(f"✅ Model saved to {OUTPUT_DIR}")


    # ============================================================
    # Final Testing Phase
    # ============================================================
    print("\n--- Starting Final Test Evaluation ---")
    controlnet.eval()  # 評価モード
    total_test_loss = 0
    
    with torch.no_grad():  # 勾配計算を無効化
        for step, batch in enumerate(test_loader):
            latents = batch["latents"].to(accelerator.device)
            conditioning_images = batch["conditioning_image"].to(accelerator.device)
            text_input_ids = batch["text_input_ids"].to(accelerator.device)

            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (latents.shape[0],),
                device=latents.device,
            ).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            encoder_hidden_states = text_encoder(text_input_ids, return_dict=False)[0]

            # ControlNet forward
            down_block_res_samples, mid_block_res_sample = controlnet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=encoder_hidden_states,
                controlnet_cond=conditioning_images,
                return_dict=False,
            )

            # UNet forward
            model_pred = unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=encoder_hidden_states,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
            ).sample

            loss = F.mse_loss(model_pred, noise)
            total_test_loss += loss.item()
            
            if (step + 1) % 10 == 0:
                print(f"Test Step [{step+1}/{len(test_loader)}], Test Loss: {loss.item():.4f}")

    # 最終的なテスト損失の平均を計算
    avg_test_loss = total_test_loss / len(test_loader)
    print("--- Test Evaluation Finished ---")
    print(f"✅ Final Average Test Loss: {avg_test_loss:.4f}")


if __name__ == "__main__":
    main()