import os
import torch
from PIL import Image
from controlnet_aux import OpenposeDetector
from tqdm import tqdm # 進捗表示ライブラリ

def preprocess_directory(input_dir, output_dir):
    """
    指定されたディレクトリ内のすべての画像をOpenPose処理し、
    結果を出力ディレクトリに保存する。
    """
    
    # 1. GPUが利用可能か確認 (利用可能な場合は高速)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 2. OpenposeDetectorモデルをロード
    print("Loading OpenposeDetector model...")
    # controlnet_aux 0.0.7以降は "lllyasviel/Annotators" が推奨されます
    try:
        openpose = OpenposeDetector.from_pretrained("lllyasviel/Annotators").to(device)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure 'controlnet_aux' is installed (`pip install controlnet-aux`)")
        return
    print("Model loaded successfully.")

    # 3. 出力ディレクトリを作成
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory ensured at: {output_dir}")

    # 4. 入力ディレクトリ内の画像ファイルを取得
    supported_extensions = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')
    try:
        image_files = [
            f for f in os.listdir(input_dir) 
            if f.lower().endswith(supported_extensions)
        ]
    except FileNotFoundError:
        print(f"Error: Input directory not found at {input_dir}")
        return
        
    if not image_files:
        print(f"No images found in {input_dir}")
        return
        
    print(f"Found {len(image_files)} images to process.")

    # 5. 1枚ずつ処理して保存 (tqdmで進捗バーを表示)
    for filename in tqdm(image_files, desc="Processing images"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename) # 同じファイル名で保存

        try:
            # 元画像を読み込み
            original_image = Image.open(input_path).convert("RGB")
            
            # OpenPose処理を実行 (PIL -> PIL)
            processed_image_pil = openpose(original_image)
            
            # 結果を保存
            processed_image_pil.save(output_path)
            
        except Exception as e:
            print(f"\n[Warning] Failed to process {filename}: {e}")
            # エラーが発生しても他の画像の処理を続行

    print("\n--- Pre-processing Finished ---")
    print(f"All processed images saved to: {output_dir}")


# ============================================================
# メイン実行ブロック
# ============================================================
if __name__ == "__main__":
    
    # --- ★ ユーザーが設定する箇所 ★ ---
    
    # 元の画像（スキーの画像など）が保存されているディレクトリ
    INPUT_IMAGE_DIR = "./dataset/arrow_dataset/input_images"
    
    # OpenPose処理後の画像を保存したいディレクトリ
    OUTPUT_POSE_DIR = "./dataset/arrow_dataset/input_images_openpose"
    
    # --- ★ 設定はここまで ★ ---
    
    preprocess_directory(INPUT_IMAGE_DIR, OUTPUT_POSE_DIR)