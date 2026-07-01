import sys
import os
import shutil
import cv2

# 元画像のコピー
src_image = r"C:\Users\nakac\.gemini\antigravity-ide\brain\15ca7026-2c99-44f8-8488-c3ae537fb43c\media__1781937703782.jpg"
dst_image = r"test_image.jpg"

if os.path.exists(src_image) and not os.path.exists(dst_image):
    try:
        shutil.copy(src_image, dst_image)
        print("Copied test image to workspace.")
    except Exception as e:
        print(f"Failed to copy image: {e}")

from gfpgan_onnx import GFPGANOnnxRunner

def main():
    print("Starting face detection test...")
    
    if not os.path.exists(dst_image):
        print(f"Error: Test image not found at {dst_image}")
        return

    # 画像の読み込み
    img_bgr = cv2.imread(dst_image)
    if img_bgr is None:
        print("Error: Failed to load image with OpenCV")
        return
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    print(f"Loaded image size: {img_rgb.shape}")

    # GFPGAN Runnerの初期化
    print("Initializing GFPGANOnnxRunner...")
    runner = GFPGANOnnxRunner(model_dir=r"models")
    
    def progress_cb(msg, pct):
        print(f"[Progress] {msg} ({pct*100:.1f}%)")

    # 顔復元のテスト実行
    print("Running restore_faces...")
    try:
        out_img = runner.restore_faces(img_rgb, fidelity=0.7, progress_callback=progress_cb)
        print("restore_faces completed successfully!")
    except Exception as e:
        print(f"Error during restore_faces: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
