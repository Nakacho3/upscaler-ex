import sys
import os

from upscaler import ImageUpscalerEngine
from PIL import Image

def main():
    # model_dir をプロジェクト直下の models に設定
    engine = ImageUpscalerEngine(model_dir="models")
    
    # テスト画像パス
    image_path = r"C:\Users\nakac\.gemini\antigravity-ide\brain\15ca7026-2c99-44f8-8488-c3ae537fb43c\media__1781937703782.jpg"
    
    print("モデルのロードを開始します...")
    engine.load_model(progress_callback=lambda msg, pct: print(f"[{pct*100:.1f}%] {msg}"))
    
    print(f"画像の処理を開始します: {image_path}")
    # 4倍拡大で処理を実行
    out_img = engine.process_image(
        image_path,
        scale=4,
        denoise_strength=10,
        sharpness_strength=15,
        face_restoration=True,
        face_restoration_fidelity=0.7,
        is_creature=True,
        progress_callback=lambda msg, pct: print(f"[{pct*100:.1f}%] {msg}")
    )
    
    # ワークスペース内に結果を保存
    out_path = "upscaled_result.png"
    out_img.save(out_path)
    print(f"処理が完了しました。保存先: {out_path}")

if __name__ == "__main__":
    main()
