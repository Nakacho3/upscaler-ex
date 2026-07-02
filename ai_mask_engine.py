import os
import cv2
import numpy as np
from PIL import Image

# MediaPipeの動的インポート (利用可能かどうかのフラグのみ設定)
MEDIAPIPE_AVAILABLE = False
try:
    import mediapipe
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    pass

class AIMaskEngine:
    def __init__(self, model_dir="models"):
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        
        # カテゴリごとにプレビュー表示用の色を定義 (RGB)
        self.category_colors = {
            "Person": (239, 68, 68),      # 赤
            "Face": (34, 197, 94),        # 緑
            "Object": (168, 85, 247),     # 紫
            "Text": (249, 115, 22),       # オレンジ
            "Sky": (14, 165, 233),        # 水色
            "Background": (100, 116, 139) # グレー
        }

    def generate_masks(self, img_rgb, confidence_threshold=0.5, progress_callback=None):
        """画像からカテゴリ別のマスク画像 (0 or 255) と信頼度スコアを生成する"""
        if not MEDIAPIPE_AVAILABLE:
            raise RuntimeError("MediaPipeライブラリがインストールされていません。pip install mediapipe を実行してください。")

        # MediaPipe Tasks API (v0.10.35+) の遅延インポート
        from mediapipe.tasks.python.vision import image_segmenter as mp_seg_module
        from mediapipe.tasks.python.vision import face_detector as mp_face_module
        from mediapipe.tasks.python.core import base_options as base_options_module
        from mediapipe.tasks.python.vision.core.image import Image as MpImage
        from mediapipe.tasks.python.vision.core.image import ImageFormat as MpImageFormat

        h, w, _ = img_rgb.shape
        masks = {}
        confidences = {}

        # デフォルトで全マスクを空（0）で初期化
        for cat in self.category_colors.keys():
            masks[cat] = np.zeros((h, w), dtype=np.uint8)
            confidences[cat] = 0.0

        if progress_callback:
            progress_callback("人物領域を解析中...", 0.2)

        # --- 1. 人物 (Person) ＆ 背景 (Background) の抽出 ---
        person_mask = np.zeros((h, w), dtype=np.uint8)
        person_score = 0.0
        segmenter_model = os.path.join(self.model_dir, "selfie_segmenter.tflite")
        
        if os.path.exists(segmenter_model):
            try:
                base_options = base_options_module.BaseOptions(model_asset_path=segmenter_model)
                options = mp_seg_module.ImageSegmenterOptions(
                    base_options=base_options,
                    output_confidence_masks=True,
                    output_category_mask=False
                )
                segmenter = mp_seg_module.ImageSegmenter.create_from_options(options)
                
                # numpy配列からMediaPipe Imageを作成 (RGB, uint8)
                mp_image = MpImage(MpImageFormat.SRGB, img_rgb.astype(np.uint8))
                result = segmenter.segment(mp_image)
                
                if result.confidence_masks and len(result.confidence_masks) > 0:
                    # selfie_segmenter は 1チャンネルの信頼度マスクを返す
                    raw_mask_img = result.confidence_masks[0]
                    raw_mask = np.copy(raw_mask_img.numpy_view())
                    # VEC32F1 の場合、shape は (h, w, 1) なので squeeze
                    if raw_mask.ndim == 3:
                        raw_mask = raw_mask[:, :, 0]
                    person_mask = (raw_mask > confidence_threshold).astype(np.uint8) * 255
                    person_score = float(np.mean(raw_mask[raw_mask > confidence_threshold])) if np.any(raw_mask > confidence_threshold) else 0.0
                
                segmenter.close()
            except Exception as e:
                raise RuntimeError(f"人物セグメンテーション (ImageSegmenter) の実行中にエラーが発生しました: {e}")
        else:
            print(f"[AI Mask] セグメンテーションモデルが見つかりません: {segmenter_model}")

        masks["Person"] = person_mask
        confidences["Person"] = person_score

        # 背景は人物の反転
        masks["Background"] = cv2.bitwise_not(person_mask)
        confidences["Background"] = 1.0 - person_score if person_score > 0.0 else 0.8

        if progress_callback:
            progress_callback("顔領域を検出中...", 0.4)

        # --- 2. 顔 (Face) の抽出 ---
        face_mask = np.zeros((h, w), dtype=np.uint8)
        face_score = 0.0
        # full_range モデルを優先、なければ short_range を使用
        face_model = os.path.join(self.model_dir, "blaze_face_full_range.tflite")
        if not os.path.exists(face_model):
            face_model = os.path.join(self.model_dir, "blaze_face_short_range.tflite")
        
        if os.path.exists(face_model):
            try:
                base_options = base_options_module.BaseOptions(model_asset_path=face_model)
                options = mp_face_module.FaceDetectorOptions(
                    base_options=base_options,
                    min_detection_confidence=confidence_threshold * 0.8
                )
                detector = mp_face_module.FaceDetector.create_from_options(options)
                
                mp_image = MpImage(MpImageFormat.SRGB, img_rgb.astype(np.uint8))
                result = detector.detect(mp_image)
                
                if result.detections:
                    scores = []
                    for det in result.detections:
                        bbox = det.bounding_box
                        px_x = max(0, bbox.origin_x)
                        px_y = max(0, bbox.origin_y)
                        px_w = min(w - px_x, bbox.width)
                        px_h = min(h - px_y, bbox.height)
                        
                        # 顔領域を楕円でマスク化
                        cv2.ellipse(
                            face_mask,
                            (px_x + px_w // 2, px_y + px_h // 2),
                            (px_w // 2, px_h // 2),
                            0, 0, 360, 255, -1
                        )
                        # 各検出のスコア
                        if det.categories:
                            scores.append(det.categories[0].score)
                    face_score = float(np.mean(scores)) if scores else 0.0
                
                detector.close()
            except Exception as e:
                raise RuntimeError(f"顔検出 (FaceDetector) の実行中にエラーが発生しました: {e}")
        else:
            print(f"[AI Mask] 顔検出モデルが見つかりません: {face_model}")

        masks["Face"] = face_mask
        confidences["Face"] = face_score

        if progress_callback:
            progress_callback("テキスト領域は手動編集を推奨しています...", 0.6)

        # 3. テキスト (Text) は自動検出しない
        text_mask = np.zeros((h, w), dtype=np.uint8)
        text_score = 0.0
        masks["Text"] = text_mask
        confidences["Text"] = text_score

        if progress_callback:
            progress_callback("空と環境領域を解析中...", 0.8)

        # 4. 空 (Sky) の抽出 (HSV色空間による青空/曇り空の簡易色相・輝度抽出)
        sky_mask = np.zeros((h, w), dtype=np.uint8)
        sky_score = 0.0
        try:
            hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
            # 青空のレンジ
            lower_blue = np.array([90, 30, 50])
            upper_blue = np.array([140, 255, 255])
            mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
            
            # 曇り空/白い空のレンジ (上部にあり、彩度が低く輝度が高い領域)
            lower_white = np.array([0, 0, 180])
            upper_white = np.array([180, 40, 255])
            mask_white = cv2.inRange(hsv, lower_white, upper_white)
            
            combined_sky = cv2.bitwise_or(mask_blue, mask_white)
            # 空は通常画像の上半分以上に存在する傾向があるため、下部をグラデーション等でカット
            y_indices = np.arange(h).reshape(h, 1)
            sky_prior = (y_indices < h * 0.6).astype(np.uint8) * 255
            sky_mask = cv2.bitwise_and(combined_sky, sky_prior)
            
            # ノイズ除去
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            sky_mask = cv2.morphologyEx(sky_mask, cv2.MORPH_CLOSE, kernel)
            sky_mask = cv2.morphologyEx(sky_mask, cv2.MORPH_OPEN, kernel)
            
            if np.any(sky_mask > 0):
                sky_score = 0.90
        except Exception as e:
            print(f"Sky detection error: {e}")

        masks["Sky"] = sky_mask
        confidences["Sky"] = sky_score

        if progress_callback:
            progress_callback("その他のオブジェクトを抽出中...", 0.95)

        # 5. オブジェクト (Object) の抽出 (前景で人物やテキスト、空以外の主要なエッジ/輪郭領域)
        object_mask = np.zeros((h, w), dtype=np.uint8)
        object_score = 0.0
        try:
            # 顕著性(Saliency)または輪郭に基づく簡易抽出
            gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
            dilated = cv2.dilate(edges, kernel, iterations=2)
            
            # 輪郭を見つける
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > (h * w * 0.01): # 画像面積の1%以上のサイズを持つものをオブジェクトとする
                    cv2.drawContours(object_mask, [cnt], -1, 255, -1)
            
            # 人物、空、テキスト領域をオブジェクトから除外する
            exclude_mask = cv2.bitwise_or(person_mask, sky_mask)
            exclude_mask = cv2.bitwise_or(exclude_mask, text_mask)
            object_mask = cv2.bitwise_and(object_mask, cv2.bitwise_not(exclude_mask))
            
            if np.any(object_mask > 0):
                object_score = 0.75
        except Exception as e:
            print(f"Object detection error: {e}")

        masks["Object"] = object_mask
        confidences["Object"] = object_score

        if progress_callback:
            progress_callback("解析完了しました。", 1.0)

        return masks, confidences

    def create_overlay(self, img_rgb, masks, enabled_categories, opacity=0.4):
        """有効なカテゴリのマスクを半透明の色で重ね合わせた画像を生成する"""
        h, w, _ = img_rgb.shape
        overlay = img_rgb.copy().astype(np.float32)

        for cat, is_enabled in enabled_categories.items():
            if not is_enabled:
                continue
            if cat not in masks or cat not in self.category_colors:
                continue
            
            mask = masks[cat]
            if np.any(mask > 0):
                color = self.category_colors[cat]
                # マスク領域(255)に対してブレンド
                mask_norm = (mask / 255.0)[:, :, np.newaxis]
                color_img = np.full((h, w, 3), color, dtype=np.float32)
                
                # 元の画像とカラー画像をブレンド
                overlay = (1.0 - mask_norm * opacity) * overlay + (mask_norm * opacity) * color_img

        return Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))
