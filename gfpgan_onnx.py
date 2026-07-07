import os
import urllib.request
import numpy as np
import cv2

# ONNXRuntimeを動的インポート (インストールされていない場合のフォールバック対応)
try:
    import onnxruntime as ort
    ONNXRUNTIME_AVAILABLE = True
except ImportError:
    ONNXRUNTIME_AVAILABLE = False

# MediaPipeの動的インポートと検知
try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

class GFPGANOnnxRunner:
    def __init__(self, model_dir="models"):
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        self.model_path = os.path.join(self.model_dir, "GFPGANv1.4.onnx")
        self.session = None
        self.cache_faces = []
        
        # 高精度顔検出モデルの設定 (小さい顔も検出できるようフルレンジモデルを使用)
        self.detector_config = {
            "filename": "blaze_face_full_range.tflite",
            "url": "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_full_range/float16/latest/blaze_face_full_range.tflite"
        }

    def _blend_restored_face(self, face_input, face_output_raw, fidelity):
        f_val = max(0.0, min(1.0, float(fidelity)))
        if f_val <= 0.0:
            return face_input.copy()

        eye_mask = np.zeros((512, 512), dtype=np.float32)
        cv2.ellipse(eye_mask, (192, 238), (76, 44), 0, 0, 360, 1.0, -1)
        cv2.ellipse(eye_mask, (320, 238), (76, 44), 0, 0, 360, 1.0, -1)
        eye_mask = cv2.GaussianBlur(eye_mask, (55, 55), 0)
        eye_mask = np.expand_dims(eye_mask, axis=2)

        # GFPGANは目元を作り替えやすいため、目だけはスライダー値より強く元画像を残す。
        curve = f_val ** 1.5
        face_strength = min(0.82, curve * 0.82)
        eye_strength = min(0.035, curve * 0.035)
        blend_map = eye_mask * eye_strength + (1.0 - eye_mask) * face_strength

        blended = (
            face_output_raw.astype(np.float32) * blend_map +
            face_input.astype(np.float32) * (1.0 - blend_map)
        )
        return np.clip(blended, 0, 255).astype(np.uint8)

    def _create_face_paste_mask(self, M_inv, width, height, fidelity):
        f_val = max(0.0, min(1.0, float(fidelity)))
        if f_val <= 0.0:
            return np.zeros((height, width, 1), dtype=np.float32)

        mask_512 = np.zeros((512, 512), dtype=np.float32)
        cv2.ellipse(mask_512, (256, 260), (170, 198), 0, 0, 360, 1.0, -1)
        mask_512 = cv2.GaussianBlur(mask_512, (71, 71), 0)

        eye_guard = np.zeros((512, 512), dtype=np.float32)
        cv2.ellipse(eye_guard, (192, 238), (82, 50), 0, 0, 360, 1.0, -1)
        cv2.ellipse(eye_guard, (320, 238), (82, 50), 0, 0, 360, 1.0, -1)
        eye_guard = cv2.GaussianBlur(eye_guard, (59, 59), 0)

        curve = f_val ** 1.5
        paste_strength = min(0.88, curve * 0.88)
        mask_512 *= paste_strength
        mask_512 *= (1.0 - eye_guard * 0.95)

        mask_orig = cv2.warpAffine(mask_512, M_inv, (width, height), flags=cv2.INTER_LINEAR)
        return np.expand_dims(mask_orig, axis=2)

    def check_and_download_model(self, progress_callback=None):
        """モデルが存在しない場合に自動ダウンロードする (複数のミラーURLを試行)"""
        if os.path.exists(self.model_path):
            return True

        if not ONNXRUNTIME_AVAILABLE:
            if progress_callback:
                progress_callback("onnxruntimeが未インストールのため、顔復元はスキップされます。", 1.0)
            return False

        # 複数のダウンロードURLを試行 (安定性重視)
        urls = [
            "https://huggingface.co/Gourieff/ReActor/resolve/main/models/facerestore_models/GFPGANv1.4.onnx",
            "https://huggingface.co/OwlMaster/AllFilesRope/resolve/main/GFPGANv1.4.onnx",
            "https://huggingface.co/Neus/GFPGANv1.4/resolve/main/GFPGANv1.4.onnx",
        ]

        if progress_callback:
            progress_callback("顔復元AIモデル (GFPGAN v1.4 ONNX) のダウンロードを開始します (~340MB)...", 0.0)

        last_error = None
        for i, url in enumerate(urls):
            try:
                if progress_callback:
                    progress_callback(f"ミラー {i+1}/{len(urls)} からダウンロード試行中...", 0.0)

                last_percent = -1
                def report_hook(block_num, block_size, total_size):
                    nonlocal last_percent
                    if progress_callback and total_size > 0:
                        percent = min(100, int(block_num * block_size * 100 / total_size))
                        if percent % 5 == 0 and percent != last_percent:
                            last_percent = percent
                            progress_callback(f"顔復元モデルダウンロード中: {percent}%", percent / 100.0)

                urllib.request.urlretrieve(url, self.model_path, reporthook=report_hook)

                # ダウンロード成功 — ファイルサイズの検証 (最低10MB以上)
                if os.path.getsize(self.model_path) > 10 * 1024 * 1024:
                    if progress_callback:
                        progress_callback("顔復元モデルのダウンロードが完了しました。", 1.0)
                    return True
                else:
                    os.remove(self.model_path)
                    last_error = "ダウンロードされたファイルが小さすぎます"
                    continue

            except Exception as e:
                last_error = str(e)
                # 不完全なファイルを削除
                if os.path.exists(self.model_path):
                    os.remove(self.model_path)
                continue

        if progress_callback:
            progress_callback(f"顔復元モデルのダウンロードに失敗しました: {last_error}", 1.0)
        return False

    def check_and_download_detector(self, progress_callback=None):
        """顔検出モデルが存在しない場合に自動ダウンロードする"""
        det_path = os.path.join(self.model_dir, self.detector_config["filename"])
        if not os.path.exists(det_path):
            url = self.detector_config["url"]
            if progress_callback:
                progress_callback("顔検出モデルのダウンロードを開始します...", 0.0)
            
            last_percent = -1
            def report_hook(block_num, block_size, total_size):
                nonlocal last_percent
                if progress_callback and total_size > 0:
                    percent = min(100, int(block_num * block_size * 100 / total_size))
                    if percent % 5 == 0 and percent != last_percent:
                        last_percent = percent
                        progress_callback(f"顔検出モデルダウンロード中: {percent}%", percent / 100.0)
            
            urllib.request.urlretrieve(url, det_path, reporthook=report_hook)
            if progress_callback:
                progress_callback("顔検出モデルのダウンロードが完了しました。", 1.0)
        return det_path

    def load_model(self, progress_callback=None):
        """ONNXモデルをONNXRuntimeにロードする"""
        if self.session is not None:
            return True

        if not ONNXRUNTIME_AVAILABLE:
            if progress_callback:
                progress_callback("onnxruntimeが未インストールのため、顔復元をスキップします。", 1.0)
            return False

        success = self.check_and_download_model(progress_callback)
        if not success:
            return False

        if progress_callback:
            progress_callback("顔復元AIをGPUにロード中...", 0.9)

        # CUDAExecutionProvider のロード時に cuDNN との不整合でハングするのを防ぐため、CPUプロバイダのみを使用
        providers = ['CPUExecutionProvider']

        
        # セッションオプションの最適化
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        try:
            self.session = ort.InferenceSession(self.model_path, sess_options=opts, providers=providers)
        except Exception as e:
            if progress_callback:
                progress_callback(f"顔復元モデルのロードに失敗しました: {str(e)}", 1.0)
            return False
        
        if progress_callback:
            progress_callback("顔復元AIの初期化が完了しました。", 1.0)
        return True

    def restore_faces(self, img_rgb, fidelity=0.7, progress_callback=None):
        """画像中のすべての顔を検出して高画質化・復元する (目のアライメント対応)"""
        self.cache_faces = []
        # モデルがロードされていない場合はロード試行
        if self.session is None:
            success = self.load_model(progress_callback)
            if not success:
                if progress_callback:
                    progress_callback("顔復元をスキップしました（モデル未対応）。通常の超解像のみ適用されます。", 1.0)
                return img_rgb

        h_orig, w_orig, _ = img_rgb.shape
        
        # MediaPipeで顔検出を実行 (スレッドフリーズを防ぎ、小さい顔も確実に検出するため安定したレガシー版を使用)
        detections = []
        if MEDIAPIPE_AVAILABLE:
            try:
                import mediapipe as mp
                mp_face_detection = mp.solutions.face_detection
                
                # model_selection=1 はフルレンジモデル (遠隔・極小顔用)
                with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detector:
                    results = face_detector.process(img_rgb)
                    raw_detections = results.detections if (results and results.detections) else []
                    
                    min_face_ratio = 0.001  # 画像面積の0.1%以上を占める顔のみ対象（小さい顔も検出可能にする）
                    image_area = h_orig * w_orig
                    filtered = []
                    
                    for det in raw_detections:
                        bbox = det.location_data.relative_bounding_box
                        px_x = int(bbox.xmin * w_orig)
                        px_y = int(bbox.ymin * h_orig)
                        px_w = int(bbox.width * w_orig)
                        px_h = int(bbox.height * h_orig)
                        
                        face_area = px_w * px_h
                        if face_area >= image_area * min_face_ratio:
                            # レガシーのキーポイント ID (0:右目, 1:左目, 2:鼻先, 3:口)
                            keypoints = []
                            if det.location_data.relative_keypoints:
                                for kp in det.location_data.relative_keypoints:
                                    class KeyPoint:
                                        def __init__(self, x, y):
                                            self.x = x
                                            self.y = y
                                    keypoints.append(KeyPoint(kp.x, kp.y))
                            
                            class DummyBBox:
                                def __init__(self, x, y, w, h):
                                    self.origin_x = x
                                    self.origin_y = y
                                    self.width = w
                                    self.height = h
                                    
                            class DummyDetection:
                                def __init__(self, keypoints, bbox_obj, score):
                                    self.keypoints = keypoints
                                    self.bounding_box = bbox_obj
                                    # NMS用のダミーカテゴリ
                                    class DummyCategory:
                                        def __init__(self, score):
                                            self.score = score
                                    self.categories = [DummyCategory(score)]
                            
                            score = det.score[0] if (det.score and len(det.score) > 0) else 0.5
                            filtered.append(DummyDetection(keypoints, DummyBBox(px_x, px_y, px_w, px_h), score))
                    
                    # フィルタリング: 重複検出の除去 (IoUベースの非最大抑制)
                    filtered.sort(key=lambda d: d.categories[0].score, reverse=True)
                    kept = []
                    for det in filtered:
                        bb = det.bounding_box
                        x1, y1 = bb.origin_x, bb.origin_y
                        w1, h1 = bb.width, bb.height
                        
                        is_duplicate = False
                        for kept_det in kept:
                            kb = kept_det.bounding_box
                            kx1, ky1 = kb.origin_x, kb.origin_y
                            kw1, kh1 = kb.width, kb.height
                            
                            # IoU (Intersection over Union) の計算
                            ix1 = max(x1, kx1)
                            iy1 = max(y1, ky1)
                            ix2 = min(x1 + w1, kx1 + kw1)
                            iy2 = min(y1 + h1, ky1 + kh1)
                            
                            if ix2 > ix1 and iy2 > iy1:
                                intersection = (ix2 - ix1) * (iy2 - iy1)
                                union = w1 * h1 + kw1 * kh1 - intersection
                                iou = intersection / union if union > 0 else 0
                                if iou > 0.3:
                                    is_duplicate = True
                                    break
                        
                        if not is_duplicate:
                            kept.append(det)
                    
                    detections = kept
            except Exception as e:
                if progress_callback:
                    progress_callback(f"顔検出の実行中にエラーが発生しました: {str(e)}", 1.0)
                return img_rgb
        else:
            if progress_callback:
                progress_callback("MediaPipeが利用できないため、顔復元をスキップします。", 1.0)
            return img_rgb

        if len(detections) == 0:
            if progress_callback:
                progress_callback("顔は検出されませんでした（通常の超解像のみ適用）", 1.0)
            return img_rgb

        if progress_callback:
            progress_callback(f"{len(detections)}個の顔を検出しました。瞳・顔のアライメント復元中...", 0.5)

        out_img = img_rgb.copy()

        for idx, detection in enumerate(detections):
            try:
                # 左右の目のキーポイントを取得 (通常、keypoints[0]が左目、keypoints[1]が右目)
                # MediaPipeのキーポイント座標は [0.0, 1.0] に正規化されているため、ピクセル座標に戻す
                keypoints = detection.keypoints
                M = None
                
                # 4点相似変換 (左右の目、鼻先、口) を試みる
                if len(keypoints) >= 4:
                    src_pts = np.array([
                        [keypoints[0].x * w_orig, keypoints[0].y * h_orig],  # 右目 (画像左側)
                        [keypoints[1].x * w_orig, keypoints[1].y * h_orig],  # 左目 (画像右側)
                        [keypoints[2].x * w_orig, keypoints[2].y * h_orig],  # 鼻先
                        [keypoints[3].x * w_orig, keypoints[3].y * h_orig]   # 口
                    ], dtype=np.float32)
                    
                    dst_pts = np.array([
                        [192.0, 240.0],  # 512 * 0.375, 512 * 0.47
                        [320.0, 240.0],  # 512 * 0.625, 512 * 0.47
                        [256.0, 300.0],  # 鼻先
                        [256.0, 385.0]   # 口
                    ], dtype=np.float32)
                    
                    M_res = cv2.estimateAffinePartial2D(src_pts, dst_pts)
                    if M_res is not None and M_res[0] is not None:
                        M = M_res[0]

                # 4点相似変換が使えない場合のフォールバック (従来の2点アライメント)
                if M is None and len(keypoints) >= 2:
                    left_eye_pt = np.array([keypoints[0].x * w_orig, keypoints[0].y * h_orig])
                    right_eye_pt = np.array([keypoints[1].x * w_orig, keypoints[1].y * h_orig])

                    # 左右の目の角度と距離を計算
                    dy = right_eye_pt[1] - left_eye_pt[1]
                    dx = right_eye_pt[0] - left_eye_pt[0]
                    angle = np.degrees(np.arctan2(dy, dx))

                    # アライメント位置の設定
                    desired_left_x = 0.375
                    desired_right_x = 0.625
                    desired_y = 0.45
                    desired_dist = desired_right_x - desired_left_x

                    eye_dist = np.sqrt(dx**2 + dy**2)
                    if eye_dist >= 1e-5:
                        desired_eye_dist_px = 512 * desired_dist
                        scale = desired_eye_dist_px / eye_dist

                        eye_center = (left_eye_pt + right_eye_pt) / 2.0
                        desired_center = (512 * 0.5, 512 * desired_y)

                        M = cv2.getRotationMatrix2D(tuple(eye_center), angle, scale)
                        M[0, 2] = desired_center[0] - (M[0, 0] * eye_center[0] + M[0, 1] * eye_center[1])
                        M[1, 2] = desired_center[1] - (M[1, 0] * eye_center[0] + M[1, 1] * eye_center[1])

                if M is None:
                    continue

                # アライメント（まっすぐ前を向き、目の位置が完全に揃った512x512の画像）を生成
                face_input = cv2.warpAffine(img_rgb, M, (512, 512), flags=cv2.INTER_LANCZOS4)

                # 前処理: RGB [0, 255] -> [-1, 1], CHW
                face_input_norm = face_input.astype(np.float32) / 127.5 - 1.0
                face_input_norm = np.transpose(face_input_norm, (2, 0, 1))
                face_input_norm = np.expand_dims(face_input_norm, axis=0)

                # GFPGANによる超画質化推論
                input_name = self.session.get_inputs()[0].name
                outputs = self.session.run(None, {input_name: face_input_norm})
                face_output = outputs[0][0]

                # 後処理: [-1, 1] -> [0, 255] np.uint8, HWC
                face_output = np.transpose(face_output, (1, 2, 0))
                face_output = ((face_output + 1.0) * 127.5).clip(0, 255).astype(np.uint8)

                # キャッシュの保存 (再合成用にM, face_input, face_output_rawを保存)
                self.cache_faces.append({
                    "M": M.copy(),
                    "face_input": face_input.copy(),
                    "face_output_raw": face_output.copy()
                })

                face_output = self._blend_restored_face(face_input, face_output, fidelity)

                # 逆アフィン変換で元の傾き・スケール・位置に正確に戻す
                M_inv = cv2.invertAffineTransform(M)
                restored_face = cv2.warpAffine(face_output, M_inv, (w_orig, h_orig), flags=cv2.INTER_LANCZOS4)

                mask_orig = self._create_face_paste_mask(M_inv, w_orig, h_orig, fidelity)

                # 元の画像にマスクブレンド合成
                out_img = (restored_face.astype(np.float32) * mask_orig + 
                           out_img.astype(np.float32) * (1.0 - mask_orig)).astype(np.uint8)
            except Exception as e:
                if progress_callback:
                    progress_callback(f"顔 {idx+1} のアライメント復元に失敗しました: {str(e)}", 0.5)
                continue

        if progress_callback:
            progress_callback(f"{len(detections)}個の顔の精密復元が完了しました。", 1.0)

        return out_img

    def recomposite_faces(self, img_rgb, fidelity=0.7):
        """検出・推論を介さず、キャッシュされた顔データと指定されたfidelityに基づいて顔復元処理を高速に再合成する"""
        if len(self.cache_faces) == 0:
            return img_rgb

        h_orig, w_orig, _ = img_rgb.shape
        out_img = img_rgb.copy()

        for face in self.cache_faces:
            try:
                M = face["M"]
                face_input = face["face_input"]
                face_output_raw = face["face_output_raw"]

                face_output = self._blend_restored_face(face_input, face_output_raw, fidelity)

                # 逆アフィン変換で元の傾き・スケール・位置に正確に戻す
                M_inv = cv2.invertAffineTransform(M)
                restored_face = cv2.warpAffine(face_output, M_inv, (w_orig, h_orig), flags=cv2.INTER_LANCZOS4)

                mask_orig = self._create_face_paste_mask(M_inv, w_orig, h_orig, fidelity)

                # 元の画像にマスクブレンド合成
                out_img = (restored_face.astype(np.float32) * mask_orig + 
                           out_img.astype(np.float32) * (1.0 - mask_orig)).astype(np.uint8)
            except Exception:
                continue

        return out_img
