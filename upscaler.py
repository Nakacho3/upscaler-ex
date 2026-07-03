import os
import urllib.request
import torch
torch.backends.cudnn.benchmark = False
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image

# CuPyの動的インポートと検知 (PyTorchとのCUDA競合によるGPUフリーズを防ぐため無効化)
CUPY_AVAILABLE = False

# TensorRTの動的インポートと検知 (マルチスレッドでのGPUフリーズ回避のため無効化)
try:
    import tensorrt as trt
    TENSORRT_AVAILABLE = False
except ImportError:
    TENSORRT_AVAILABLE = False

# MediaPipe (TFLiteタスク) の動的インポートと検知
try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

# ==========================================
# 1. Real-ESRGAN (RRDBNet) モデルの定義
# ==========================================
class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x

class RRDB(nn.Module):
    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.rdb1 = ResidualDenseBlock_5C(nf, gc)
        self.rdb2 = ResidualDenseBlock_5C(nf, gc)
        self.rdb3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        return self.rdb3(self.rdb2(self.rdb1(x))) * 0.2 + x

class RRDBNet(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=23, gc=32, scale=4):
        super(RRDBNet, self).__init__()
        self.scale = scale
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.body = nn.Sequential(*[RRDB(nf=nf, gc=gc) for _ in range(nb)])
        self.conv_body = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_up1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        if self.scale == 4:
            self.conv_up2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_hr = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.conv_body(self.body(fea))
        fea = fea + trunk

        fea = self.lrelu(self.conv_up1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        if self.scale == 4:
            fea = self.lrelu(self.conv_up2(F.interpolate(fea, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.conv_hr(fea)))
        return out


# ==========================================
# 2. 画像処理エンジンクラス
# ==========================================
class ImageUpscalerEngine:
    def __init__(self, model_dir="models"):
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # モデルごとの定義
        self.model_configs = {
            "Standard (高画質)": {
                "filename": "RealESRGAN_x4plus.pth",
                "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
                "nb": 23
            },
            "Fast (高速)": {
                "filename": "RealESRGAN_x4plus_anime_6B.pth",
                "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
                "nb": 6
            }
        }
        
        # 背景シャープネス低減用セグメンテーションモデルの設定
        self.segmenter_config = {
            "filename": "selfie_segmenter.tflite",
            "url": "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite"
        }
        
        self.current_model_name = "Standard (高画質)"  # 初期デフォルトを高画質版にする
        self.model = None
        self.trt_engine = None
        self.trt_context = None
        self.face_runner = None
        
        # 再合成用の中間キャッシュ
        self.cache_upscaled_base = None
        self.cache_faces = []
        self.cache_is_creature = False
        self.cache_mask_low = None

    def set_model(self, model_name):
        """モデルを変更する"""
        if model_name in self.model_configs and model_name != self.current_model_name:
            self.current_model_name = model_name
            self.model = None
            self.trt_engine = None
            self.trt_context = None

    def get_paths(self):
        cfg = self.model_configs[self.current_model_name]
        filename = cfg["filename"]
        base = os.path.splitext(filename)[0]
        
        model_path = os.path.join(self.model_dir, filename)
        onnx_path = os.path.join(self.model_dir, f"{base}.onnx")
        engine_path = os.path.join(self.model_dir, f"{base}.engine")
        return model_path, onnx_path, engine_path

    def check_and_download_model(self, progress_callback=None):
        """モデルが存在しない場合に自動ダウンロードする"""
        model_path, _, _ = self.get_paths()
        cfg = self.model_configs[self.current_model_name]
        
        # メインAIモデルのダウンロード
        if not os.path.exists(model_path):
            url = cfg["url"]
            if progress_callback:
                progress_callback(f"AIモデル ({self.current_model_name}) のダウンロードを開始します...", 0.0)
            
            last_percent = -1
            def report_hook(block_num, block_size, total_size):
                nonlocal last_percent
                if progress_callback and total_size > 0:
                    percent = min(100, int(block_num * block_size * 100 / total_size))
                    if percent % 5 == 0 and percent != last_percent:
                        last_percent = percent
                        progress_callback(f"モデルダウンロード中: {percent}%", percent / 100.0)
            
            urllib.request.urlretrieve(url, model_path, reporthook=report_hook)
            if progress_callback:
                progress_callback("ダウンロードが完了しました。", 1.0)

    def check_and_download_segmenter(self, progress_callback=None):
        """セグメンテーションモデルを自動ダウンロードする"""
        seg_path = os.path.join(self.model_dir, self.segmenter_config["filename"])
        if not os.path.exists(seg_path):
            url = self.segmenter_config["url"]
            if progress_callback:
                progress_callback("背景分離AIモデルのダウンロードを開始します...", 0.0)
            
            last_percent = -1
            def report_hook(block_num, block_size, total_size):
                nonlocal last_percent
                if progress_callback and total_size > 0:
                    percent = min(100, int(block_num * block_size * 100 / total_size))
                    if percent % 5 == 0 and percent != last_percent:
                        last_percent = percent
                        progress_callback(f"背景分離モデルダウンロード中: {percent}%", percent / 100.0)
            
            urllib.request.urlretrieve(url, seg_path, reporthook=report_hook)
            if progress_callback:
                progress_callback("背景分離モデルのダウンロードが完了しました。", 1.0)

    def load_model(self, progress_callback=None):
        """モデルを読み込む (TensorRTエンジン、またはPyTorch)"""
        self.check_and_download_model(progress_callback)
        model_path, onnx_path, engine_path = self.get_paths()
        cfg = self.model_configs[self.current_model_name]
        nb = cfg["nb"]

        # 1. TensorRTの構築と読み込みの試行 (利用可能な場合)
        if TENSORRT_AVAILABLE and torch.cuda.is_available():
            try:
                if not os.path.exists(engine_path):
                    if progress_callback:
                        progress_callback(f"TensorRTエンジンの初回構築を行っています ({self.current_model_name}、数分かかる場合があります)...", 0.1)
                    self._build_tensorrt_engine(model_path, onnx_path, engine_path, nb, progress_callback)
                
                self._load_tensorrt_engine(engine_path)
                if progress_callback:
                    progress_callback(f"TensorRTエンジン (FP16) を読み込みました。[{self.current_model_name}]", 1.0)
                return "TensorRT"
            except Exception as e:
                if progress_callback:
                    progress_callback(f"TensorRT構築失敗。通常のPyTorch CUDAにフォールバックします: {str(e)}", 0.5)

        # 2. PyTorch (CUDA または CPU) の読み込み
        if progress_callback:
            progress_callback(f"PyTorchモデルをロード中 ({self.device})...", 0.7)
        
        self.model = RRDBNet(in_nc=3, out_nc=3, nf=64, nb=nb, gc=32, scale=4)
        loadnet = torch.load(model_path, map_location=self.device)
        
        keyname = 'params_ema' if 'params_ema' in loadnet else ('params' if 'params' in loadnet else None)
        state_dict = loadnet[keyname] if keyname else loadnet
            
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()
        self.model.to(self.device)
        
        if progress_callback:
            progress_callback(f"PyTorchモデルを読み込みました (デバイス: {self.device})", 1.0)
        return f"PyTorch ({self.device})"

    def _build_tensorrt_engine(self, model_path, onnx_path, engine_path, nb, progress_callback=None):
        """PyTorchモデルをONNX経由でTensorRTエンジンに変換する"""
        if progress_callback:
            progress_callback("ONNXモデルへ変換中...", 0.2)
            
        # まずPyTorchモデルを読み込む
        pytorch_model = RRDBNet(in_nc=3, out_nc=3, nf=64, nb=nb, gc=32, scale=4)
        loadnet = torch.load(model_path, map_location="cpu")
        keyname = 'params_ema' if 'params_ema' in loadnet else ('params' if 'params' in loadnet else None)
        state_dict = loadnet[keyname] if keyname else loadnet
        pytorch_model.load_state_dict(state_dict, strict=True)
        pytorch_model.eval()

        # ONNXへエクスポート (動的解像度対応のため、動的入力シェイプを定義)
        dummy_input = torch.randn(1, 3, 256, 256)
        torch.onnx.export(
            pytorch_model,
            dummy_input,
            onnx_path,
            opset_version=14,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {2: 'height', 3: 'width'},
                'output': {2: 'height', 3: 'width'}
            }
        )

        if progress_callback:
            progress_callback("TensorRTビルダーを起動して最適化中...", 0.4)

        # TensorRTビルダーとネットワークの作成
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        config = builder.create_builder_config()
        
        # ワークスペースサイズ設定 (2GB)
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 * 1024 * 1024 * 1024)
        
        # FP16精度を有効化してTensorコアを最大活用
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        
        flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network_v2(flag)
        parser = trt.OnnxParser(network, logger)
        
        with open(onnx_path, 'rb') as model_file:
            if not parser.parse(model_file.read()):
                raise RuntimeError("ONNXファイルのパースに失敗しました。")

        # 動的シェイプ（プロファイル）の設定
        profile = builder.create_optimization_profile()
        profile.set_shape('input', (1, 3, 128, 128), (1, 3, 512, 512), (1, 3, 1024, 1024))
        config.add_optimization_profile(profile)

        if progress_callback:
            progress_callback("TensorRT FP16 カーネル最適化エンジンをコンパイル中...", 0.6)

        # エンジンのビルドと保存
        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            raise RuntimeError("TensorRTエンジンのシリアライズに失敗しました。")
            
        with open(engine_path, 'wb') as f:
            f.write(serialized_engine)

        # 一時ONNXファイルの削除
        if os.path.exists(onnx_path):
            os.remove(onnx_path)

    def _load_tensorrt_engine(self, engine_path):
        """ビルド済みTensorRTエンジンのロード"""
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(engine_path, 'rb') as f:
            self.trt_engine = runtime.deserialize_cuda_engine(f.read())
        self.trt_context = self.trt_engine.create_execution_context()

    # ==========================================
    # 3. フィルタ処理 (CuPy / OpenCV)
    # ==========================================
    def apply_denoise(self, img_np, strength=10):
        """デノイズ処理の適用 (CuPy優先、OpenCVフォールバック)"""
        if strength <= 0:
            return img_np

        if CUPY_AVAILABLE:
            try:
                img_cp = cp.array(img_np)
                img_out = cp.zeros_like(img_cp)
                for i in range(3):
                    img_out[:, :, i] = cp_ndimage.median_filter(img_cp[:, :, i], size=3)
                
                alpha = strength / 100.0
                result_cp = cp.clip(img_cp * (1.0 - alpha) + img_out * alpha, 0, 255)
                return cp.asnumpy(result_cp).astype(np.uint8)
            except Exception:
                pass
        
        d = max(3, int(strength * 0.15))
        sigma_color = strength * 1.5
        sigma_space = strength * 1.5
        return cv2.bilateralFilter(img_np, d, sigma_color, sigma_space)

    def apply_sharpness(self, img_np, strength_percent=30):
        """シャープネスの適用 (CuPy優先、OpenCVフォールバック)"""
        if strength_percent <= 0:
            return img_np

        strength = strength_percent / 100.0

        if CUPY_AVAILABLE:
            try:
                img_cp = cp.array(img_np, dtype=cp.float32)
                sharp_channels = []
                for i in range(3):
                    c = img_cp[:, :, i]
                    blurred = cp_ndimage.gaussian_filter(c, sigma=1.5)
                    sharp = c + (c - blurred) * strength * 2.0
                    sharp_channels.append(sharp)
                img_out = cp.stack(sharp_channels, axis=2)
                return cp.asnumpy(cp.clip(img_out, 0, 255)).astype(np.uint8)
            except Exception:
                pass

        img_float = img_np.astype(np.float32)
        blurred = cv2.GaussianBlur(img_float, (0, 0), 1.5)
        sharp = img_float + (img_float - blurred) * strength * 2.0
        return np.clip(sharp, 0, 255).astype(np.uint8)

    # ==========================================
    # 4. アップスケーリングメイン処理
    # ==========================================
    def _build_region_mask_low(self, region_masks, width, height):
        if not region_masks:
            return None

        combined = np.zeros((height, width), dtype=np.uint8)
        for mask in region_masks.values():
            if mask is None:
                continue
            mask_np = np.asarray(mask)
            if mask_np.ndim == 3:
                mask_np = mask_np[:, :, 0]
            if mask_np.shape[:2] != (height, width):
                mask_np = cv2.resize(mask_np.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
            combined = cv2.bitwise_or(combined, (mask_np > 0).astype(np.uint8) * 255)

        if not np.any(combined > 0):
            return None

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
        return (combined > 0).astype(np.float32)

    def process_image(self, input_image_path, scale=4, denoise_strength=10, sharpness_strength=30, face_restoration=False, face_restoration_fidelity=0.7, is_creature=False, region_masks=None, progress_callback=None):
        """画像処理の全体フローを実行する"""
        if progress_callback:
            progress_callback("画像を読み込んでいます...", 0.05)
            
        # 日本語パス対応 of ファイル読み込み
        try:
            img_array = np.fromfile(input_image_path, dtype=np.uint8)
            img_cv = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        except Exception as e:
            raise ValueError(f"画像ファイルを読み込めませんでした: {input_image_path} ({str(e)})")
            
        if img_cv is None:
            raise ValueError(f"画像ファイルをデコードできませんでした: {input_image_path}")
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        h, w, c = img_cv.shape

        # 0. AIマスクタブの編集済みマスクを優先し、なければ従来の背景分離マスクを作成
        mask_low = None
        if region_masks:
            if progress_callback:
                progress_callback("編集済みマスクをアップスケール処理へ適用中...", 0.15)
            mask_low = self._build_region_mask_low(region_masks, w, h)

        if mask_low is None and is_creature and MEDIAPIPE_AVAILABLE:
            try:
                if progress_callback:
                    progress_callback("被写体（人物・動物）の領域を解析中...", 0.15)
                
                # レガシーな mp.solutions.selfie_segmentation を使用してハングを完全に防止
                import mediapipe as mp
                mp_selfie_segmentation = mp.solutions.selfie_segmentation
                
                # model_selection=0 は一般的なセグメンテーションモデル
                with mp_selfie_segmentation.SelfieSegmentation(model_selection=0) as segmenter:
                    results = segmenter.process(img_cv)
                    if results.segmentation_mask is not None:
                        # 前景確率0.5以上の領域を被写体領域とする
                        mask_low = (results.segmentation_mask > 0.5).astype(np.float32)
            except Exception as e:
                if progress_callback:
                    progress_callback(f"領域解析をスキップしました (エラー: {str(e)})", 0.18)

        # 1. デノイズ処理 (前処理)
        if denoise_strength > 0:
            if progress_callback:
                progress_callback("デノイズ（ノイズ除去）を適用中...", 0.2)
            img_cv = self.apply_denoise(img_cv, denoise_strength)

        # 2. AIアップスケーリング (Real-ESRGAN)
        if progress_callback:
            progress_callback("AIアップスケーリング（超解像）を実行中...", 0.4)

        upscaled_img = None
        
        # TensorRTがロードされている場合の推論
        if self.trt_context is not None and torch.cuda.is_available():
            try:
                upscaled_img = self._inference_tensorrt(img_cv)
            except Exception as e:
                if progress_callback:
                    progress_callback(f"TensorRT推論エラー。PyTorchで再試行します: {str(e)}", 0.5)

        # PyTorch (またはTensorRT失敗時) の推論
        if upscaled_img is None:
            if self.model is None:
                self.load_model(lambda msg, pct: progress_callback(msg, 0.4 + pct * 0.4) if progress_callback else None)
            upscaled_img = self._inference_pytorch(img_cv)

        # 3. 倍率の調整
        if scale == 2:
            if progress_callback:
                progress_callback("指定倍率 (2倍) に縮小調整中...", 0.8)
            h_new, w_new = h * 2, w * 2
            upscaled_img = cv2.resize(upscaled_img, (w_new, h_new), interpolation=cv2.INTER_CUBIC)

        # 中間ベース画像と領域マスクを再合成用にキャッシュ
        self.cache_upscaled_base = upscaled_img.copy()
        self.cache_is_creature = mask_low is not None
        self.cache_mask_low = mask_low

        # 3.5 顔復元 (GFPGAN ONNX) の実行
        if face_restoration and face_restoration_fidelity > 0.0:
            try:
                if self.face_runner is None:
                    from gfpgan_onnx import GFPGANOnnxRunner
                    self.face_runner = GFPGANOnnxRunner(self.model_dir)
                
                # 顔復元を実行 (進捗率は 0.8〜0.9 の間を割り当て)
                upscaled_img = self.face_runner.restore_faces(
                    upscaled_img,
                    fidelity=face_restoration_fidelity,
                    progress_callback=lambda msg, pct: progress_callback(msg, 0.8 + pct * 0.1) if progress_callback else None
                )
            except Exception as e:
                # 顔復元に失敗しても処理を続行する
                if progress_callback:
                    progress_callback(f"顔復元をスキップしました: {str(e)}", 0.9)

        # 4. シャープネス処理 (後処理)
        if sharpness_strength > 0:
            if progress_callback:
                progress_callback("シャープネス（輪郭強調）を適用中...", 0.9)
            
            # 生物タイプかつマスクが正しく生成されている場合、背景部分のシャープネスを低減する
            if mask_low is not None:
                try:
                    h_out, w_out, _ = upscaled_img.shape
                    # マスクを拡大サイズにリサイズ
                    mask_high = cv2.resize(mask_low, (w_out, h_out), interpolation=cv2.INTER_LINEAR)
                    
                    # 境界線を滑らかにするためのガウシアンブルー
                    blur_size = max(5, int(min(h_out, w_out) * 0.015))
                    if blur_size % 2 == 0:
                        blur_size += 1
                    mask_high = cv2.GaussianBlur(mask_high, (blur_size, blur_size), 0)
                    mask_high = np.expand_dims(mask_high, axis=2) # (H, W, 1)
                    
                    # 被写体用（設定値）と背景用（設定値の15%）の2枚のシャープネス画像を作成して合成
                    img_sharp_high = self.apply_sharpness(upscaled_img, sharpness_strength)
                    img_sharp_low = self.apply_sharpness(upscaled_img, max(0, int(sharpness_strength * 0.15)))
                    
                    upscaled_img = (img_sharp_high * mask_high + img_sharp_low * (1.0 - mask_high)).astype(np.uint8)
                except Exception as e:
                    # 合成処理に失敗した場合は全体に通常のシャープネスを適用
                    upscaled_img = self.apply_sharpness(upscaled_img, sharpness_strength)
            else:
                upscaled_img = self.apply_sharpness(upscaled_img, sharpness_strength)

        if progress_callback:
            progress_callback("すべての処理が完了しました。", 1.0)

        return Image.fromarray(upscaled_img)

    # ==========================================
    # 5. 推論内部実装
    # ==========================================
    def _inference_pytorch(self, img_np):
        """PyTorchを使用した推論 (自動タイル処理対応)"""
        h, w, c = img_np.shape
        tile_size = 512
        tile_pad = 32
        scale = 4

        # 画像サイズがタイルサイズより小さい場合は、分割せずに一括処理
        if h <= tile_size and w <= tile_size:
            img_t = torch.from_numpy(img_np.transpose((2, 0, 1))).float() / 255.0
            img_t = img_t.unsqueeze(0).to(self.device)
            with torch.no_grad():
                if self.device.type == 'cuda':
                    with torch.amp.autocast('cuda', dtype=torch.float16):
                        output_t = self.model(img_t)
                else:
                    output_t = self.model(img_t)
                output_t = output_t.squeeze(0).clamp(0.0, 1.0)
                output_np = (output_t.cpu().numpy().transpose((1, 2, 0)) * 255.0).round().astype(np.uint8)
            return output_np

        # タイル処理の実行
        img_t = torch.from_numpy(img_np.transpose((2, 0, 1))).float() / 255.0
        img_t = img_t.unsqueeze(0).to(self.device)

        b, c, h, w = img_t.size()
        output_h = h * scale
        output_w = w * scale
        
        # 出力用のバッファと重み（ブレンド用）テンソルを初期化
        output_t = torch.zeros((b, c, output_h, output_w), dtype=img_t.dtype, device='cpu')
        output_weights = torch.zeros((b, 1, output_h, output_w), dtype=img_t.dtype, device='cpu')

        # 縦横のグリッド（開始位置）を計算
        h_starts = []
        h_idx = 0
        while h_idx < h:
            h_starts.append(h_idx)
            if h_idx + tile_size >= h:
                break
            h_idx += tile_size - 2 * tile_pad

        w_starts = []
        w_idx = 0
        while w_idx < w:
            w_starts.append(w_idx)
            if w_idx + tile_size >= w:
                break
            w_idx += tile_size - 2 * tile_pad

        # 各タイルについて推論を実行
        for h_start in h_starts:
            for w_start in w_starts:
                # 入力タイルの切り出し範囲の決定（境界のパディングを考慮）
                h_end = min(h_start + tile_size, h)
                w_end = min(w_start + tile_size, w)
                
                # パディング分の座標拡張
                h_start_pad = max(h_start - tile_pad, 0)
                h_end_pad = min(h_end + tile_pad, h)
                w_start_pad = max(w_start - tile_pad, 0)
                w_end_pad = min(w_end + tile_pad, w)

                # タイル切り出し
                tile_in = img_t[:, :, h_start_pad:h_end_pad, w_start_pad:w_end_pad]
                
                # タイルの推論
                with torch.no_grad():
                    if self.device.type == 'cuda':
                        with torch.amp.autocast('cuda', dtype=torch.float16):
                            tile_out = self.model(tile_in)
                    else:
                        tile_out = self.model(tile_in)

                # 出力位置の決定（スケール4倍を適用）
                out_h_start = h_start_pad * scale
                out_h_end = h_end_pad * scale
                out_w_start = w_start_pad * scale
                out_w_end = w_end_pad * scale

                # ブレンド用重み（ウィンドウ関数）の作成
                # タイルの境界に行くほど重みが0になるように滑らかな重みマップを作成
                tile_h_len = out_h_end - out_h_start
                tile_w_len = out_w_end - out_w_start
                
                # 1次元のコサインウィンドウ
                h_weight = torch.ones((tile_h_len,), dtype=img_t.dtype, device='cpu')
                w_weight = torch.ones((tile_w_len,), dtype=img_t.dtype, device='cpu')
                
                pad_scale = tile_pad * scale
                if h_start_pad > 0 and tile_h_len > pad_scale:
                    h_weight[:pad_scale] = torch.linspace(0, 1, pad_scale, dtype=img_t.dtype, device='cpu')
                if h_end_pad < h and tile_h_len > pad_scale:
                    h_weight[-pad_scale:] = torch.linspace(1, 0, pad_scale, dtype=img_t.dtype, device='cpu')
                if w_start_pad > 0 and tile_w_len > pad_scale:
                    w_weight[:pad_scale] = torch.linspace(0, 1, pad_scale, dtype=img_t.dtype, device='cpu')
                if w_end_pad < w and tile_w_len > pad_scale:
                    w_weight[-pad_scale:] = torch.linspace(1, 0, pad_scale, dtype=img_t.dtype, device='cpu')
                
                # 2次元の重みマップに合成
                tile_weights = h_weight.unsqueeze(1) * w_weight.unsqueeze(0)
                tile_weights = tile_weights.unsqueeze(0).unsqueeze(0) # [1, 1, H, W]

                # 出力バッファへ蓄積 (tile_outをCPUへ転送)
                output_t[:, :, out_h_start:out_h_end, out_w_start:out_w_end] += tile_out.cpu() * tile_weights
                output_weights[:, :, out_h_start:out_h_end, out_w_start:out_w_end] += tile_weights

        # 重みで除算してブレンドを平滑化
        output_t = output_t / (output_weights + 1e-8)
        output_t = output_t.squeeze(0).clamp(0.0, 1.0)
        output_np = (output_t.cpu().numpy().transpose((1, 2, 0)) * 255.0).round().astype(np.uint8)

        return output_np

    def _inference_tensorrt(self, img_np):
        """TensorRTを使用した高速推論"""
        img_t = torch.from_numpy(img_np.transpose((2, 0, 1))).float().unsqueeze(0) / 255.0
        img_t = img_t.to("cuda")

        if self.trt_engine.get_tensor_dtype('input') == trt.DataType.HALF:
            img_t = img_t.half()

        self.trt_context.set_input_shape('input', img_t.shape)

        out_shape = (img_t.shape[0], img_t.shape[1], img_t.shape[2] * 4, img_t.shape[3] * 4)
        output_t = torch.empty(out_shape, dtype=img_t.dtype, device="cuda")

        bindings = {
            'input': img_t.data_ptr(),
            'output': output_t.data_ptr()
        }
        
        for i in range(self.trt_engine.num_io_tensors):
            name = self.trt_engine.get_tensor_name(i)
            self.trt_context.set_tensor_address(name, bindings[name])

        stream = torch.cuda.current_stream().cuda_stream
        self.trt_context.execute_async_v3(stream_handle=stream)
        torch.cuda.current_stream().synchronize()

        output_t = output_t.squeeze(0).clamp(0.0, 1.0)
        output_np = (output_t.float().cpu().numpy().transpose((1, 2, 0)) * 255.0).round().astype(np.uint8)
        
        return output_np

    def recomposite_image(self, face_restoration=False, face_restoration_fidelity=0.7, sharpness_strength=30):
        """キャッシュされた中間データを用いて、推論を実行せずに顔復元強度とシャープネスを高速に再合成する"""
        if self.cache_upscaled_base is None:
            return None
            
        out_img = self.cache_upscaled_base.copy()
        
        # 1. 顔復元の再合成 (顔検出・推論なし)
        if face_restoration and face_restoration_fidelity > 0.0 and self.face_runner is not None:
            out_img = self.face_runner.recomposite_faces(out_img, fidelity=face_restoration_fidelity)
            
        # 2. シャープネスの再適用 (推論なし)
        if sharpness_strength > 0:
            if self.cache_is_creature and self.cache_mask_low is not None:
                h_out, w_out, _ = out_img.shape
                mask_high = cv2.resize(self.cache_mask_low, (w_out, h_out), interpolation=cv2.INTER_LINEAR)
                
                blur_size = max(5, int(min(h_out, w_out) * 0.015))
                if blur_size % 2 == 0:
                    blur_size += 1
                mask_high = cv2.GaussianBlur(mask_high, (blur_size, blur_size), 0)
                mask_high = np.expand_dims(mask_high, axis=2)
                
                img_sharp_high = self.apply_sharpness(out_img, sharpness_strength)
                img_sharp_low = self.apply_sharpness(out_img, max(0, int(sharpness_strength * 0.15)))
                
                out_img = (img_sharp_high * mask_high + img_sharp_low * (1.0 - mask_high)).astype(np.uint8)
            else:
                out_img = self.apply_sharpness(out_img, sharpness_strength)
                
        return Image.fromarray(out_img)
