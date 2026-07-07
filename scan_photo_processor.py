import os
from dataclasses import dataclass

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


@dataclass
class ScanProcessOptions:
    degrid_strength: int = 55
    detail_keep: int = 30
    moire_reduction: bool = True
    max_files: int = 100
    suffix: str = "_scan_clean"


class ScanPhotoProcessor:
    """Printed-photo scan cleanup focused on halftone dots and moire patterns."""

    def __init__(self, tile_size=2048, overlap=96):
        self.tile_size = tile_size
        self.overlap = overlap

    def list_image_files(self, paths, max_files=100):
        image_files = []
        for path in paths:
            if not path:
                continue
            if os.path.isdir(path):
                for name in sorted(os.listdir(path)):
                    file_path = os.path.join(path, name)
                    if self._is_image_file(file_path):
                        image_files.append(file_path)
            elif self._is_image_file(path):
                image_files.append(path)

            if len(image_files) >= max_files:
                break

        return image_files[:max_files]

    def process_batch(self, file_paths, output_dir, options=None, progress_callback=None):
        options = options or ScanProcessOptions()
        file_paths = self.list_image_files(file_paths, options.max_files)
        if not file_paths:
            raise ValueError("処理対象の画像がありません。")

        os.makedirs(output_dir, exist_ok=True)
        results = []
        total = len(file_paths)

        for index, input_path in enumerate(file_paths, start=1):
            base_progress = (index - 1) / total
            if progress_callback:
                progress_callback(
                    f"{index}/{total}: {os.path.basename(input_path)} を処理中...",
                    base_progress,
                )

            output_path = self._make_output_path(input_path, output_dir, options.suffix)
            self.process_file(
                input_path,
                output_path,
                options,
                lambda msg, pct, i=index: progress_callback(
                    f"{i}/{total}: {msg}",
                    base_progress + (pct / total),
                )
                if progress_callback
                else None,
            )
            results.append(output_path)

        if progress_callback:
            progress_callback(f"{total}枚のスキャン画像処理が完了しました。", 1.0)
        return results

    def process_file(self, input_path, output_path, options=None, progress_callback=None):
        options = options or ScanProcessOptions()
        img_bgr = self._read_image(input_path)
        if img_bgr is None:
            raise ValueError(f"画像ファイルを読み込めませんでした: {input_path}")

        if progress_callback:
            progress_callback("網点・モアレの周期ノイズを解析中...", 0.15)

        cleaned_bgr = self.clean_scan_image(img_bgr, options)

        if progress_callback:
            progress_callback("保存中...", 0.92)

        self._write_image(output_path, cleaned_bgr)
        if progress_callback:
            progress_callback("保存完了", 1.0)
        return output_path

    def clean_scan_image(self, img_bgr, options=None):
        options = options or ScanProcessOptions()
        strength = np.clip(options.degrid_strength, 0, 100) / 100.0
        detail_keep = np.clip(options.detail_keep, 0, 100) / 100.0

        if strength <= 0:
            return img_bgr.copy()

        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        filtered_l = self._remove_periodic_noise(l_channel, strength, options.moire_reduction)

        smooth_strength = max(1, int(3 + strength * 8))
        smooth_l = cv2.bilateralFilter(
            filtered_l,
            d=0,
            sigmaColor=8 + strength * 34,
            sigmaSpace=smooth_strength,
        )
        filtered_l = cv2.addWeighted(filtered_l, 1.0 - strength * 0.35, smooth_l, strength * 0.35, 0)

        original_l_float = l_channel.astype(np.float32)
        base_l_float = filtered_l.astype(np.float32)
        detail = original_l_float - cv2.GaussianBlur(original_l_float, (0, 0), 1.2)
        restored_l = base_l_float + detail * detail_keep * 0.35
        restored_l = np.clip(restored_l, 0, 255).astype(np.uint8)

        chroma_strength = strength * 0.25
        if chroma_strength > 0:
            a_channel = self._soften_chroma_noise(a_channel, chroma_strength)
            b_channel = self._soften_chroma_noise(b_channel, chroma_strength)

        cleaned_lab = cv2.merge([restored_l, a_channel, b_channel])
        return cv2.cvtColor(cleaned_lab, cv2.COLOR_LAB2BGR)

    def _remove_periodic_noise(self, channel, strength, moire_reduction):
        if channel.shape[0] * channel.shape[1] > self.tile_size * self.tile_size:
            return self._remove_periodic_noise_tiled(channel, strength, moire_reduction)
        return self._remove_periodic_noise_single(channel, strength, moire_reduction)

    def _remove_periodic_noise_tiled(self, channel, strength, moire_reduction):
        height, width = channel.shape
        output = np.zeros((height, width), dtype=np.float32)
        weights = np.zeros((height, width), dtype=np.float32)
        step = max(256, self.tile_size - self.overlap * 2)

        for y in range(0, height, step):
            for x in range(0, width, step):
                y0 = max(0, y - self.overlap)
                x0 = max(0, x - self.overlap)
                y1 = min(height, y + step + self.overlap)
                x1 = min(width, x + step + self.overlap)
                tile = channel[y0:y1, x0:x1]
                cleaned = self._remove_periodic_noise_single(tile, strength, moire_reduction).astype(np.float32)

                weight = np.ones(cleaned.shape, dtype=np.float32)
                fade = min(self.overlap, cleaned.shape[0] // 3, cleaned.shape[1] // 3)
                if fade > 0:
                    ramp = np.linspace(0.05, 1.0, fade, dtype=np.float32)
                    weight[:fade, :] *= ramp[:, None]
                    weight[-fade:, :] *= ramp[::-1, None]
                    weight[:, :fade] *= ramp[None, :]
                    weight[:, -fade:] *= ramp[::-1][None, :]

                output[y0:y1, x0:x1] += cleaned * weight
                weights[y0:y1, x0:x1] += weight

        return np.clip(output / np.maximum(weights, 1e-6), 0, 255).astype(np.uint8)

    def _remove_periodic_noise_single(self, channel, strength, moire_reduction):
        source = channel.astype(np.float32)
        source -= np.mean(source)

        spectrum = np.fft.fftshift(np.fft.fft2(source))
        magnitude = np.log1p(np.abs(spectrum))
        height, width = channel.shape
        center_y, center_x = height // 2, width // 2

        yy, xx = np.ogrid[:height, :width]
        distance = np.sqrt((yy - center_y) ** 2 + (xx - center_x) ** 2)
        low_cut = max(12, int(min(height, width) * (0.035 if moire_reduction else 0.055)))
        high_cut = int(min(height, width) * 0.47)
        search_mask = (distance > low_cut) & (distance < high_cut)

        if not np.any(search_mask):
            return channel.copy()

        threshold = np.percentile(magnitude[search_mask], 99.88)
        dilated = cv2.dilate(magnitude.astype(np.float32), np.ones((5, 5), np.uint8))
        peaks = (magnitude >= threshold) & (magnitude >= dilated - 1e-6) & search_mask
        peak_points = np.argwhere(peaks)

        if peak_points.size == 0:
            return channel.copy()

        peak_values = magnitude[peaks]
        order = np.argsort(peak_values)[-28:]
        notch_mask = np.ones((height, width), dtype=np.float32)
        radius = max(4.0, min(height, width) * (0.004 + strength * 0.006))
        attenuation = 0.35 + strength * 0.55

        for point_index in order:
            py, px = peak_points[point_index]
            mirror_y = (2 * center_y - py) % height
            mirror_x = (2 * center_x - px) % width
            for ny, nx in ((py, px), (mirror_y, mirror_x)):
                gaussian = np.exp(-(((yy - ny) ** 2 + (xx - nx) ** 2) / (2.0 * radius * radius)))
                notch_mask *= 1.0 - attenuation * gaussian.astype(np.float32)

        filtered = np.real(np.fft.ifft2(np.fft.ifftshift(spectrum * notch_mask)))
        filtered += np.mean(channel)
        return np.clip(filtered, 0, 255).astype(np.uint8)

    def _soften_chroma_noise(self, channel, strength):
        blurred = cv2.GaussianBlur(channel, (0, 0), 0.8 + strength * 1.2)
        return cv2.addWeighted(channel, 1.0 - strength, blurred, strength, 0)

    def _make_output_path(self, input_path, output_dir, suffix):
        base, _ = os.path.splitext(os.path.basename(input_path))
        output_path = os.path.join(output_dir, f"{base}{suffix}.png")
        counter = 2
        while os.path.exists(output_path):
            output_path = os.path.join(output_dir, f"{base}{suffix}_{counter}.png")
            counter += 1
        return output_path

    def _is_image_file(self, path):
        return os.path.isfile(path) and os.path.splitext(path)[1].lower() in IMAGE_EXTENSIONS

    def _read_image(self, path):
        data = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)

    def _write_image(self, path, img_bgr):
        ext = os.path.splitext(path)[1] or ".png"
        ok, encoded = cv2.imencode(ext, img_bgr)
        if not ok:
            raise ValueError(f"画像の保存エンコードに失敗しました: {path}")
        encoded.tofile(path)
