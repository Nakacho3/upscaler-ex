import os
import threading
import queue
import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk
from PIL import Image, ImageTk, ImageOps
import cv2
import numpy as np

# 自作エンジンのインポート
from upscaler import ImageUpscalerEngine, CUPY_AVAILABLE, TENSORRT_AVAILABLE
from ai_mask_engine import AIMaskEngine
from scan_photo_processor import ScanPhotoProcessor, ScanProcessOptions

# tkinterdnd2 を使用した安全なドラッグ＆ドロップのサポート
try:
    from tkinterdnd2 import TkinterDnD, DND_FILES
    TKDND_AVAILABLE = True
except ImportError:
    TKDND_AVAILABLE = False
# アプリのカラーテーマ定義
COLOR_BG_DARK = "#0D0D11"
COLOR_PANEL_DARK = "#15151C"
COLOR_ACCENT = "#10B981"  # エメラルドグリーン
COLOR_ACCENT_HOVER = "#059669"
COLOR_TEXT = "#E2E8F0"
COLOR_TEXT_MUTED = "#94A3B8"
COLOR_BORDER = "#2D2D39"

# ==========================================
# A. インタラクティブ画像比較スライダーウィジェット
# ==========================================
# ==========================================
# A. インタラクティブ画像比較スライダーウィジェット
# ==========================================
class ImageCompareSlider(ctk.CTkFrame):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, fg_color=COLOR_BG_DARK, **kwargs)
        
        self.img_before = None  # PIL.Image
        self.img_after = None   # PIL.Image
        
        self.split_ratio = 0.5  # 初期スプリット位置 (中央)
        self.zoom_factor = 1.0  # ズーム倍率
        self.view_cx = 0        # 表示中心 (画像座標系)
        self.view_cy = 0
        
        # クロップ（トリミング）関連の状態
        self.crop_ratio_str = "なし"
        self.crop_ratio = None  # アスペクト比 (width / height) 数値、または "free"
        self.crop_rect = None  # [x0, y0, x1, y1] (キャンバス座標)
        self.drag_mode = None  # "pan", "split", "crop_move", "crop_resize_lt" 等
        self.crop_handle_size = 10
        
        self.last_mouse_x = 0
        self.last_mouse_y = 0
        
        # キャンバスの配置
        self.canvas = tk.Canvas(
            self,
            bg=COLOR_BG_DARK,
            highlightthickness=0,
            bd=0
        )
        self.canvas.pack(fill="both", expand=True)
        
        # バインディング
        self.canvas.bind("<Configure>", self.on_resize)
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<Double-Button-1>", self.on_double_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<Motion>", self.check_hover)
        self.canvas.bind("<MouseWheel>", self.on_wheel)

    def set_images(self, img_before: Image.Image, img_after: Image.Image, reset_view=True):
        """表示する2枚の画像をセットする (同じサイズを想定)"""
        self.img_before = img_before
        self.img_after = img_after
        if reset_view:
            self.reset_view()
        if self.crop_ratio_str != "なし":
            self.set_crop_ratio(self.crop_ratio_str)
        self.render()

    def reset_view(self):
        """ビューのスケールと位置をリセット"""
        if self.img_before:
            w, h = self.img_before.size
            self.view_cx = w / 2
            self.view_cy = h / 2
        else:
            self.view_cx = 0
            self.view_cy = 0
        self.zoom_factor = 1.0
        self.split_ratio = 0.5

    def on_resize(self, event):
        if self.crop_rect is not None:
            self.init_crop_rect()
        self.render()

    def on_double_click(self, event):
        if self.img_before is None or self.crop_rect is None:
            return
            
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        x0, y0, x1, y1 = self.crop_rect
        hs = self.crop_handle_size
        
        # ハンドル位置の判定 (左上, 右上, 左下, 右下)
        handles = {
            "lt": (x0, y0),
            "rt": (x1, y0),
            "lb": (x0, y1),
            "rb": (x1, y1)
        }
        
        clicked_handle = False
        for name, (hx, hy) in handles.items():
            if abs(event.x - hx) <= hs + 5 and abs(event.y - hy) <= hs + 5:
                clicked_handle = True
                break
                
        if clicked_handle:
            # キャンバス上での画像の実際の表示範囲（バウンディングボックス）を計算
            img_w, img_h = self.img_before.size
            ratio = min(cw / img_w, ch / img_h)
            scale = ratio * self.zoom_factor
            
            # 元画像 (Before) における切り出し範囲の計算
            half_w_img = (cw / 2) / scale
            half_h_img = (ch / 2) / scale
            
            x0_b = self.view_cx - half_w_img
            y0_b = self.view_cy - half_h_img
            x1_b = self.view_cx + half_w_img
            y1_b = self.view_cy + half_h_img
            
            # 画像のキャンバス上での表示座標範囲
            img_cx0 = cw * (0 - x0_b) / (x1_b - x0_b) if (x1_b - x0_b) != 0 else 0
            img_cy0 = ch * (0 - y0_b) / (y1_b - y0_b) if (y1_b - y0_b) != 0 else 0
            img_cx1 = cw * (img_w - x0_b) / (x1_b - x0_b) if (x1_b - x0_b) != 0 else cw
            img_cy1 = ch * (img_h - y0_b) / (y1_b - y0_b) if (y1_b - y0_b) != 0 else ch
            
            # キャンバス範囲でクランプして、現在見えている画像の部分のキャンバス座標を求める
            vis_x0 = max(0.0, img_cx0)
            vis_y0 = max(0.0, img_cy0)
            vis_x1 = min(float(cw), img_cx1)
            vis_y1 = min(float(ch), img_cy1)
            
            vis_w = vis_x1 - vis_x0
            vis_h = vis_y1 - vis_y0
            
            if vis_w > 1 and vis_h > 1:
                if self.crop_ratio == "free" or self.crop_ratio is None:
                    # フリーアスペクト比の場合は、画像表示領域全体にフィットさせる
                    self.crop_rect = [int(vis_x0), int(vis_y0), int(vis_x1), int(vis_y1)]
                else:
                    # 指定されたアスペクト比を維持しつつ、画像表示領域にフィットする最大サイズを求める
                    if vis_w / vis_h > self.crop_ratio:
                        h = vis_h
                        w = h * self.crop_ratio
                    else:
                        w = vis_w
                        h = w / self.crop_ratio
                        
                    # 画像表示領域の中央に配置する
                    x0_new = vis_x0 + (vis_w - w) / 2
                    y0_new = vis_y0 + (vis_h - h) / 2
                    self.crop_rect = [int(x0_new), int(y0_new), int(x0_new + w), int(y0_new + h)]
                self.render()

    def on_click(self, event):
        if self.img_before is None:
            return
            
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        
        # クロップ枠の判定
        if self.crop_rect is not None:
            x0, y0, x1, y1 = self.crop_rect
            hs = self.crop_handle_size
            
            # ハンドル位置の判定 (左上, 右上, 左下, 右下)
            handles = {
                "lt": (x0, y0),
                "rt": (x1, y0),
                "lb": (x0, y1),
                "rb": (x1, y1)
            }
            
            for name, (hx, hy) in handles.items():
                if abs(event.x - hx) <= hs + 5 and abs(event.y - hy) <= hs + 5:
                    self.drag_mode = f"crop_resize_{name}"
                    self.canvas.config(cursor="size_nw_se" if name in ["lt", "rb"] else "size_ne_sw")
                    self.last_mouse_x = event.x
                    self.last_mouse_y = event.y
                    return
            
            # 枠の内部判定（移動）
            if x0 < event.x < x1 and y0 < event.y < y1:
                split_x = int(cw * self.split_ratio)
                if abs(event.x - split_x) < 15:
                    self.drag_mode = "split"
                    self.canvas.config(cursor="sb_h_double_arrow")
                else:
                    self.drag_mode = "crop_move"
                    self.canvas.config(cursor="fleur")
                self.last_mouse_x = event.x
                self.last_mouse_y = event.y
                return

        # 従来の分割線判定
        split_x = int(cw * self.split_ratio)
        if abs(event.x - split_x) < 25:
            self.drag_mode = "split"
            self.canvas.config(cursor="sb_h_double_arrow")
        else:
            self.drag_mode = "pan"
            self.last_mouse_x = event.x
            self.last_mouse_y = event.y
            self.canvas.config(cursor="fleur")

    def on_drag(self, event):
        if self.img_before is None:
            return
            
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        dx = event.x - self.last_mouse_x
        dy = event.y - self.last_mouse_y
        self.last_mouse_x = event.x
        self.last_mouse_y = event.y
        
        # キャンバス上での画像の実際の表示範囲（バウンディングボックス）を計算
        img_w, img_h = self.img_before.size
        ratio = min(cw / img_w, ch / img_h)
        scale = ratio * self.zoom_factor
        half_w_img = (cw / 2) / scale
        half_h_img = (ch / 2) / scale
        x0_b = self.view_cx - half_w_img
        y0_b = self.view_cy - half_h_img
        x1_b = self.view_cx + half_w_img
        y1_b = self.view_cy + half_h_img
        
        img_cx0 = cw * (0 - x0_b) / (x1_b - x0_b) if (x1_b - x0_b) != 0 else 0
        img_cy0 = ch * (0 - y0_b) / (y1_b - y0_b) if (y1_b - y0_b) != 0 else 0
        img_cx1 = cw * (img_w - x0_b) / (x1_b - x0_b) if (x1_b - x0_b) != 0 else cw
        img_cy1 = ch * (img_h - y0_b) / (y1_b - y0_b) if (y1_b - y0_b) != 0 else ch
        
        vis_x0 = max(0.0, img_cx0)
        vis_y0 = max(0.0, img_cy0)
        vis_x1 = min(float(cw), img_cx1)
        vis_y1 = min(float(ch), img_cy1)
        
        if self.drag_mode == "split":
            self.split_ratio = max(0.0, min(1.0, event.x / cw))
            self.render()
        elif self.drag_mode == "pan":
            img_w, img_h = self.img_before.size
            ratio = min(cw / img_w, ch / img_h)
            scale = ratio * self.zoom_factor
            self.view_cx -= dx / scale
            self.view_cy -= dy / scale
            self.view_cx = max(0.0, min(float(img_w), self.view_cx))
            self.view_cy = max(0.0, min(float(img_h), self.view_cy))
            self.render()
        elif self.drag_mode == "crop_move":
            x0, y0, x1, y1 = self.crop_rect
            # 移動先を画像表示領域内に制限
            new_x0 = max(vis_x0, min(vis_x1 - (x1 - x0), x0 + dx))
            new_y0 = max(vis_y0, min(vis_y1 - (y1 - y0), y0 + dy))
            w = x1 - x0
            h = y1 - y0
            self.crop_rect = [int(new_x0), int(new_y0), int(new_x0 + w), int(new_y0 + h)]
            self.render()
        elif self.drag_mode and self.drag_mode.startswith("crop_resize_"):
            handle = self.drag_mode.split("_")[-1]
            x0, y0, x1, y1 = self.crop_rect
            
            # 各ハンドルの移動制限を画像表示領域内に変更
            if handle == "lt":
                x0 = max(vis_x0, min(x1 - 20, x0 + dx))
                y0 = max(vis_y0, min(y1 - 20, y0 + dy))
            elif handle == "rt":
                x1 = max(x0 + 20, min(vis_x1, x1 + dx))
                y0 = max(vis_y0, min(y1 - 20, y0 + dy))
            elif handle == "lb":
                x0 = max(vis_x0, min(x1 - 20, x0 + dx))
                y1 = max(y0 + 20, min(vis_y1, y1 + dy))
            elif handle == "rb":
                x1 = max(x0 + 20, min(vis_x1, x1 + dx))
                y1 = max(y0 + 20, min(vis_y1, y1 + dy))
                
            # アスペクト比維持の適用
            if self.crop_ratio != "free" and self.crop_ratio is not None:
                w = x1 - x0
                h = y1 - y0
                current_ratio = w / h
                if current_ratio > self.crop_ratio:
                    if handle in ["rt", "rb"]:
                        x1 = x0 + int(h * self.crop_ratio)
                    else:
                        x0 = x1 - int(h * self.crop_ratio)
                else:
                    if handle in ["lb", "rb"]:
                        y1 = y0 + int(w / self.crop_ratio)
                    else:
                        y0 = y1 - int(w / self.crop_ratio)
                        
            if x0 >= vis_x0 and y0 >= vis_y0 and x1 <= vis_x1 and y1 <= vis_y1:
                self.crop_rect = [int(x0), int(y0), int(x1), int(y1)]
            self.render()

    def on_release(self, event):
        self.drag_mode = None
        self.canvas.config(cursor="")
        self.check_hover(event)

    def check_hover(self, event):
        if self.img_before is None or self.drag_mode is not None:
            return
        cw = self.canvas.winfo_width()
        split_x = int(cw * self.split_ratio)
        
        if self.crop_rect is not None:
            x0, y0, x1, y1 = self.crop_rect
            hs = self.crop_handle_size
            handles = [(x0, y0), (x1, y0), (x0, y1), (x1, y1)]
            for hx, hy in handles:
                if abs(event.x - hx) <= hs + 5 and abs(event.y - hy) <= hs + 5:
                    self.canvas.config(cursor="size_nw_se" if (hx, hy) in [(x0, y0), (x1, y1)] else "size_ne_sw")
                    return
            
            if x0 < event.x < x1 and y0 < event.y < y1:
                if abs(event.x - split_x) < 15:
                    self.canvas.config(cursor="sb_h_double_arrow")
                else:
                    self.canvas.config(cursor="fleur")
                return

        if abs(event.x - split_x) < 15:
            self.canvas.config(cursor="sb_h_double_arrow")
        else:
            self.canvas.config(cursor="")

    def on_wheel(self, event):
        """マウスホイールによるマウス位置基準のズーム"""
        if self.img_before is None:
            return
            
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw <= 1 or ch <= 1:
            return
            
        # マウス位置 (キャンバス上の座標)
        mx = event.x
        my = event.y
        
        img_w, img_h = self.img_before.size
        ratio = min(cw / img_w, ch / img_h)
        old_scale = ratio * self.zoom_factor
        
        # マウス直下にある画像座標系での位置を逆算
        ix = self.view_cx + (mx - cw / 2) / old_scale
        iy = self.view_cy + (my - ch / 2) / old_scale
        
        # ズーム倍率の変更 (Windowsでのevent.deltaは通常120単位の正負)
        zoom_step = 1.15
        if event.delta > 0:
            self.zoom_factor = min(self.zoom_factor * zoom_step, 15.0)  # 最大15倍
        else:
            self.zoom_factor = max(self.zoom_factor / zoom_step, 0.8)   # 最小0.8倍
            
        new_scale = ratio * self.zoom_factor
        
        # ズーム後の新しい表示中心を計算 (マウス下の画像位置を固定)
        self.view_cx = ix - (mx - cw / 2) / new_scale
        self.view_cy = iy - (my - ch / 2) / new_scale
        
        # 境界クランプ
        self.view_cx = max(0.0, min(float(img_w), self.view_cx))
        self.view_cy = max(0.0, min(float(img_h), self.view_cy))
        
        self.render()

    def render(self):
        """切り出し、合成、レンダリング処理 (解像度比を考慮)"""
        if self.img_before is None or self.img_after is None:
            return
            
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw <= 1 or ch <= 1:
            return
            
        img_w, img_h = self.img_before.size
        ratio = min(cw / img_w, ch / img_h)
        scale = ratio * self.zoom_factor
        
        # 1. 元画像 (Before) における切り出し範囲の計算
        half_w_img = (cw / 2) / scale
        half_h_img = (ch / 2) / scale
        
        x0_b = int(self.view_cx - half_w_img)
        y0_b = int(self.view_cy - half_h_img)
        x1_b = int(self.view_cx + half_w_img)
        y1_b = int(self.view_cy + half_h_img)
        
        crop_w_b = x1_b - x0_b
        crop_h_b = y1_b - y0_b
        if crop_w_b <= 0 or crop_h_b <= 0:
            return
            
        # 2. 超解像後画像 (After) とのサイズ比を算出
        after_w, after_h = self.img_after.size
        scale_ratio_x = after_w / img_w
        scale_ratio_y = after_h / img_h
        
        # 3. 超解像後画像 (After) における切り出し範囲をスケール
        x0_a = int(x0_b * scale_ratio_x)
        y0_a = int(y0_b * scale_ratio_y)
        x1_a = int(x1_b * scale_ratio_x)
        y1_a = int(y1_b * scale_ratio_y)
        
        # 4. それぞれ切り出して、キャンバスサイズへリサイズ
        try:
            cropped_before = self.img_before.crop((x0_b, y0_b, x1_b, y1_b)).resize((cw, ch), Image.Resampling.LANCZOS)
            cropped_after = self.img_after.crop((x0_a, y0_a, x1_a, y1_a)).resize((cw, ch), Image.Resampling.LANCZOS)
        except Exception:
            return  # フェイルセーフ
            
        # 分割位置 (キャンバス座標)
        split_x = int(cw * self.split_ratio)
        
        # 左右を合成
        combined = Image.new("RGB", (cw, ch), color=COLOR_BG_DARK)
        if split_x > 0:
            left_part = cropped_before.crop((0, 0, split_x, ch))
            combined.paste(left_part, (0, 0))
        if split_x < cw:
            right_part = cropped_after.crop((split_x, 0, cw, ch))
            combined.paste(right_part, (split_x, 0))
            
        # 描画
        self.tk_image = ImageTk.PhotoImage(combined)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)
        
        # 分割線とドラッグハンドル (◀ ▶)
        self.canvas.create_line(
            split_x, 0,
            split_x, ch,
            fill=COLOR_ACCENT, width=2
        )
        cy = ch // 2
        self.canvas.create_oval(
            split_x - 16, cy - 16,
            split_x + 16, cy + 16,
            fill=COLOR_ACCENT, outline="#FFFFFF", width=2
        )
        self.canvas.create_text(
            split_x, cy,
            text="◀ ▶", fill="#FFFFFF",
            font=("Arial", 10, "bold")
        )
        
        # ラベルと倍率の描画
        zoom_pct = int(self.zoom_factor * 100)
        self.canvas.create_text(
            40, 20,
            text="Original", fill="#FFFFFF",
            font=("Segoe UI", 12, "bold"), anchor="w"
        )
        self.canvas.create_text(
            cw // 2, 20,
            text=f"{zoom_pct}%", fill="#FFFFFF",
            font=("Segoe UI", 12, "bold"), anchor="center"
        )
        self.canvas.create_text(
            cw - 40, 20,
            text="AI Upscaled", fill=COLOR_ACCENT,
            font=("Segoe UI", 12, "bold"), anchor="e"
        )
        
        # 安全なクロップ枠描画
        if self.crop_rect is not None:
            x0, y0, x1, y1 = self.crop_rect
            
            # クロップ枠線 (点線とアクセントカラー)
            self.canvas.create_rectangle(
                x0, y0, x1, y1,
                outline="#FFFFFF", width=2, dash=(6, 4)
            )
            self.canvas.create_rectangle(
                x0 - 1, y0 - 1, x1 + 1, y1 + 1,
                outline=COLOR_ACCENT, width=1
            )
            
            # ハンドル (小さな四角形)
            hs = self.crop_handle_size
            handles = [
                (x0, y0), (x1, y0), (x0, y1), (x1, y1)
            ]
            for hx, hy in handles:
                self.canvas.create_rectangle(
                    hx - hs//2, hy - hs//2, hx + hs//2, hy + hs//2,
                    fill=COLOR_ACCENT, outline="#FFFFFF", width=1.5
                )

    def set_crop_ratio(self, ratio_str):
        self.crop_ratio_str = ratio_str
        if ratio_str == "なし":
            self.crop_ratio = None
            self.crop_rect = None
        elif ratio_str == "フリー":
            self.crop_ratio = "free"
            self.init_crop_rect()
        else:
            try:
                w_h = ratio_str.split(":")
                self.crop_ratio = float(w_h[0]) / float(w_h[1])
                self.init_crop_rect()
            except Exception:
                self.crop_ratio = None
                self.crop_rect = None
        self.render()

    def init_crop_rect(self):
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw <= 1 or ch <= 1:
            cw, ch = 800, 600
        
        margin = 0.8
        if self.crop_ratio == "free" or self.crop_ratio is None:
            w = int(cw * margin)
            h = int(ch * margin)
        else:
            if cw / ch > self.crop_ratio:
                h = int(ch * margin)
                w = int(h * self.crop_ratio)
            else:
                w = int(cw * margin)
                h = int(w / self.crop_ratio)
        
        x0 = (cw - w) // 2
        y0 = (ch - h) // 2
        self.crop_rect = [x0, y0, x0 + w, y0 + h]

    def get_cropped_image(self, original_image):
        if self.crop_rect is None or original_image is None:
            return original_image
            
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw <= 1 or ch <= 1:
            return original_image
            
        x0, y0, x1, y1 = self.crop_rect
        
        img_w, img_h = self.img_before.size
        ratio = min(cw / img_w, ch / img_h)
        scale = ratio * self.zoom_factor
        
        half_w_img = (cw / 2) / scale
        half_h_img = (ch / 2) / scale
        
        x0_b = self.view_cx - half_w_img
        y0_b = self.view_cy - half_h_img
        x1_b = self.view_cx + half_w_img
        y1_b = self.view_cy + half_h_img
        
        crop_w_b = x1_b - x0_b
        crop_h_b = y1_b - y0_b
        
        rx0 = x0_b + (x0 / cw) * crop_w_b
        ry0 = y0_b + (y0 / ch) * crop_h_b
        rx1 = x0_b + (x1 / cw) * crop_w_b
        ry1 = y0_b + (y1 / ch) * crop_h_b
        
        orig_w, orig_h = original_image.size
        scale_ratio_x = orig_w / img_w
        scale_ratio_y = orig_h / img_h
        
        rx0_scaled = int(max(0, rx0 * scale_ratio_x))
        ry0_scaled = int(max(0, ry0 * scale_ratio_y))
        rx1_scaled = int(min(orig_w, rx1 * scale_ratio_x))
        ry1_scaled = int(min(orig_h, ry1 * scale_ratio_y))
        
        if rx1_scaled > rx0_scaled and ry1_scaled > ry0_scaled:
            return original_image.crop((rx0_scaled, ry0_scaled, rx1_scaled, ry1_scaled))
        return original_image


# ==========================================
# A-2. AIマスクプレビューウィジェット
# ==========================================
class MaskPreviewCanvas(ctk.CTkFrame):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, fg_color=COLOR_BG_DARK, **kwargs)
        self.img_base = None
        self.img_overlay = None
        self.zoom_factor = 1.0
        self.view_cx = 0
        self.view_cy = 0
        self.last_mouse_x = 0
        self.last_mouse_y = 0
        self.drag_mode = None
        self.manual_mode = "pan"
        self.manual_brush_size = 24
        self.manual_mask_callback = None
        
        self.canvas = tk.Canvas(
            self,
            bg=COLOR_BG_DARK,
            highlightthickness=0,
            bd=0
        )
        self.canvas.pack(fill="both", expand=True)
        
        self.canvas.bind("<Configure>", self.on_resize)
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<MouseWheel>", self.on_wheel)

    def set_manual_mask_callback(self, callback):
        self.manual_mask_callback = callback

    def set_manual_mode(self, mode, brush_size=None):
        self.manual_mode = mode
        if brush_size is not None:
            self.manual_brush_size = int(brush_size)

    def set_image(self, img_base: Image.Image, img_overlay: Image.Image = None, reset_view=True):
        self.img_base = img_base
        self.img_overlay = img_overlay
        if reset_view and img_base:
            self.view_cx = img_base.width / 2
            self.view_cy = img_base.height / 2
            self.zoom_factor = 1.0
        self.render()

    def on_resize(self, event):
        self.render()

    def on_click(self, event):
        if self.img_base:
            if self.manual_mode in ("draw", "erase"):
                self.drag_mode = self.manual_mode
                self.paint_manual_mask(event)
                return
            self.drag_mode = "pan"
            self.last_mouse_x = event.x
            self.last_mouse_y = event.y
            self.canvas.config(cursor="fleur")

    def on_drag(self, event):
        if self.img_base and self.drag_mode in ("draw", "erase"):
            self.paint_manual_mask(event)
            return
        if self.img_base and self.drag_mode == "pan":
            cw = self.canvas.winfo_width()
            ch = self.canvas.winfo_height()
            if cw <= 1 or ch <= 1:
                return
            dx = event.x - self.last_mouse_x
            dy = event.y - self.last_mouse_y
            self.last_mouse_x = event.x
            self.last_mouse_y = event.y
            
            ratio = min(cw / self.img_base.width, ch / self.img_base.height)
            scale = ratio * self.zoom_factor
            self.view_cx -= dx / scale
            self.view_cy -= dy / scale
            
            self.view_cx = max(0.0, min(float(self.img_base.width), self.view_cx))
            self.view_cy = max(0.0, min(float(self.img_base.height), self.view_cy))
            self.render()

    def on_release(self, event):
        self.drag_mode = None
        self.canvas.config(cursor="")

    def screen_to_image(self, x, y):
        if not self.img_base:
            return None
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw <= 1 or ch <= 1:
            return None
        ratio = min(cw / self.img_base.width, ch / self.img_base.height)
        scale = ratio * self.zoom_factor
        ix = self.view_cx + (x - cw / 2) / scale
        iy = self.view_cy + (y - ch / 2) / scale
        if ix < 0 or iy < 0 or ix >= self.img_base.width or iy >= self.img_base.height:
            return None
        return int(ix), int(iy)

    def paint_manual_mask(self, event):
        if not self.manual_mask_callback:
            return
        pos = self.screen_to_image(event.x, event.y)
        if pos is None:
            return
        self.manual_mask_callback(pos[0], pos[1], self.manual_brush_size, self.drag_mode == "erase")

    def on_wheel(self, event):
        if not self.img_base:
            return
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw <= 1 or ch <= 1:
            return
        mx = event.x
        my = event.y
        
        ratio = min(cw / self.img_base.width, ch / self.img_base.height)
        old_scale = ratio * self.zoom_factor
        
        ix = self.view_cx + (mx - cw / 2) / old_scale
        iy = self.view_cy + (my - ch / 2) / old_scale
        
        zoom_step = 1.15
        if event.delta > 0:
            self.zoom_factor = min(self.zoom_factor * zoom_step, 15.0)
        else:
            self.zoom_factor = max(self.zoom_factor / zoom_step, 0.8)
            
        new_scale = ratio * self.zoom_factor
        self.view_cx = ix - (mx - cw / 2) / new_scale
        self.view_cy = iy - (my - ch / 2) / new_scale
        
        self.view_cx = max(0.0, min(float(self.img_base.width), self.view_cx))
        self.view_cy = max(0.0, min(float(self.img_base.height), self.view_cy))
        self.render()

    def render(self):
        if not self.img_base:
            return
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw <= 1 or ch <= 1:
            return
            
        ratio = min(cw / self.img_base.width, ch / self.img_base.height)
        scale = ratio * self.zoom_factor
        
        half_w = (cw / 2) / scale
        half_h = (ch / 2) / scale
        
        x0 = int(self.view_cx - half_w)
        y0 = int(self.view_cy - half_h)
        x1 = int(self.view_cx + half_w)
        y1 = int(self.view_cy + half_h)
        
        img_to_crop = self.img_overlay if self.img_overlay else self.img_base
        try:
            cropped = img_to_crop.crop((x0, y0, x1, y1)).resize((cw, ch), Image.Resampling.LANCZOS)
        except Exception:
            return
            
        self.tk_image = ImageTk.PhotoImage(cropped)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)
        
        # ズーム率表示
        zoom_pct = int(self.zoom_factor * 100)
        self.canvas.create_text(
            cw // 2, 20,
            text=f"{zoom_pct}% (ドラッグで移動, ホイールでズーム)", fill="#FFFFFF",
            font=("Segoe UI", 12, "bold"), anchor="center"
        )


# ==========================================
# B. メインアプリケーションウィンドウクラス
# ==========================================
class UpscalerApp(ctk.CTk, TkinterDnD.DnDWrapper if TKDND_AVAILABLE else object):
    def __init__(self):
        super().__init__()
        
        # tkinterdnd2 の初期化 (DnDメソッドをこのウィンドウに有効化)
        if TKDND_AVAILABLE:
            try:
                self.TkdndVersion = TkinterDnD._require(self)
            except Exception as e:
                print(f"TkinterDnD _require failed: {e}")
        
        # ウィンドウ初期設定
        self.title("AI Upscaler & Sharpener EX")
        self.geometry("1200x850")
        self.minsize(950, 780)
        
        # テーマ設定 (ダークテーマ強制)
        ctk.set_appearance_mode("dark")
        
        # 状態変数
        self.input_file_path = None
        self.input_image_pil = None
        self.output_image_pil = None
        self.processing = False
        
        # AIマスク用の状態変数
        self.aimask_masks = None
        self.aimask_confidences = None
        self.aimask_manual_masks = {}
        self.aimask_manual_erase_masks = {}
        self.aimask_enabled_categories = {
            "Person": True,
            "Face": True,
            "Object": True,
            "Text": True,
            "Sky": True,
            "Background": False
        }
        self.aimask_manual_target_var = tk.StringVar(value="Text")
        self.aimask_manual_mode_var = tk.StringVar(value="pan")
        self.aimask_manual_brush_var = tk.IntVar(value=24)
        self.scan_file_paths = []
        self.scan_output_dir_var = tk.StringVar(value="")
        self.scan_degrid_strength_var = tk.IntVar(value=55)
        self.scan_detail_keep_var = tk.IntVar(value=30)
        self.scan_moire_var = tk.BooleanVar(value=True)
        
        # プリセット適正値の定義
        self.presets = {
            "生物 (人物・動物)": {"denoise": 10, "sharpness": 15},
            "物質 (建物・イラスト)": {"denoise": 5, "sharpness": 40},
            "風景 (自然・街並み)": {"denoise": 20, "sharpness": 25},
            "カスタム (手動調整)": None
        }
        
        # 画像処理エンジンの初期化
        self.engine = ImageUpscalerEngine()
        self.mask_engine = AIMaskEngine()
        self.scan_processor = ScanPhotoProcessor()
        
        # GUIスレッド通信用のセーフキュー
        self.gui_queue = queue.Queue()
        
        # UIコンポーネントの構築
        self.build_ui()
        
        # ドラッグ＆ドロップ設定
        self.setup_drag_and_drop()
        
        # 定期ポーリングループ (100ms周期) の開始
        self.after(100, self.process_gui_queue)
        
        # 初回起動時にバックグラウンドでモデル初期化
        self.init_model_async()
        
        # 起動後にウィンドウを最大化 (全画面表示)
        self.after(200, lambda: self.state("zoomed"))

    def build_ui(self):
        # メインウィンドウをタブ化
        self.tabview = ctk.CTkTabview(
            self,
            segmented_button_selected_color=COLOR_ACCENT,
            segmented_button_selected_hover_color=COLOR_ACCENT_HOVER,
            text_color="#FFFFFF"
        )
        self.tabview.pack(fill="both", expand=True)
        
        self.tab_upscale = self.tabview.add("アップスケール")
        self.tab_aimask = self.tabview.add("AIマスク生成")
        self.tab_scan = self.tabview.add("印刷スキャン除去")
        
        # --- アップスケールタブのグリッド構成 ---
        self.tab_upscale.grid_columnconfigure(0, weight=0, minsize=320)  # コントロールパネル
        self.tab_upscale.grid_columnconfigure(1, weight=1)              # プレビューエリア
        self.tab_upscale.grid_rowconfigure(0, weight=1)
        
        # ==========================================
        # 1. 左側: コントロールパネル
        # ==========================================
        self.side_panel = ctk.CTkFrame(
            self.tab_upscale,
            fg_color=COLOR_PANEL_DARK,
            border_color=COLOR_BORDER,
            border_width=1,
            corner_radius=0
        )
        self.side_panel.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
        self.side_panel.grid_columnconfigure(0, weight=1)
        
        # 1.1 アプリケーションロゴ
        self.logo_label = ctk.CTkLabel(
            self.side_panel,
            text="⚡ UPSCALER EX",
            font=ctk.CTkFont(family="Segoe UI", size=22, weight="bold"),
            text_color=COLOR_ACCENT
        )
        self.logo_label.pack(pady=(25, 5), padx=20, anchor="w")
        
        self.sub_logo_label = ctk.CTkLabel(
            self.side_panel,
            text="NVIDIA RTX 20/30 Tensor Core Optimized",
            font=ctk.CTkFont(family="Segoe UI", size=11),
            text_color=COLOR_TEXT_MUTED
        )
        self.sub_logo_label.pack(pady=(0, 25), padx=20, anchor="w")
        
        # 区切り線
        self.create_separator(self.side_panel)

        # 1.2 設定パラメータエリア (スクロール可能なフレーム)
        self.scroll_settings = ctk.CTkScrollableFrame(
            self.side_panel,
            fg_color="transparent"
        )
        self.scroll_settings.pack(fill="both", expand=True, padx=15, pady=5)

        # --- AIモデル選択 ---
        self.model_label = ctk.CTkLabel(
            self.scroll_settings,
            text="AIモデル (高画質化エンジン)",
            font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
            text_color=COLOR_TEXT
        )
        self.model_label.pack(anchor="w", pady=(5, 4))
        
        self.model_var = ctk.StringVar(value="Standard (高画質)")
        self.model_segment = ctk.CTkSegmentedButton(
            self.scroll_settings,
            values=["Standard (高画質)", "Fast (高速)"],
            variable=self.model_var,
            selected_color=COLOR_ACCENT,
            selected_hover_color=COLOR_ACCENT_HOVER,
            text_color="#FFFFFF",
            command=self.on_model_changed
        )
        self.model_segment.pack(fill="x", pady=(0, 10))

        # --- 画像タイプ (複数選択) ---
        self.type_label = ctk.CTkLabel(
            self.scroll_settings,
            text="画像タイプ (対象要素・複数選択可)",
            font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
            text_color=COLOR_TEXT
        )
        self.type_label.pack(anchor="w", pady=(5, 4))
        
        self.type_frame = ctk.CTkFrame(self.scroll_settings, fg_color="transparent")
        self.type_frame.pack(fill="x", pady=(0, 10))
        
        self.type_person_var = tk.BooleanVar(value=True)
        self.type_animal_var = tk.BooleanVar(value=False)
        self.type_landscape_var = tk.BooleanVar(value=False)
        self.type_object_var = tk.BooleanVar(value=False)
        
        self.cb_person = ctk.CTkCheckBox(
            self.type_frame, text="人物", variable=self.type_person_var,
            fg_color=COLOR_ACCENT, hover_color=COLOR_ACCENT_HOVER,
            command=self.on_type_selection_changed, height=22
        )
        self.cb_person.grid(row=0, column=0, sticky="w", pady=2, padx=(0, 10))
        
        self.cb_animal = ctk.CTkCheckBox(
            self.type_frame, text="動物", variable=self.type_animal_var,
            fg_color=COLOR_ACCENT, hover_color=COLOR_ACCENT_HOVER,
            command=self.on_type_selection_changed, height=22
        )
        self.cb_animal.grid(row=0, column=1, sticky="w", pady=2)
        
        self.cb_landscape = ctk.CTkCheckBox(
            self.type_frame, text="風景", variable=self.type_landscape_var,
            fg_color=COLOR_ACCENT, hover_color=COLOR_ACCENT_HOVER,
            command=self.on_type_selection_changed, height=22
        )
        self.cb_landscape.grid(row=1, column=0, sticky="w", pady=2, padx=(0, 10))
        
        self.cb_object = ctk.CTkCheckBox(
            self.type_frame, text="建物・イラスト", variable=self.type_object_var,
            fg_color=COLOR_ACCENT, hover_color=COLOR_ACCENT_HOVER,
            command=self.on_type_selection_changed, height=22
        )
        self.cb_object.grid(row=1, column=1, sticky="w", pady=2)


        # --- 倍率選択 ---
        self.scale_label = ctk.CTkLabel(
            self.scroll_settings,
            text="アップスケーリング倍率",
            font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
            text_color=COLOR_TEXT
        )
        self.scale_label.pack(anchor="w", pady=(5, 4))
        
        self.scale_var = ctk.StringVar(value="4x")
        self.scale_segment = ctk.CTkSegmentedButton(
            self.scroll_settings,
            values=["2x", "4x"],
            variable=self.scale_var,
            selected_color=COLOR_ACCENT,
            selected_hover_color=COLOR_ACCENT_HOVER,
            text_color="#FFFFFF"
        )
        self.scale_segment.pack(fill="x", pady=(0, 10))

        # --- デノイズ強度 ---
        self.denoise_frame = ctk.CTkFrame(self.scroll_settings, fg_color="transparent")
        self.denoise_frame.pack(fill="x", pady=(0, 8))
        
        self.denoise_title_label = ctk.CTkLabel(
            self.denoise_frame,
            text="デノイズ強度 (ノイズ除去)",
            font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
            text_color=COLOR_TEXT
        )
        self.denoise_title_label.grid(row=0, column=0, sticky="w")
        
        self.denoise_val_label = ctk.CTkLabel(
            self.denoise_frame,
            text="10%",
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
            text_color=COLOR_ACCENT
        )
        self.denoise_val_label.grid(row=0, column=1, sticky="e")
        self.denoise_frame.grid_columnconfigure(0, weight=1)
        
        self.denoise_slider = ctk.CTkSlider(
            self.scroll_settings,
            from_=0, to=100,
            number_of_steps=100,
            button_color=COLOR_ACCENT,
            button_hover_color=COLOR_ACCENT_HOVER,
            progress_color=COLOR_ACCENT,
            command=self.update_denoise_label
        )
        self.denoise_slider.set(10)
        self.denoise_slider.pack(fill="x", pady=(0, 5))

        # --- シャープネス強度 ---
        self.sharp_frame = ctk.CTkFrame(self.scroll_settings, fg_color="transparent")
        self.sharp_frame.pack(fill="x", pady=(0, 8))
        
        self.sharp_title_label = ctk.CTkLabel(
            self.sharp_frame,
            text="シャープネス強度 (輪郭強調)",
            font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
            text_color=COLOR_TEXT
        )
        self.sharp_title_label.grid(row=0, column=0, sticky="w")
        
        self.sharp_val_label = ctk.CTkLabel(
            self.sharp_frame,
            text="15%",
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
            text_color=COLOR_ACCENT
        )
        self.sharp_val_label.grid(row=0, column=1, sticky="e")
        self.sharp_frame.grid_columnconfigure(0, weight=1)
        
        self.sharp_slider = ctk.CTkSlider(
            self.scroll_settings,
            from_=0, to=100,
            number_of_steps=100,
            button_color=COLOR_ACCENT,
            button_hover_color=COLOR_ACCENT_HOVER,
            progress_color=COLOR_ACCENT,
            command=self.update_sharp_label
        )
        self.sharp_slider.set(15)
        self.sharp_slider.pack(fill="x", pady=(0, 5))

        # --- 顔復元オプション ---
        self.face_restore_var = tk.BooleanVar(value=True)
        self.face_restore_cb = ctk.CTkCheckBox(
            self.scroll_settings,
            text="👤 人物の顔・瞳を自然に修復",
            font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
            variable=self.face_restore_var,
            fg_color=COLOR_ACCENT,
            hover_color=COLOR_ACCENT_HOVER,
            text_color=COLOR_TEXT,
            command=self.toggle_face_restore_slider
        )
        self.face_restore_cb.pack(fill="x", pady=(8, 2))

        # 顔修復の強度スライダー
        self.face_restore_slider_frame = ctk.CTkFrame(self.scroll_settings, fg_color="transparent")
        self.face_restore_slider_frame.pack(fill="x", pady=(0, 8))

        self.face_restore_label = ctk.CTkLabel(
            self.face_restore_slider_frame,
            text="　修復強度: 45%",
            font=ctk.CTkFont(family="Segoe UI", size=12),
            text_color=COLOR_TEXT_MUTED
        )
        self.face_restore_label.pack(anchor="w", pady=(0, 2))

        self.face_restore_slider = ctk.CTkSlider(
            self.face_restore_slider_frame,
            from_=0,
            to=100,
            number_of_steps=20,  # 5%刻み
            button_color=COLOR_ACCENT,
            button_hover_color=COLOR_ACCENT_HOVER,
            progress_color=COLOR_ACCENT,
            command=self.update_face_restore_label
        )
        self.face_restore_slider.set(45)
        self.face_restore_slider.pack(fill="x")

        # --- 保存先設定 ---
        self.save_loc_label = ctk.CTkLabel(
            self.scroll_settings,
            text="📁 保存先設定",
            font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
            text_color=COLOR_TEXT
        )
        self.save_loc_label.pack(anchor="w", pady=(10, 4))

        self.save_dir_mode_var = tk.StringVar(value="same")
        
        self.save_same_rb = ctk.CTkRadioButton(
            self.scroll_settings,
            text="元ファイルと同じフォルダ",
            font=ctk.CTkFont(family="Segoe UI", size=12),
            variable=self.save_dir_mode_var,
            value="same",
            fg_color=COLOR_ACCENT,
            hover_color=COLOR_ACCENT_HOVER,
            text_color=COLOR_TEXT,
            command=self.toggle_save_dir_ui
        )
        self.save_same_rb.pack(fill="x", pady=(2, 2))

        self.save_custom_rb = ctk.CTkRadioButton(
            self.scroll_settings,
            text="指定したフォルダに保存",
            font=ctk.CTkFont(family="Segoe UI", size=12),
            variable=self.save_dir_mode_var,
            value="custom",
            fg_color=COLOR_ACCENT,
            hover_color=COLOR_ACCENT_HOVER,
            text_color=COLOR_TEXT,
            command=self.toggle_save_dir_ui
        )
        self.save_custom_rb.pack(fill="x", pady=(2, 2))

        # 指定フォルダパス用のフレーム
        self.custom_dir_frame = ctk.CTkFrame(self.scroll_settings, fg_color="transparent")
        self.custom_dir_frame.pack(fill="x", pady=(2, 8))

        self.custom_save_dir_var = tk.StringVar(value="")
        self.custom_dir_entry = ctk.CTkEntry(
            self.custom_dir_frame,
            textvariable=self.custom_save_dir_var,
            font=ctk.CTkFont(family="Segoe UI", size=11),
            height=28,
            state="disabled"
        )
        self.custom_dir_entry.pack(side="left", fill="x", expand=True, padx=(0, 5))

        self.custom_dir_btn = ctk.CTkButton(
            self.custom_dir_frame,
            text="参照",
            font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
            width=50,
            height=28,
            fg_color=COLOR_PANEL_DARK,
            hover_color=COLOR_BORDER,
            state="disabled",
            command=self.select_custom_save_dir
        )
        self.custom_dir_btn.pack(side="right")

        # --- トリミング比率 (保存前) ---
        self.crop_label = ctk.CTkLabel(
            self.scroll_settings,
            text="✂️ トリミング比率 (保存前)",
            font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
            text_color=COLOR_TEXT
        )
        self.crop_label.pack(anchor="w", pady=(5, 4))
        
        self.crop_combo = ctk.CTkComboBox(
            self.scroll_settings,
            values=["なし", "1:1", "4:3", "3:4", "16:9", "9:16", "フリー"],
            dropdown_fg_color=COLOR_PANEL_DARK,
            command=self.on_crop_changed
        )
        self.crop_combo.set("なし")
        self.crop_combo.pack(fill="x", pady=(0, 5))

        # 1.3 操作ボタンと進行状況エリア
        self.create_separator(self.side_panel)
        
        self.action_frame = ctk.CTkFrame(self.side_panel, fg_color="transparent")
        self.action_frame.pack(fill="x", side="bottom", padx=20, pady=25)
        
        # 処理状況テキスト
        self.status_label = ctk.CTkLabel(
            self.action_frame,
            text="モデルを読み込んでいます...",
            font=ctk.CTkFont(family="Segoe UI", size=12),
            text_color=COLOR_TEXT_MUTED,
            wraplength=280
        )
        self.status_label.pack(fill="x", pady=(0, 15))

        self.upscale_mask_label = ctk.CTkLabel(
            self.action_frame,
            text="AIマスク: 未使用",
            font=ctk.CTkFont(family="Segoe UI", size=11),
            text_color=COLOR_TEXT_MUTED,
            wraplength=280
        )
        self.upscale_mask_label.pack(fill="x", pady=(0, 10))
        
        # プログレスバー（デタミネイト型に変更）
        self.progressbar = ctk.CTkProgressBar(
            self.action_frame,
            progress_color=COLOR_ACCENT,
            height=8
        )
        self.progressbar.pack(fill="x", pady=(0, 10))
        self.progressbar.set(0)
        self.progressbar.pack_forget()
        
        # 📂 別の画像を開くボタン
        self.open_btn = ctk.CTkButton(
            self.action_frame,
            text="📂 別の画像を開く",
            font=ctk.CTkFont(family="Segoe UI", size=14, weight="bold"),
            fg_color="#2D2D39",
            hover_color="#3F3F52",
            text_color="#FFFFFF",
            height=40,
            corner_radius=8,
            command=self.select_file
        )
        self.open_btn.pack(fill="x", pady=(0, 10))

        # 実行ボタン
        self.process_btn = ctk.CTkButton(
            self.action_frame,
            text="⚡ 画像を高画質化する",
            font=ctk.CTkFont(family="Segoe UI", size=15, weight="bold"),
            fg_color=COLOR_ACCENT,
            hover_color=COLOR_ACCENT_HOVER,
            text_color="#FFFFFF",
            height=45,
            corner_radius=8,
            command=self.start_process_async,
            state="disabled"
        )
        self.process_btn.pack(fill="x", pady=(0, 10))
        
        # 保存ボタン
        self.save_btn = ctk.CTkButton(
            self.action_frame,
            text="💾 処理後の画像を保存",
            font=ctk.CTkFont(family="Segoe UI", size=14, weight="bold"),
            fg_color="transparent",
            border_color=COLOR_BORDER,
            border_width=2,
            hover_color="#272730",
            text_color=COLOR_TEXT,
            height=40,
            corner_radius=8,
            command=self.save_image,
            state="disabled"
        )
        self.save_btn.pack(fill="x")

        # ==========================================
        # 2. 右側: プレビュー / 表示エリア
        # ==========================================
        self.main_view = ctk.CTkFrame(
            self.tab_upscale,
            fg_color=COLOR_BG_DARK,
            corner_radius=0
        )
        self.main_view.grid(row=0, column=1, sticky="nsew", padx=0, pady=0)
        
        # 画像比較スライダー
        self.compare_slider = ImageCompareSlider(self.main_view)
        
        # 画像未ロード時のプレースホルダー
        self.placeholder_frame = ctk.CTkFrame(
            self.main_view,
            fg_color=COLOR_PANEL_DARK,
            border_color=COLOR_BORDER,
            border_width=2,
            corner_radius=12
        )
        self.placeholder_frame.pack(fill="both", expand=True, padx=30, pady=30)
        
        # プレースホルダーの内部配置
        self.upload_label_icon = ctk.CTkLabel(
            self.placeholder_frame,
            text="🖼️",
            font=ctk.CTkFont(size=64)
        )
        self.upload_label_icon.pack(expand=True, pady=(80, 0))
        
        self.upload_label_main = ctk.CTkLabel(
            self.placeholder_frame,
            text="ここに画像をドラッグ＆ドロップ",
            font=ctk.CTkFont(family="Segoe UI", size=18, weight="bold"),
            text_color=COLOR_TEXT
        )
        self.upload_label_main.pack(expand=True, pady=(10, 0))
        
        self.upload_label_sub = ctk.CTkLabel(
            self.placeholder_frame,
            text="または、ボタンをクリックしてファイルを選択",
            font=ctk.CTkFont(family="Segoe UI", size=13),
            text_color=COLOR_TEXT_MUTED
        )
        self.upload_label_sub.pack(expand=True, pady=(0, 20))
        
        self.select_file_btn = ctk.CTkButton(
            self.placeholder_frame,
            text="画像ファイルを選択",
            font=ctk.CTkFont(family="Segoe UI", size=14, weight="bold"),
            fg_color="#2D2D39",
            hover_color="#3F3F52",
            text_color="#FFFFFF",
            height=40,
            command=self.select_file
        )
        self.select_file_btn.pack(expand=True, pady=(0, 80))

        self.placeholder_frame.bind("<Button-1>", lambda e: self.select_file())

        # ==========================================
        # 3. AIマスク生成タブのUI構築
        # ==========================================
        self.tab_aimask.grid_columnconfigure(0, weight=0, minsize=320)
        self.tab_aimask.grid_columnconfigure(1, weight=1)
        self.tab_aimask.grid_rowconfigure(0, weight=1)

        # 3.1 左側: AIマスク設定サイドパネル
        self.aimask_side_panel = ctk.CTkFrame(
            self.tab_aimask,
            fg_color=COLOR_PANEL_DARK,
            border_color=COLOR_BORDER,
            border_width=1,
            corner_radius=0
        )
        self.aimask_side_panel.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
        
        # タイトルラベル
        self.aimask_title = ctk.CTkLabel(
            self.aimask_side_panel,
            text="🎭 AI MASK GENERATOR",
            font=ctk.CTkFont(family="Segoe UI", size=18, weight="bold"),
            text_color=COLOR_ACCENT
        )
        self.aimask_title.pack(pady=(20, 5), padx=20, anchor="w")
        
        self.aimask_subtitle = ctk.CTkLabel(
            self.aimask_side_panel,
            text="AIによる自動セグメンテーション＆顔検出",
            font=ctk.CTkFont(family="Segoe UI", size=11),
            text_color=COLOR_TEXT_MUTED
        )
        self.aimask_subtitle.pack(pady=(0, 15), padx=20, anchor="w")
        
        self.create_separator(self.aimask_side_panel)

        # 設定スクロールフレーム
        self.aimask_scroll = ctk.CTkScrollableFrame(
            self.aimask_side_panel,
            fg_color="transparent"
        )
        self.aimask_scroll.pack(fill="both", expand=True, padx=15, pady=5)

        # --- 解析パラメータ ---
        self.param_label = ctk.CTkLabel(
            self.aimask_scroll,
            text="■ 解析パラメータ",
            font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
            text_color=COLOR_TEXT
        )
        self.param_label.pack(anchor="w", pady=(5, 5))

        # GPUデバイス選択
        self.device_label = ctk.CTkLabel(
            self.aimask_scroll,
            text="使用デバイス",
            font=ctk.CTkFont(family="Segoe UI", size=11),
            text_color=COLOR_TEXT_MUTED
        )
        self.device_label.pack(anchor="w", pady=(2, 2))
        
        # CUDA検出
        import torch
        device_name = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
        self.aimask_device_combo = ctk.CTkComboBox(
            self.aimask_scroll,
            values=[device_name],
            dropdown_fg_color=COLOR_PANEL_DARK
        )
        self.aimask_device_combo.set(device_name)
        self.aimask_device_combo.configure(state="disabled")
        self.aimask_device_combo.pack(fill="x", pady=(0, 10))

        # 検出感度（confidence threshold）
        self.sens_frame = ctk.CTkFrame(self.aimask_scroll, fg_color="transparent")
        self.sens_frame.pack(fill="x", pady=(0, 2))
        
        self.sens_title_label = ctk.CTkLabel(
            self.sens_frame,
            text="検出感度 (閾値)",
            font=ctk.CTkFont(family="Segoe UI", size=11),
            text_color=COLOR_TEXT_MUTED
        )
        self.sens_title_label.grid(row=0, column=0, sticky="w")
        
        self.sens_val_label = ctk.CTkLabel(
            self.sens_frame,
            text="0.50",
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
            text_color=COLOR_ACCENT
        )
        self.sens_val_label.grid(row=0, column=1, sticky="e")
        self.sens_frame.grid_columnconfigure(0, weight=1)
        
        self.aimask_sens_slider = ctk.CTkSlider(
            self.aimask_scroll,
            from_=0.1, to=0.9,
            number_of_steps=16,
            button_color=COLOR_ACCENT,
            button_hover_color=COLOR_ACCENT_HOVER,
            progress_color=COLOR_ACCENT,
            command=self.update_aimask_sens_label
        )
        self.aimask_sens_slider.set(0.5)
        self.aimask_sens_slider.pack(fill="x", pady=(0, 15))

        self.create_separator(self.aimask_scroll)

        # --- マスクレイヤー一覧 ---
        self.layers_title_label = ctk.CTkLabel(
            self.aimask_scroll,
            text="■ マスクレイヤー一覧",
            font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
            text_color=COLOR_TEXT
        )
        self.layers_title_label.pack(anchor="w", pady=(5, 5))

        # 各カテゴリのチェックボックスとスコア表示用のラベルを辞書で管理
        self.aimask_layer_checkboxes = {}
        self.aimask_layer_score_labels = {}
        
        # カテゴリ一覧の作成
        categories_ja = {
            "Person": "人物 (Person)",
            "Face": "顔 (Face)",
            "Object": "オブジェクト (Object)",
            "Text": "テキスト・文字 (Text)",
            "Sky": "空・天候 (Sky)",
            "Background": "背景 (Background)"
        }
        
        for cat, label_ja in categories_ja.items():
            cat_frame = ctk.CTkFrame(self.aimask_scroll, fg_color="transparent")
            cat_frame.pack(fill="x", pady=4)
            cat_frame.grid_columnconfigure(0, weight=1)
            
            # チェックボックス用BooleanVar
            var = tk.BooleanVar(value=self.aimask_enabled_categories[cat])
            
            # チェックボックス色指定
            color = self.mask_engine.category_colors[cat]
            hex_color = '#%02x%02x%02x' % color
            
            cb = ctk.CTkCheckBox(
                cat_frame,
                text=label_ja,
                variable=var,
                fg_color=hex_color,
                hover_color=hex_color,
                text_color=COLOR_TEXT,
                command=lambda c=cat: self.on_aimask_layer_toggle(c),
                height=22
            )
            cb.grid(row=0, column=0, sticky="w")
            self.aimask_layer_checkboxes[cat] = (cb, var)
            
            score_lbl = ctk.CTkLabel(
                cat_frame,
                text="-",
                font=ctk.CTkFont(family="Segoe UI", size=10),
                text_color=COLOR_TEXT_MUTED
            )
            score_lbl.grid(row=0, column=1, sticky="e")
            self.aimask_layer_score_labels[cat] = score_lbl

        self.create_separator(self.aimask_scroll)

        self.manual_title_label = ctk.CTkLabel(
            self.aimask_scroll,
            text="■ 手動マスク",
            font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
            text_color=COLOR_TEXT
        )
        self.manual_title_label.pack(anchor="w", pady=(5, 5))

        self.manual_target_label = ctk.CTkLabel(
            self.aimask_scroll,
            text="編集対象レイヤー",
            font=ctk.CTkFont(family="Segoe UI", size=12),
            text_color=COLOR_TEXT_MUTED
        )
        self.manual_target_label.pack(anchor="w", pady=(0, 4))

        self.manual_target_combo = ctk.CTkOptionMenu(
            self.aimask_scroll,
            values=["Person", "Face", "Object", "Text", "Sky", "Background"],
            variable=self.aimask_manual_target_var,
            command=self.update_aimask_manual_target
        )
        self.manual_target_combo.pack(fill="x", pady=(0, 8))

        self.manual_mode_frame = ctk.CTkFrame(self.aimask_scroll, fg_color="transparent")
        self.manual_mode_frame.pack(fill="x", pady=(0, 8))

        self.manual_pan_radio = ctk.CTkRadioButton(
            self.manual_mode_frame,
            text="移動",
            variable=self.aimask_manual_mode_var,
            value="pan",
            command=self.update_aimask_manual_mode,
            text_color=COLOR_TEXT
        )
        self.manual_pan_radio.pack(side="left", padx=(0, 10))

        self.manual_draw_radio = ctk.CTkRadioButton(
            self.manual_mode_frame,
            text="描画",
            variable=self.aimask_manual_mode_var,
            value="draw",
            command=self.update_aimask_manual_mode,
            text_color=COLOR_TEXT
        )
        self.manual_draw_radio.pack(side="left", padx=(0, 10))

        self.manual_erase_radio = ctk.CTkRadioButton(
            self.manual_mode_frame,
            text="消去",
            variable=self.aimask_manual_mode_var,
            value="erase",
            command=self.update_aimask_manual_mode,
            text_color=COLOR_TEXT
        )
        self.manual_erase_radio.pack(side="left")

        self.manual_brush_label = ctk.CTkLabel(
            self.aimask_scroll,
            text="ブラシサイズ: 24px",
            font=ctk.CTkFont(family="Segoe UI", size=12),
            text_color=COLOR_TEXT_MUTED
        )
        self.manual_brush_label.pack(anchor="w", pady=(2, 2))

        self.manual_brush_slider = ctk.CTkSlider(
            self.aimask_scroll,
            from_=4,
            to=96,
            number_of_steps=92,
            command=self.update_aimask_brush_size
        )
        self.manual_brush_slider.set(24)
        self.manual_brush_slider.pack(fill="x", pady=(0, 8))

        self.manual_clear_btn = ctk.CTkButton(
            self.aimask_scroll,
            text="選択レイヤーをクリア",
            fg_color="transparent",
            border_color=COLOR_BORDER,
            border_width=2,
            hover_color="#272730",
            text_color=COLOR_TEXT,
            height=32,
            corner_radius=8,
            command=self.clear_aimask_manual_mask
        )
        self.manual_clear_btn.pack(fill="x", pady=(0, 10))

        # 3.2 アクション・進行状況エリア
        self.aimask_action_frame = ctk.CTkFrame(self.aimask_side_panel, fg_color="transparent")
        self.aimask_action_frame.pack(fill="x", side="bottom", padx=20, pady=25)
        
        self.aimask_status_label = ctk.CTkLabel(
            self.aimask_action_frame,
            text="画像をロードすると解析を開始できます",
            font=ctk.CTkFont(family="Segoe UI", size=12),
            text_color=COLOR_TEXT_MUTED,
            wraplength=280
        )
        self.aimask_status_label.pack(fill="x", pady=(0, 10))
        
        self.aimask_progressbar = ctk.CTkProgressBar(
            self.aimask_action_frame,
            progress_color=COLOR_ACCENT,
            height=8
        )
        self.aimask_progressbar.pack(fill="x", pady=(0, 10))
        self.aimask_progressbar.set(0)
        self.aimask_progressbar.pack_forget()

        # 解析開始ボタン
        self.aimask_run_btn = ctk.CTkButton(
            self.aimask_action_frame,
            text="⚡ AIマスク解析を開始",
            font=ctk.CTkFont(family="Segoe UI", size=15, weight="bold"),
            fg_color=COLOR_ACCENT,
            hover_color=COLOR_ACCENT_HOVER,
            text_color="#FFFFFF",
            height=45,
            corner_radius=8,
            command=self.start_aimask_process_async,
            state="disabled"
        )
        self.aimask_run_btn.pack(fill="x", pady=(0, 10))

        # 内部処理用。マスク保存UIは使わず、生成マスクはアップスケール処理へ渡す。
        self.aimask_save_btn = ctk.CTkButton(
            self.aimask_action_frame,
            text="💾 解析結果(マスク)を保存",
            font=ctk.CTkFont(family="Segoe UI", size=14, weight="bold"),
            fg_color="transparent",
            border_color=COLOR_BORDER,
            border_width=2,
            hover_color="#272730",
            text_color=COLOR_TEXT,
            height=40,
            corner_radius=8,
            command=self.save_aimasks,
            state="disabled"
        )

        # 3.3 右側: AIマスクプレビューエリア
        self.aimask_main_view = ctk.CTkFrame(
            self.tab_aimask,
            fg_color=COLOR_BG_DARK,
            corner_radius=0
        )
        self.aimask_main_view.grid(row=0, column=1, sticky="nsew", padx=0, pady=0)
        
        self.aimask_preview = MaskPreviewCanvas(self.aimask_main_view)
        self.aimask_preview.set_manual_mask_callback(self.paint_aimask_manual_mask)
        
        # 画像未ロード時のプレースホルダー (アップスケール側と同一デザイン)
        self.aimask_placeholder = ctk.CTkFrame(
            self.aimask_main_view,
            fg_color=COLOR_PANEL_DARK,
            border_color=COLOR_BORDER,
            border_width=2,
            corner_radius=12
        )
        self.aimask_placeholder.pack(fill="both", expand=True, padx=30, pady=30)
        
        self.aimask_placeholder_icon = ctk.CTkLabel(
            self.aimask_placeholder,
            text="🎭",
            font=ctk.CTkFont(size=64)
        )
        self.aimask_placeholder_icon.pack(expand=True, pady=(80, 0))
        
        self.aimask_placeholder_text = ctk.CTkLabel(
            self.aimask_placeholder,
            text="画像をロードしてください",
            font=ctk.CTkFont(family="Segoe UI", size=18, weight="bold"),
            text_color=COLOR_TEXT
        )
        self.aimask_placeholder_text.pack(expand=True, pady=(10, 80))

        self.build_scan_tab()

    def build_scan_tab(self):
        self.tab_scan.grid_columnconfigure(0, weight=0, minsize=340)
        self.tab_scan.grid_columnconfigure(1, weight=1)
        self.tab_scan.grid_rowconfigure(0, weight=1)

        self.scan_side_panel = ctk.CTkFrame(
            self.tab_scan,
            fg_color=COLOR_PANEL_DARK,
            border_color=COLOR_BORDER,
            border_width=1,
            corner_radius=0
        )
        self.scan_side_panel.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)

        self.scan_title = ctk.CTkLabel(
            self.scan_side_panel,
            text="PRINT SCAN CLEANER",
            font=ctk.CTkFont(family="Segoe UI", size=18, weight="bold"),
            text_color=COLOR_ACCENT
        )
        self.scan_title.pack(pady=(20, 5), padx=20, anchor="w")

        self.scan_subtitle = ctk.CTkLabel(
            self.scan_side_panel,
            text="写真集・グラビアの網点、モアレ、スキャンノイズを低減",
            font=ctk.CTkFont(family="Segoe UI", size=11),
            text_color=COLOR_TEXT_MUTED,
            wraplength=290
        )
        self.scan_subtitle.pack(pady=(0, 15), padx=20, anchor="w")

        self.create_separator(self.scan_side_panel)

        self.scan_scroll = ctk.CTkScrollableFrame(self.scan_side_panel, fg_color="transparent")
        self.scan_scroll.pack(fill="both", expand=True, padx=15, pady=5)

        self.scan_files_label = ctk.CTkLabel(
            self.scan_scroll,
            text="処理する画像 (最大100枚)",
            font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
            text_color=COLOR_TEXT
        )
        self.scan_files_label.pack(anchor="w", pady=(5, 4))

        self.scan_add_files_btn = ctk.CTkButton(
            self.scan_scroll,
            text="画像を追加",
            fg_color="#2D2D39",
            hover_color="#3F3F52",
            command=self.select_scan_files
        )
        self.scan_add_files_btn.pack(fill="x", pady=(0, 6))

        self.scan_add_folder_btn = ctk.CTkButton(
            self.scan_scroll,
            text="フォルダから追加",
            fg_color="#2D2D39",
            hover_color="#3F3F52",
            command=self.select_scan_folder
        )
        self.scan_add_folder_btn.pack(fill="x", pady=(0, 6))

        self.scan_clear_btn = ctk.CTkButton(
            self.scan_scroll,
            text="リストをクリア",
            fg_color="transparent",
            border_color=COLOR_BORDER,
            border_width=2,
            hover_color="#272730",
            text_color=COLOR_TEXT,
            command=self.clear_scan_files
        )
        self.scan_clear_btn.pack(fill="x", pady=(0, 10))

        self.scan_count_label = ctk.CTkLabel(
            self.scan_scroll,
            text="0枚選択中",
            font=ctk.CTkFont(family="Segoe UI", size=12),
            text_color=COLOR_TEXT_MUTED
        )
        self.scan_count_label.pack(anchor="w", pady=(0, 4))

        self.scan_output_label = ctk.CTkLabel(
            self.scan_scroll,
            text="保存先フォルダ",
            font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
            text_color=COLOR_TEXT
        )
        self.scan_output_label.pack(anchor="w", pady=(8, 4))

        self.scan_output_frame = ctk.CTkFrame(self.scan_scroll, fg_color="transparent")
        self.scan_output_frame.pack(fill="x", pady=(0, 10))

        self.scan_output_entry = ctk.CTkEntry(
            self.scan_output_frame,
            textvariable=self.scan_output_dir_var,
            font=ctk.CTkFont(family="Segoe UI", size=11),
            height=30
        )
        self.scan_output_entry.pack(side="left", fill="x", expand=True, padx=(0, 6))

        self.scan_output_btn = ctk.CTkButton(
            self.scan_output_frame,
            text="参照",
            width=54,
            height=30,
            fg_color="#2D2D39",
            hover_color="#3F3F52",
            command=self.select_scan_output_dir
        )
        self.scan_output_btn.pack(side="right")

        self.scan_strength_frame = ctk.CTkFrame(self.scan_scroll, fg_color="transparent")
        self.scan_strength_frame.pack(fill="x", pady=(0, 4))
        self.scan_strength_title = ctk.CTkLabel(
            self.scan_strength_frame,
            text="網点・ノイズ除去強度",
            font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
            text_color=COLOR_TEXT
        )
        self.scan_strength_title.grid(row=0, column=0, sticky="w")
        self.scan_strength_value = ctk.CTkLabel(
            self.scan_strength_frame,
            text="55%",
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
            text_color=COLOR_ACCENT
        )
        self.scan_strength_value.grid(row=0, column=1, sticky="e")
        self.scan_strength_frame.grid_columnconfigure(0, weight=1)

        self.scan_strength_slider = ctk.CTkSlider(
            self.scan_scroll,
            from_=0,
            to=100,
            number_of_steps=100,
            command=self.update_scan_strength_label
        )
        self.scan_strength_slider.set(55)
        self.scan_strength_slider.pack(fill="x", pady=(0, 10))

        self.scan_detail_frame = ctk.CTkFrame(self.scan_scroll, fg_color="transparent")
        self.scan_detail_frame.pack(fill="x", pady=(0, 4))
        self.scan_detail_title = ctk.CTkLabel(
            self.scan_detail_frame,
            text="質感を残す量",
            font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
            text_color=COLOR_TEXT
        )
        self.scan_detail_title.grid(row=0, column=0, sticky="w")
        self.scan_detail_value = ctk.CTkLabel(
            self.scan_detail_frame,
            text="30%",
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
            text_color=COLOR_ACCENT
        )
        self.scan_detail_value.grid(row=0, column=1, sticky="e")
        self.scan_detail_frame.grid_columnconfigure(0, weight=1)

        self.scan_detail_slider = ctk.CTkSlider(
            self.scan_scroll,
            from_=0,
            to=100,
            number_of_steps=100,
            command=self.update_scan_detail_label
        )
        self.scan_detail_slider.set(30)
        self.scan_detail_slider.pack(fill="x", pady=(0, 10))

        self.scan_moire_cb = ctk.CTkCheckBox(
            self.scan_scroll,
            text="モアレ低減を有効にする",
            variable=self.scan_moire_var,
            fg_color=COLOR_ACCENT,
            hover_color=COLOR_ACCENT_HOVER,
            text_color=COLOR_TEXT
        )
        self.scan_moire_cb.pack(fill="x", pady=(0, 8))

        self.scan_action_frame = ctk.CTkFrame(self.scan_side_panel, fg_color="transparent")
        self.scan_action_frame.pack(fill="x", side="bottom", padx=20, pady=25)

        self.scan_status_label = ctk.CTkLabel(
            self.scan_action_frame,
            text="画像を追加してください",
            font=ctk.CTkFont(family="Segoe UI", size=12),
            text_color=COLOR_TEXT_MUTED,
            wraplength=290
        )
        self.scan_status_label.pack(fill="x", pady=(0, 10))

        self.scan_progressbar = ctk.CTkProgressBar(
            self.scan_action_frame,
            progress_color=COLOR_ACCENT,
            height=8
        )
        self.scan_progressbar.pack(fill="x", pady=(0, 10))
        self.scan_progressbar.set(0)
        self.scan_progressbar.pack_forget()

        self.scan_run_btn = ctk.CTkButton(
            self.scan_action_frame,
            text="一括除去を開始",
            font=ctk.CTkFont(family="Segoe UI", size=15, weight="bold"),
            fg_color=COLOR_ACCENT,
            hover_color=COLOR_ACCENT_HOVER,
            text_color="#FFFFFF",
            height=45,
            corner_radius=8,
            command=self.start_scan_process_async,
            state="disabled"
        )
        self.scan_run_btn.pack(fill="x")

        self.scan_main_view = ctk.CTkFrame(
            self.tab_scan,
            fg_color=COLOR_BG_DARK,
            corner_radius=0
        )
        self.scan_main_view.grid(row=0, column=1, sticky="nsew", padx=0, pady=0)

        self.scan_list_frame = ctk.CTkFrame(
            self.scan_main_view,
            fg_color=COLOR_PANEL_DARK,
            border_color=COLOR_BORDER,
            border_width=2,
            corner_radius=12
        )
        self.scan_list_frame.pack(fill="both", expand=True, padx=30, pady=30)

        self.scan_list_title = ctk.CTkLabel(
            self.scan_list_frame,
            text="処理待ち画像",
            font=ctk.CTkFont(family="Segoe UI", size=18, weight="bold"),
            text_color=COLOR_TEXT
        )
        self.scan_list_title.pack(anchor="w", padx=20, pady=(18, 8))

        self.scan_file_listbox = tk.Listbox(
            self.scan_list_frame,
            bg=COLOR_BG_DARK,
            fg=COLOR_TEXT,
            selectbackground=COLOR_ACCENT,
            highlightthickness=0,
            bd=0,
            font=("Segoe UI", 11)
        )
        self.scan_file_listbox.pack(fill="both", expand=True, padx=20, pady=(0, 20))

    def update_scan_strength_label(self, val):
        value = int(float(val))
        self.scan_degrid_strength_var.set(value)
        self.scan_strength_value.configure(text=f"{value}%")

    def update_scan_detail_label(self, val):
        value = int(float(val))
        self.scan_detail_keep_var.set(value)
        self.scan_detail_value.configure(text=f"{value}%")

    def select_scan_files(self):
        file_paths = filedialog.askopenfilenames(
            filetypes=[
                ("Image files", "*.png;*.jpg;*.jpeg;*.webp;*.bmp;*.tif;*.tiff"),
                ("All files", "*.*")
            ]
        )
        self.add_scan_files(file_paths)

    def select_scan_folder(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            image_files = self.scan_processor.list_image_files([folder_path], max_files=100)
            self.add_scan_files(image_files)
            if not self.scan_output_dir_var.get():
                self.scan_output_dir_var.set(os.path.join(folder_path, "scan_cleaned"))

    def add_scan_files(self, file_paths):
        if self.processing:
            return

        added = 0
        for file_path in file_paths:
            if len(self.scan_file_paths) >= 100:
                break
            normalized = os.path.normpath(file_path)
            ext = os.path.splitext(normalized)[1].lower()
            if ext not in [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"]:
                continue
            if normalized not in self.scan_file_paths:
                self.scan_file_paths.append(normalized)
                added += 1

        if added and not self.scan_output_dir_var.get() and self.scan_file_paths:
            base_dir = os.path.dirname(self.scan_file_paths[0])
            self.scan_output_dir_var.set(os.path.join(base_dir, "scan_cleaned"))

        self.refresh_scan_file_list()

    def clear_scan_files(self):
        if self.processing:
            return
        self.scan_file_paths = []
        self.refresh_scan_file_list()

    def refresh_scan_file_list(self):
        self.scan_file_listbox.delete(0, tk.END)
        for file_path in self.scan_file_paths:
            self.scan_file_listbox.insert(tk.END, file_path)

        count = len(self.scan_file_paths)
        self.scan_count_label.configure(text=f"{count}枚選択中")
        if count:
            self.scan_status_label.configure(text="設定を確認して一括除去を開始できます", text_color=COLOR_TEXT_MUTED)
            if not self.processing:
                self.scan_run_btn.configure(state="normal")
        else:
            self.scan_status_label.configure(text="画像を追加してください", text_color=COLOR_TEXT_MUTED)
            self.scan_run_btn.configure(state="disabled")

    def select_scan_output_dir(self):
        selected_dir = filedialog.askdirectory(
            initialdir=self.scan_output_dir_var.get() or os.path.expanduser("~")
        )
        if selected_dir:
            self.scan_output_dir_var.set(selected_dir)

    def set_scan_controls_state(self, state):
        for widget in (
            self.scan_add_files_btn,
            self.scan_add_folder_btn,
            self.scan_clear_btn,
            self.scan_output_entry,
            self.scan_output_btn,
            self.scan_strength_slider,
            self.scan_detail_slider,
            self.scan_moire_cb,
            self.scan_run_btn,
        ):
            widget.configure(state=state)

    def start_scan_process_async(self):
        if self.processing or not self.scan_file_paths:
            return

        output_dir = self.scan_output_dir_var.get().strip()
        if not output_dir:
            messagebox.showerror("エラー", "保存先フォルダを指定してください。")
            return

        self.processing = True
        self.set_scan_controls_state("disabled")
        self.scan_progressbar.pack(fill="x", pady=(0, 10), after=self.scan_status_label)
        self.scan_progressbar.configure(mode="determinate")
        self.scan_progressbar.set(0)
        self.scan_status_label.configure(text="スキャン画像の一括除去を開始します...", text_color=COLOR_TEXT_MUTED)

        options = ScanProcessOptions(
            degrid_strength=int(self.scan_degrid_strength_var.get()),
            detail_keep=int(self.scan_detail_keep_var.get()),
            moire_reduction=self.scan_moire_var.get(),
            max_files=100,
        )
        target_files = list(self.scan_file_paths)

        def task():
            try:
                def progress_cb(msg, progress):
                    self.gui_queue.put({
                        "type": "scan_status",
                        "text": msg,
                        "progress": progress,
                    })

                outputs = self.scan_processor.process_batch(
                    target_files,
                    output_dir,
                    options=options,
                    progress_callback=progress_cb,
                )
                self.gui_queue.put({"type": "scan_complete", "outputs": outputs})
            except Exception as e:
                self.gui_queue.put({"type": "scan_error", "error": str(e)})

        threading.Thread(target=task, daemon=True).start()

    def on_scan_process_complete(self, outputs):
        self.processing = False
        self.scan_progressbar.pack_forget()
        self.set_scan_controls_state("normal")
        self.scan_run_btn.configure(state="normal" if self.scan_file_paths else "disabled")
        self.scan_status_label.configure(
            text=f"完了: {len(outputs)}枚を保存しました",
            text_color=COLOR_ACCENT
        )
        messagebox.showinfo("完了", f"{len(outputs)}枚のスキャン画像処理が完了しました。")

    def on_scan_process_error(self, err_msg):
        self.processing = False
        self.scan_progressbar.pack_forget()
        self.set_scan_controls_state("normal")
        self.scan_run_btn.configure(state="normal" if self.scan_file_paths else "disabled")
        self.scan_status_label.configure(text="処理エラーが発生しました。", text_color="#EF4444")
        messagebox.showerror("処理エラー", f"スキャン画像処理中にエラーが発生しました:\n{err_msg}")

    def select_file(self):
        """画像ファイル選択ダイアログを開く"""
        file_path = filedialog.askopenfilename(
            filetypes=[
                ("Image files", "*.png;*.jpg;*.jpeg;*.webp;*.bmp;*.tiff"),
                ("All files", "*.*")
            ]
        )
        if file_path:
            self.load_input_image(file_path)

    def setup_drag_and_drop(self):
        """ドラッグ＆ドロップのイベントをウィンドウとプレースホルダーにバインドする"""
        if TKDND_AVAILABLE:
            try:
                self.drop_target_register(DND_FILES)
                self.dnd_bind("<<Drop>>", self.on_file_drop)
                
                inner_widget = self.placeholder_frame
                if hasattr(self.placeholder_frame, "_canvas"):
                    inner_widget = self.placeholder_frame._canvas
                
                TkinterDnD.DnDWrapper.drop_target_register(inner_widget, DND_FILES)
                inner_widget.dnd_bind("<<Drop>>", self.on_file_drop)
            except Exception as e:
                print(f"Failed to setup drag and drop: {e}")

    def on_file_drop(self, event):
        """ドラッグ＆ドロップで画像が落とされた時の処理"""
        if self.processing:
            return
            
        try:
            files = self.tk.splitlist(event.data)
            if not files:
                return
            file_path = files[0]
        except Exception:
            file_path = event.data
            
        if not file_path:
            return
            
        if file_path.startswith("{") and file_path.endswith("}"):
            file_path = file_path[1:-1]
            
        file_path = os.path.normpath(file_path)
            
        ext = os.path.splitext(file_path)[1].lower()
        if ext in [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"]:
            self.load_input_image(file_path)
        else:
            messagebox.showerror("エラー", f"対応していないファイル形式です: {ext}\n画像ファイル (PNG, JPG, WEBPなど) を選択してください。")

    # ==========================================
    # C. イベントハンドラ & ユーティリティ
    # ==========================================
    def create_separator(self, parent):
        sep = ctk.CTkFrame(parent, height=1, fg_color=COLOR_BORDER)
        sep.pack(fill="x", padx=15, pady=5)
        return sep

    def update_denoise_label(self, val):
        self.denoise_val_label.configure(text=f"{int(val)}%")

    def update_sharp_label(self, val):
        self.sharp_val_label.configure(text=f"{int(val)}%")
        
        # 処理完了後の画像が存在し、かつ現在処理中でなければリアルタイムに再合成してプレビューを更新
        if hasattr(self, 'output_image_pil') and self.output_image_pil is not None and not self.processing:
            face_fidelity = float(self.face_restore_slider.get()) / 100.0 if hasattr(self, 'face_restore_slider') else 0.7
            recomposed_img = self.engine.recomposite_image(
                face_restoration=self.face_restore_var.get(),
                face_restoration_fidelity=face_fidelity,
                sharpness_strength=int(val)
            )
            if recomposed_img:
                self.output_image_pil = recomposed_img
                self.compare_slider.set_images(self.input_image_pil, self.output_image_pil, reset_view=False)

    def update_face_restore_label(self, val):
        self.face_restore_label.configure(text=f"　修復強度: {int(val)}%")
        
        # 処理完了後の画像が存在し、かつ現在処理中でなければリアルタイムに再合成してプレビューを更新
        if hasattr(self, 'output_image_pil') and self.output_image_pil is not None and not self.processing:
            face_fidelity = float(val) / 100.0
            sharpness = int(self.sharp_slider.get())
            recomposed_img = self.engine.recomposite_image(
                face_restoration=self.face_restore_var.get(),
                face_restoration_fidelity=face_fidelity,
                sharpness_strength=sharpness
            )
            if recomposed_img:
                self.output_image_pil = recomposed_img
                self.compare_slider.set_images(self.input_image_pil, self.output_image_pil, reset_view=False)

    def toggle_face_restore_slider(self):
        """顔修復チェックボックスの状態に合わせてスライダーを有効・無効化、およびプレビューのリアルタイム更新"""
        if hasattr(self, 'face_restore_slider'):
            if self.face_restore_var.get() and not self.processing:
                self.face_restore_slider.configure(state="normal")
            else:
                self.face_restore_slider.configure(state="disabled")
        
        # 処理完了後の画像が存在し、かつ現在処理中でなければリアルタイムに再合成してプレビューを更新
        if hasattr(self, 'output_image_pil') and self.output_image_pil is not None and not self.processing:
            face_fidelity = float(self.face_restore_slider.get()) / 100.0 if hasattr(self, 'face_restore_slider') else 0.7
            sharpness = int(self.sharp_slider.get())
            recomposed_img = self.engine.recomposite_image(
                face_restoration=self.face_restore_var.get(),
                face_restoration_fidelity=face_fidelity,
                sharpness_strength=sharpness
            )
            if recomposed_img:
                self.output_image_pil = recomposed_img
                self.compare_slider.set_images(self.input_image_pil, self.output_image_pil, reset_view=False)

    def on_type_selection_changed(self):
        """画像タイプのチェックボックス状態が変更された時のイベントハンドラ"""
        # 処理完了後の画像が存在し、かつ現在処理中でなければリアルタイムに再合成してプレビューを更新
        if hasattr(self, 'output_image_pil') and self.output_image_pil is not None and not self.processing:
            is_creature = self.type_person_var.get() or self.type_animal_var.get()
            self.engine.cache_is_creature = is_creature
            
            face_fidelity = float(self.face_restore_slider.get()) / 100.0 if hasattr(self, 'face_restore_slider') else 0.7
            sharpness = int(self.sharp_slider.get())
            recomposed_img = self.engine.recomposite_image(
                face_restoration=self.face_restore_var.get(),
                face_restoration_fidelity=face_fidelity,
                sharpness_strength=sharpness
            )
            if recomposed_img:
                self.output_image_pil = recomposed_img
                self.compare_slider.set_images(self.input_image_pil, self.output_image_pil, reset_view=False)

    def toggle_save_dir_ui(self):
        """保存先ラジオボタンの選択に合わせて入力欄とボタンの有効・無効化"""
        if hasattr(self, 'custom_dir_entry') and hasattr(self, 'custom_dir_btn'):
            if not self.processing and self.save_dir_mode_var.get() == "custom":
                self.custom_dir_entry.configure(state="normal")
                self.custom_dir_btn.configure(state="normal")
            else:
                self.custom_dir_entry.configure(state="disabled")
                self.custom_dir_btn.configure(state="disabled")

    def select_custom_save_dir(self):
        """カスタム保存先ディレクトリの選択"""
        selected_dir = filedialog.askdirectory(
            initialdir=self.custom_save_dir_var.get() or os.path.expanduser("~")
        )
        if selected_dir:
            self.custom_save_dir_var.set(selected_dir)

    # ==========================================
    # AIマスク生成タブ用イベントハンドラ＆処理
    # ==========================================
    def update_aimask_sens_label(self, val):
        self.sens_val_label.configure(text=f"{val:.2f}")

    def update_aimask_manual_target(self, value):
        self.aimask_manual_target_var.set(value)
        self.update_aimask_preview()

    def update_aimask_manual_mode(self):
        mode = self.aimask_manual_mode_var.get()
        brush_size = self.aimask_manual_brush_var.get()
        self.aimask_preview.set_manual_mode(mode, brush_size)

    def update_aimask_brush_size(self, val):
        brush_size = int(float(val))
        self.aimask_manual_brush_var.set(brush_size)
        self.manual_brush_label.configure(text=f"ブラシサイズ: {brush_size}px")
        self.aimask_preview.set_manual_mode(self.aimask_manual_mode_var.get(), brush_size)

    def ensure_aimask_edit_masks(self, cat=None):
        if self.input_image_pil is None:
            return None, None
        cat = cat or self.aimask_manual_target_var.get()
        h = self.input_image_pil.height
        w = self.input_image_pil.width
        if self.aimask_manual_masks is None:
            self.aimask_manual_masks = {}
        if self.aimask_manual_erase_masks is None:
            self.aimask_manual_erase_masks = {}
        if cat not in self.aimask_manual_masks:
            self.aimask_manual_masks[cat] = np.zeros((h, w), dtype=np.uint8)
        if cat not in self.aimask_manual_erase_masks:
            self.aimask_manual_erase_masks[cat] = np.zeros((h, w), dtype=np.uint8)
        return self.aimask_manual_masks[cat], self.aimask_manual_erase_masks[cat]

    def aimask_layer_has_manual_edit(self, cat):
        add_mask = self.aimask_manual_masks.get(cat)
        erase_mask = self.aimask_manual_erase_masks.get(cat)
        has_add = add_mask is not None and np.any(add_mask > 0)
        has_erase = erase_mask is not None and np.any(erase_mask > 0)
        return has_add or has_erase

    def has_saveable_aimask(self):
        if self.aimask_masks is None and not self.aimask_manual_masks and not self.aimask_manual_erase_masks:
            return False
        merged = self.get_merged_aimask_masks()
        if merged is None:
            return False
        for cat, is_enabled in self.aimask_enabled_categories.items():
            if is_enabled and cat in merged and np.any(merged[cat] > 0):
                return True
        return False

    def paint_aimask_manual_mask(self, x, y, brush_size, erase=False):
        target_cat = self.aimask_manual_target_var.get()
        add_mask, erase_mask = self.ensure_aimask_edit_masks(target_cat)
        if add_mask is None or erase_mask is None:
            return
        radius = max(1, brush_size // 2)
        if erase:
            cv2.circle(add_mask, (x, y), radius, 0, -1)
            cv2.circle(erase_mask, (x, y), radius, 255, -1)
        else:
            cv2.circle(add_mask, (x, y), radius, 255, -1)
            cv2.circle(erase_mask, (x, y), radius, 0, -1)
        lbl = self.aimask_layer_score_labels.get(target_cat)
        if lbl:
            lbl.configure(text="手動編集あり" if self.aimask_layer_has_manual_edit(target_cat) else "未作成")
        self.aimask_save_btn.configure(state="normal" if self.has_saveable_aimask() else "disabled")
        self.update_aimask_preview()
        self.update_upscale_mask_label()

    def clear_aimask_manual_mask(self):
        target_cat = self.aimask_manual_target_var.get()
        add_mask, erase_mask = self.ensure_aimask_edit_masks(target_cat)
        if add_mask is None or erase_mask is None:
            return
        add_mask[:, :] = 0
        erase_mask[:, :] = 0
        lbl = self.aimask_layer_score_labels.get(target_cat)
        if lbl:
            score = self.aimask_confidences.get(target_cat, 0.0) if self.aimask_confidences else 0.0
            lbl.configure(text=f"信頼度: {score:.2f}" if score > 0.0 else "未検出")
        self.aimask_save_btn.configure(state="normal" if self.has_saveable_aimask() else "disabled")
        self.update_aimask_preview()
        self.update_upscale_mask_label()

    def get_merged_aimask_masks(self):
        if self.input_image_pil is None:
            return None
        h = self.input_image_pil.height
        w = self.input_image_pil.width
        merged = {}
        for cat in self.aimask_enabled_categories.keys():
            if self.aimask_masks is not None and cat in self.aimask_masks:
                merged[cat] = self.aimask_masks[cat].copy()
            else:
                merged[cat] = np.zeros((h, w), dtype=np.uint8)

        for cat, add_mask in self.aimask_manual_masks.items():
            if cat not in merged:
                merged[cat] = np.zeros((h, w), dtype=np.uint8)
            merged[cat] = cv2.bitwise_or(merged[cat], add_mask)

        for cat, erase_mask in self.aimask_manual_erase_masks.items():
            if cat not in merged:
                continue
            merged[cat] = cv2.bitwise_and(merged[cat], cv2.bitwise_not(erase_mask))
        return merged

    def get_upscale_region_masks(self):
        merged_masks = self.get_merged_aimask_masks()
        if merged_masks is None:
            return None
        upscale_categories = ("Person", "Face", "Object", "Text")
        selected_masks = {}
        for cat in upscale_categories:
            if not self.aimask_enabled_categories.get(cat, False):
                continue
            mask = merged_masks.get(cat)
            if mask is not None and np.any(mask > 0):
                selected_masks[cat] = mask.copy()
        return selected_masks if selected_masks else None

    def update_upscale_mask_label(self):
        if not hasattr(self, "upscale_mask_label"):
            return
        region_masks = self.get_upscale_region_masks()
        if not region_masks:
            self.upscale_mask_label.configure(text="AIマスク: 未使用", text_color=COLOR_TEXT_MUTED)
            return
        labels = {
            "Person": "人物",
            "Face": "顔",
            "Object": "オブジェクト",
            "Text": "テキスト"
        }
        active_names = [labels.get(cat, cat) for cat in region_masks.keys()]
        self.upscale_mask_label.configure(
            text=f"AIマスク: {', '.join(active_names)} を反映",
            text_color=COLOR_ACCENT
        )

    def on_aimask_layer_toggle(self, cat):
        if cat in self.aimask_layer_checkboxes:
            _, var = self.aimask_layer_checkboxes[cat]
            self.aimask_enabled_categories[cat] = var.get()
            self.update_aimask_preview()
            self.update_upscale_mask_label()

    def update_aimask_preview(self):
        if self.input_image_pil is None:
            return
        
        merged_masks = self.get_merged_aimask_masks()
        if merged_masks is not None:
            overlay_img = self.mask_engine.create_overlay(
                np.array(self.input_image_pil),
                merged_masks,
                self.aimask_enabled_categories,
                opacity=0.4
            )
            self.aimask_preview.set_image(self.input_image_pil, overlay_img, reset_view=False)
        else:
            self.aimask_preview.set_image(self.input_image_pil, reset_view=False)

    def start_aimask_process_async(self):
        if self.processing or not self.input_image_pil:
            return
            
        self.processing = True
        self.aimask_run_btn.configure(state="disabled")
        self.aimask_save_btn.configure(state="disabled")
        self.aimask_sens_slider.configure(state="disabled")
        self.manual_target_combo.configure(state="disabled")
        
        self.aimask_progressbar.pack(fill="x", pady=(0, 10), after=self.aimask_status_label)
        self.aimask_progressbar.configure(mode="determinate")
        self.aimask_progressbar.set(0)
        
        sens = self.aimask_sens_slider.get()
        
        def task():
            try:
                def progress_cb(msg, progress):
                    self.update_status_safe(f"aimask_status:{msg}", progress)
                
                img_rgb = np.array(self.input_image_pil)
                masks, confidences = self.mask_engine.generate_masks(
                    img_rgb,
                    confidence_threshold=sens,
                    progress_callback=progress_cb
                )
                
                self.gui_queue.put({
                    "type": "aimask_complete",
                    "masks": masks,
                    "confidences": confidences
                })
            except Exception as e:
                err_msg = str(e)
                self.gui_queue.put({"type": "aimask_error", "error": err_msg})

        threading.Thread(target=task, daemon=True).start()

    def on_aimask_process_complete(self, masks, confidences):
        self.processing = False
        self.aimask_progressbar.pack_forget()
        self.aimask_run_btn.configure(state="normal")
        self.aimask_save_btn.configure(state="normal")
        self.aimask_sens_slider.configure(state="normal")
        self.manual_target_combo.configure(state="normal")
        
        previous_manual_masks = {}
        for cat, mask in self.aimask_manual_masks.items():
            previous_manual_masks[cat] = mask.copy()
        previous_manual_erase_masks = {}
        for cat, mask in self.aimask_manual_erase_masks.items():
            previous_manual_erase_masks[cat] = mask.copy()

        self.aimask_masks = masks
        self.aimask_confidences = confidences
        self.aimask_manual_masks = previous_manual_masks
        self.aimask_manual_erase_masks = previous_manual_erase_masks
        if self.input_image_pil is not None:
            for cat in self.aimask_enabled_categories.keys():
                if cat not in self.aimask_manual_masks:
                    self.aimask_manual_masks[cat] = np.zeros((self.input_image_pil.height, self.input_image_pil.width), dtype=np.uint8)
                if cat not in self.aimask_manual_erase_masks:
                    self.aimask_manual_erase_masks[cat] = np.zeros((self.input_image_pil.height, self.input_image_pil.width), dtype=np.uint8)
        
        # 信頼度スコアをラベルに反映
        for cat, score in confidences.items():
            lbl = self.aimask_layer_score_labels.get(cat)
            if lbl:
                if self.aimask_layer_has_manual_edit(cat):
                    lbl.configure(text="手動編集あり")
                elif score > 0.0:
                    lbl.configure(text=f"信頼度: {score:.2f}")
                else:
                    lbl.configure(text="未検出")
                    
        self.aimask_status_label.configure(
            text="AI解析が完了しました。",
            text_color=COLOR_ACCENT
        )
        self.update_aimask_preview()
        self.update_upscale_mask_label()

    def on_aimask_process_error(self, err_msg):
        self.processing = False
        self.aimask_progressbar.pack_forget()
        self.aimask_run_btn.configure(state="normal")
        self.aimask_save_btn.configure(state="disabled")
        self.aimask_sens_slider.configure(state="normal")
        self.manual_target_combo.configure(state="normal")
        
        self.aimask_status_label.configure(text="解析中にエラーが発生しました。", text_color="#EF4444")
        messagebox.showerror("解析エラー", f"AIマスク解析中にエラーが発生しました:\n{err_msg}")

    def save_aimasks(self):
        if self.input_file_path is None:
            return
        merged_masks = self.get_merged_aimask_masks()
        if merged_masks is None:
            return
            
        base, _ = os.path.splitext(os.path.basename(self.input_file_path))
        default_dir = os.path.dirname(self.input_file_path)
        
        selected_dir = filedialog.askdirectory(
            initialdir=default_dir,
            title="マスク画像を保存するフォルダを選択してください"
        )
        
        if selected_dir:
            try:
                saved_count = 0
                for cat, is_enabled in self.aimask_enabled_categories.items():
                    if is_enabled and merged_masks and cat in merged_masks:
                        mask = merged_masks[cat]
                        if np.any(mask > 0):
                            save_path = os.path.join(selected_dir, f"{base}_{cat.lower()}_mask.png")
                            # 日本語パスでも安全に保存するために OpenCV を利用
                            _, img_encoded = cv2.imencode(".png", mask)
                            img_encoded.tofile(save_path)
                            saved_count += 1
                
                if saved_count > 0:
                    messagebox.showinfo("成功", f"{saved_count} 個のマスク画像を保存しました。")
                else:
                    messagebox.showwarning("警告", "有効な（かつ検出された）マスクレイヤーが選択されていませんでした。")
            except Exception as e:
                messagebox.showerror("保存エラー", f"マスクの保存中にエラーが発生しました:\n{str(e)}")

    def detect_and_print_gender(self, pil_img):
        """人物画像を読み込んだ際、性別を鑑別してコンソールに出力する"""
        return
        if not self.type_person_var.get():
            return
            
        try:
            # OpenCV形式 (BGR) に変換
            img_rgb = np.array(pil_img)
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            h, w, _ = img_bgr.shape
            
            # MediaPipeによる顔検出
            model_dir = "models"
            det_filename = "blaze_face_full_range.tflite"
            det_path = os.path.join(model_dir, det_filename)
            
            # 簡易的にダウンロードをチェック
            if not os.path.exists(det_path):
                url = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_full_range/float16/latest/blaze_face_full_range.tflite"
                os.makedirs(model_dir, exist_ok=True)
                import urllib.request
                urllib.request.urlretrieve(url, det_path)
            
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision
            
            base_options = mp_python.BaseOptions(model_asset_path=det_path)
            options = mp_vision.FaceDetectorOptions(base_options=base_options, min_detection_confidence=0.5)
            
            with mp_vision.FaceDetector.create_from_options(options) as detector:
                # 顔検出の高速化のため、検出用の縮小画像を生成 (最大長を 1024px に制限)
                max_size = 1024
                scale_factor = 1.0
                if max(h, w) > max_size:
                    scale_factor = max_size / max(h, w)
                    w_det = int(w * scale_factor)
                    h_det = int(h * scale_factor)
                    img_rgb_det = cv2.resize(img_rgb, (w_det, h_det), interpolation=cv2.INTER_AREA)
                else:
                    img_rgb_det = img_rgb
                
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb_det)
                detection_result = detector.detect(mp_image)
                detections = detection_result.detections
                
                if not detections:
                    print("[性別判定] 画像から顔が検出されませんでした。")
                    return
                
                print(f"[性別判定] {len(detections)}個の顔を検出しました。判定中...")
                
                # Caffe 性別判定モデルのロード
                proto_path = os.path.join(model_dir, "gender_deploy.prototxt")
                model_path = os.path.join(model_dir, "gender_net.caffemodel")
                
                proto_url = "https://huggingface.co/AjaySharma/genderDetection/resolve/main/gender_deploy.prototxt"
                model_url = "https://huggingface.co/AjaySharma/genderDetection/resolve/main/gender_net.caffemodel"
                
                if not os.path.exists(proto_path):
                    print("[性別判定] 設定ファイル (gender_deploy.prototxt) をダウンロードしています...")
                    import urllib.request
                    urllib.request.urlretrieve(proto_url, proto_path)
                    
                if not os.path.exists(model_path):
                    print("[性別判定] 判定モデル (gender_net.caffemodel) をダウンロードしています...")
                    import urllib.request
                    urllib.request.urlretrieve(model_url, model_path)
                
                # OpenCV DNNでロード
                net = cv2.dnn.readNetFromCaffe(proto_path, model_path)
                
                for idx, det in enumerate(detections):
                    bb = det.bounding_box
                    keypoints = det.keypoints
                    
                    # 縮小画像での検出座標を、元の元画像スケールに逆換算する
                    orig_bb_origin_x = bb.origin_x / scale_factor
                    orig_bb_origin_y = bb.origin_y / scale_factor
                    orig_bb_width = bb.width / scale_factor
                    orig_bb_height = bb.height / scale_factor
                    
                    # 目の位置情報を取得してアライメントを行う (傾き補正)
                    M = None
                    if len(keypoints) >= 2:
                        left_eye_pt = np.array([keypoints[0].x * w, keypoints[0].y * h])
                        right_eye_pt = np.array([keypoints[1].x * w, keypoints[1].y * h])
                        
                        # 左右の目の角度と距離を計算
                        dy = right_eye_pt[1] - left_eye_pt[1]
                        dx = right_eye_pt[0] - left_eye_pt[0]
                        angle = np.degrees(np.arctan2(dy, dx))
                        
                        desired_left_x = 0.35
                        desired_right_x = 0.65
                        desired_y = 0.40
                        desired_dist = desired_right_x - desired_left_x
                        
                        eye_dist = np.sqrt(dx**2 + dy**2)
                        if eye_dist >= 1e-5:
                            # 227x227サイズへアライメント用の変換行列を作成
                            scale = (227 * desired_dist) / eye_dist
                            eye_center = (left_eye_pt + right_eye_pt) / 2.0
                            desired_center = (227 * 0.5, 227 * desired_y)
                            
                            M = cv2.getRotationMatrix2D(tuple(eye_center), angle, scale)
                            M[0, 2] = desired_center[0] - (M[0, 0] * eye_center[0] + M[0, 1] * eye_center[1])
                            M[1, 2] = desired_center[1] - (M[1, 0] * eye_center[0] + M[1, 1] * eye_center[1])
                            
                    if M is not None:
                        # 傾きと位置を正面に補正した顔画像を生成
                        face_bgr = cv2.warpAffine(img_bgr, M, (227, 227), flags=cv2.INTER_LANCZOS4)
                    else:
                        # アライメントできない場合のフォールバック（通常の切り出し）
                        margin = 0.2
                        bx0 = max(0, int(orig_bb_origin_x - orig_bb_width * margin))
                        by0 = max(0, int(orig_bb_origin_y - orig_bb_height * margin))
                        bx1 = min(w, int(orig_bb_origin_x + orig_bb_width * (1.0 + margin)))
                        by1 = min(h, int(orig_bb_origin_y + orig_bb_height * (1.0 + margin)))
                        if (bx1 - bx0) > 10 and (by1 - by0) > 10:
                            face_bgr = img_bgr[by0:by1, bx0:bx1]
                            face_bgr = cv2.resize(face_bgr, (227, 227))
                        else:
                            continue
                            
                    # OpenCV DNN用に前処理 (すでに 227x227 にアライメント済み)
                    blob = cv2.dnn.blobFromImage(face_bgr, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
                    net.setInput(blob)
                    gender_preds = net.forward()
                    
                    # すでに Softmax 済みのため、そのまま確率として使用
                    probs = gender_preds[0].flatten()
                    
                    gender_str = "男性" if probs[0] > probs[1] else "女性"
                    prob = probs[0] if probs[0] > probs[1] else probs[1]
                    print(f"[性別判定] 顔 #{idx+1}: {gender_str} (確信度: {prob*100:.1f}%)")
        except Exception as e:
            print(f"[性別判定] エラーが発生しました: {str(e)}")


    def load_input_image(self, file_path):
        try:
            self.input_file_path = file_path
            self.input_image_pil = Image.open(file_path).convert("RGB")
            
            self.placeholder_frame.pack_forget()
            self.compare_slider.pack(fill="both", expand=True, padx=10, pady=10)
            self.compare_slider.set_images(self.input_image_pil, self.input_image_pil)
            
            self.status_label.configure(
                text=f"ロード完了: {os.path.basename(file_path)} ({self.input_image_pil.width}x{self.input_image_pil.height})",
                text_color=COLOR_TEXT
            )
            
            self.process_btn.configure(state="normal")
            self.save_btn.configure(state="disabled")
            self.output_image_pil = None

            # AIマスク生成側の状態初期化
            self.aimask_placeholder.pack_forget()
            self.aimask_preview.pack(fill="both", expand=True, padx=10, pady=10)
            self.aimask_preview.set_image(self.input_image_pil)
            self.aimask_run_btn.configure(state="normal")
            self.aimask_save_btn.configure(state="disabled")
            self.aimask_masks = None
            self.aimask_confidences = None
            self.aimask_manual_masks = {}
            self.aimask_manual_erase_masks = {}
            for cat in self.aimask_layer_score_labels:
                self.aimask_layer_score_labels[cat].configure(text="-")
            self.manual_target_combo.set("Text")
            self.aimask_manual_target_var.set("Text")
            self.aimask_status_label.configure(
                text="解析ボタンを押すとAIセグメンテーションを開始します",
                text_color=COLOR_TEXT
            )
            self.update_upscale_mask_label()
            
        except Exception as e:
            messagebox.showerror("エラー", f"画像の読み込みに失敗しました: {str(e)}")

    def on_model_changed(self, selected_model):
        """モデル切り替えイベント"""
        self.engine.set_model(selected_model)
        self.init_model_async()

    def on_crop_changed(self, selected_ratio):
        """トリミング比率変更イベント"""
        self.compare_slider.set_crop_ratio(selected_ratio)

    # ==========================================
    # D. 非同期モデル初期化 & 画像処理
    # ==========================================
    def init_model_async(self):
        """起動時またはモデル変更時にバックグラウンドでロード・ビルドを実行"""
        if self.processing:
            return
            
        self.processing = True
        self.process_btn.configure(state="disabled")
        self.model_segment.configure(state="disabled")
        self.open_btn.configure(state="disabled")
        if hasattr(self, 'face_restore_cb'):
            self.face_restore_cb.configure(state="disabled")
        if hasattr(self, 'face_restore_slider'):
            self.face_restore_slider.configure(state="disabled")
        if hasattr(self, 'save_same_rb'):
            self.save_same_rb.configure(state="disabled")
        if hasattr(self, 'save_custom_rb'):
            self.save_custom_rb.configure(state="disabled")
        self.toggle_save_dir_ui()
        
        self.progressbar.pack(fill="x", pady=(0, 10), after=self.status_label)
        self.progressbar.configure(mode="determinate")
        self.progressbar.set(0)

        def task():
            try:
                def progress_cb(msg, progress):
                    self.update_status_safe(msg, progress)
                
                mode_str = self.engine.load_model(progress_callback=progress_cb)
                self.gui_queue.put({"type": "model_loaded", "mode": mode_str})
            except Exception as e:
                err_msg = str(e)
                self.gui_queue.put({"type": "model_error", "error": err_msg})

        threading.Thread(target=task, daemon=True).start()

    def on_model_load_complete(self, mode_str):
        self.processing = False
        self.progressbar.pack_forget()
        self.model_segment.configure(state="normal")
        self.open_btn.configure(state="normal")
        if hasattr(self, 'face_restore_cb'):
            self.face_restore_cb.configure(state="normal")
        self.toggle_face_restore_slider()
        if hasattr(self, 'save_same_rb'):
            self.save_same_rb.configure(state="normal")
        if hasattr(self, 'save_custom_rb'):
            self.save_custom_rb.configure(state="normal")
        self.toggle_save_dir_ui()
        if self.input_image_pil:
            self.process_btn.configure(state="normal")
            
        self.status_label.configure(
            text=f"動作モード: {mode_str} | 準備完了",
            text_color=COLOR_ACCENT
        )

    def on_model_load_error(self, err_msg):
        self.processing = False
        self.progressbar.pack_forget()
        self.model_segment.configure(state="normal")
        self.open_btn.configure(state="normal")
        if hasattr(self, 'face_restore_cb'):
            self.face_restore_cb.configure(state="normal")
        self.toggle_face_restore_slider()
        if hasattr(self, 'save_same_rb'):
            self.save_same_rb.configure(state="normal")
        if hasattr(self, 'save_custom_rb'):
            self.save_custom_rb.configure(state="normal")
        self.toggle_save_dir_ui()
        self.status_label.configure(text=f"モデルエラー: {err_msg}", text_color="#EF4444")
        messagebox.showerror("モデルロードエラー", f"モデルの読み込みまたはTensorRTビルドに失敗しました:\n{err_msg}")

    def process_gui_queue(self):
        """100msごとにキューをチェックしてメインスレッドでGUIを安全に更新する"""
        try:
            while True:
                msg_data = self.gui_queue.get_nowait()
                msg_type = msg_data.get("type")
                
                if msg_type == "status":
                    text = msg_data.get("text", "")
                    progress = msg_data.get("progress", 0.0)
                    is_error = msg_data.get("is_error", False)
                    color = "#EF4444" if is_error else COLOR_TEXT_MUTED
                    
                    if text.startswith("aimask_status:"):
                        display_text = text.replace("aimask_status:", "")
                        self.aimask_status_label.configure(text=display_text, text_color=color)
                        self.aimask_progressbar.set(progress)
                    else:
                        self.status_label.configure(text=text, text_color=color)
                        self.progressbar.set(progress)
                    
                elif msg_type == "complete":
                    output_img = msg_data.get("image")
                    self.on_process_complete(output_img)
                    
                elif msg_type == "error":
                    err_msg = msg_data.get("error")
                    self.on_process_error(err_msg)
                    
                elif msg_type == "model_loaded":
                    mode_str = msg_data.get("mode")
                    self.on_model_load_complete(mode_str)
                    
                elif msg_type == "model_error":
                    err_msg = msg_data.get("error")
                    self.on_model_load_error(err_msg)

                elif msg_type == "aimask_complete":
                    masks = msg_data.get("masks")
                    confidences = msg_data.get("confidences")
                    self.on_aimask_process_complete(masks, confidences)

                elif msg_type == "aimask_error":
                    err_msg = msg_data.get("error")
                    self.on_aimask_process_error(err_msg)

                elif msg_type == "scan_status":
                    text = msg_data.get("text", "")
                    progress = msg_data.get("progress", 0.0)
                    self.scan_status_label.configure(text=text, text_color=COLOR_TEXT_MUTED)
                    self.scan_progressbar.set(progress)

                elif msg_type == "scan_complete":
                    outputs = msg_data.get("outputs", [])
                    self.on_scan_process_complete(outputs)

                elif msg_type == "scan_error":
                    err_msg = msg_data.get("error")
                    self.on_scan_process_error(err_msg)
                    
                self.gui_queue.task_done()
        except queue.Empty:
            pass
            
        # 100ms後に再呼び出し
        self.after(100, self.process_gui_queue)

    def update_status_safe(self, text, progress_val=0.0, is_error=False):
        """別スレッドから安全にステータス文字列とプログレスバーを更新するためにキューへ投入"""
        self.gui_queue.put({
            "type": "status",
            "text": text,
            "progress": progress_val,
            "is_error": is_error
        })

    def start_process_async(self):
        """画像処理を非同期で開始"""
        if self.processing or not self.input_image_pil:
            return
            
        self.processing = True
        self.process_btn.configure(state="disabled")
        self.save_btn.configure(state="disabled")
        self.model_segment.configure(state="disabled")
        self.open_btn.configure(state="disabled")
        self.face_restore_cb.configure(state="disabled")
        if hasattr(self, 'face_restore_slider'):
            self.face_restore_slider.configure(state="disabled")
        if hasattr(self, 'save_same_rb'):
            self.save_same_rb.configure(state="disabled")
        if hasattr(self, 'save_custom_rb'):
            self.save_custom_rb.configure(state="disabled")
        self.toggle_save_dir_ui()
        
        # プログレスバー（シークバー）を表示
        self.progressbar.pack(fill="x", pady=(0, 10), after=self.status_label)
        self.progressbar.configure(mode="determinate")
        self.progressbar.set(0)
        
        scale_str = self.scale_var.get()
        scale = 4 if scale_str == "4x" else 2
        denoise = int(self.denoise_slider.get())
        sharpness = int(self.sharp_slider.get())
        face_fidelity = float(self.face_restore_slider.get()) / 100.0 if hasattr(self, 'face_restore_slider') else 0.7
        region_masks = self.get_upscale_region_masks()
        self.update_upscale_mask_label()
        if region_masks:
            self.status_label.configure(text="AIマスクを反映して高画質化します...", text_color=COLOR_ACCENT)
        
        def task():
            try:
                def progress_cb(msg, progress):
                    self.update_status_safe(msg, progress)
                
                # 「人物」または「動物」のいずれかが選択されている場合に背景シャープネス低減（被写体分離）を有効にする
                is_creature = self.type_person_var.get() or self.type_animal_var.get()

                
                out_pil = self.engine.process_image(
                    self.input_file_path,
                    scale=scale,
                    denoise_strength=denoise,
                    sharpness_strength=sharpness,
                    face_restoration=self.face_restore_var.get(),
                    face_restoration_fidelity=face_fidelity,
                    is_creature=is_creature,
                    region_masks=region_masks,
                    progress_callback=progress_cb
                )
                
                self.gui_queue.put({"type": "complete", "image": out_pil})
            except Exception as e:
                err_msg = str(e)
                self.gui_queue.put({"type": "error", "error": err_msg})

        threading.Thread(target=task, daemon=True).start()


    def on_process_complete(self, output_img):
        self.processing = False
        self.progressbar.pack_forget()
        self.model_segment.configure(state="normal")
        self.open_btn.configure(state="normal")
        self.face_restore_cb.configure(state="normal")
        self.toggle_face_restore_slider()
        if hasattr(self, 'save_same_rb'):
            self.save_same_rb.configure(state="normal")
        if hasattr(self, 'save_custom_rb'):
            self.save_custom_rb.configure(state="normal")
        self.toggle_save_dir_ui()
        self.output_image_pil = output_img
        
        self.compare_slider.set_images(self.input_image_pil, self.output_image_pil, reset_view=False)
        
        self.process_btn.configure(state="normal")
        self.save_btn.configure(state="normal")
        self.status_label.configure(
            text=f"高画質化完了: {self.output_image_pil.width}x{self.output_image_pil.height}",
            text_color=COLOR_ACCENT
        )

    def on_process_error(self, err_msg):
        self.processing = False
        self.progressbar.pack_forget()
        self.model_segment.configure(state="normal")
        self.open_btn.configure(state="normal")
        self.face_restore_cb.configure(state="normal")
        self.toggle_face_restore_slider()
        if hasattr(self, 'save_same_rb'):
            self.save_same_rb.configure(state="normal")
        if hasattr(self, 'save_custom_rb'):
            self.save_custom_rb.configure(state="normal")
        self.toggle_save_dir_ui()
        
        self.process_btn.configure(state="normal")
        self.status_label.configure(text="処理エラーが発生しました。", text_color="#EF4444")
        messagebox.showerror("処理エラー", f"画像処理中にエラーが発生しました:\n{err_msg}")


    def save_image(self):
        """処理後の画像をファイルに保存"""
        if not self.output_image_pil:
            return
            
        # デロットファイル名の作成 (例: original_name_2x_sharp.png)
        base, ext = os.path.splitext(os.path.basename(self.input_file_path))
        scale_str = self.scale_var.get()
        default_name = f"{base}_{scale_str}_upscaled.png"
        
        # 初期ディレクトリの取得
        if self.save_dir_mode_var.get() == "custom" and self.custom_save_dir_var.get():
            initial_dir = self.custom_save_dir_var.get()
        else:
            initial_dir = os.path.dirname(self.input_file_path) if self.input_file_path else None
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png"), ("JPEG Image", "*.jpg;*.jpeg")],
            initialdir=initial_dir,
            initialfile=default_name
        )
        
        if file_path:
            try:
                # 保存前に指定比率でトリミング
                final_image = self.compare_slider.get_cropped_image(self.output_image_pil)
                final_image.save(file_path)
                messagebox.showinfo("成功", f"画像を保存しました:\n{os.path.basename(file_path)}")
                
                # 保存後にプレビューをリセットしてファイルを閉じる
                self.close_current_file()
            except Exception as e:
                messagebox.showerror("保存エラー", f"画像の保存に失敗しました:\n{str(e)}")

    def close_current_file(self):
        """現在開いているファイルを閉じ、プレビューとUI状態をリセットする"""
        self.input_file_path = None
        self.input_image_pil = None
        self.output_image_pil = None
        
        # プレビューコンポーネントを非表示にし、プレースホルダーを表示
        self.compare_slider.pack_forget()
        self.placeholder_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # ボタン状態のリセット
        self.process_btn.configure(state="disabled")
        self.save_btn.configure(state="disabled")
        
        # ステータスラベルの更新
        self.status_label.configure(
            text="画像を保存しました。次の画像を読み込んでください。",
            text_color=COLOR_TEXT_MUTED
        )

        # AIマスク生成側のリセット
        self.aimask_preview.pack_forget()
        self.aimask_placeholder.pack(fill="both", expand=True, padx=30, pady=30)
        self.aimask_run_btn.configure(state="disabled")
        self.aimask_save_btn.configure(state="disabled")
        self.aimask_masks = None
        self.aimask_confidences = None
        self.aimask_manual_masks = {}
        self.aimask_manual_erase_masks = {}
        for cat in self.aimask_layer_score_labels:
            self.aimask_layer_score_labels[cat].configure(text="-")
        self.aimask_status_label.configure(
            text="画像をロードすると解析を開始できます",
            text_color=COLOR_TEXT_MUTED
        )
        self.update_upscale_mask_label()


# ==========================================
# E. メインエントリーポイント
# ==========================================
if __name__ == "__main__":
    app = UpscalerApp()
    app.mainloop()
