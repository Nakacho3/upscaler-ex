# D&D機能の修正および高画質化ハングアップ対策

ドラッグ＆ドロップ（D&D）機能が動作しない問題と、高画質化実行時に処理が停止（ハングアップ）する問題を解消するための修正を行います。

## 変更の目的と概要

1. **D&D機能のバインド先修正**
   CustomTkinterのフレーム (`ctk.CTkFrame`) に直接 `drop_target_register` メソッドを呼び出そうとすると `AttributeError` 等でセットアップが失敗するため、内部の Canvas ウィジェット (`_canvas`) および `TkinterDnD.DnDWrapper` を通して安全に登録されるように `setup_drag_and_drop` メソッドを修正します。

2. **ONNXRuntimeのCPUプロバイダーへのフォールバック（ハングアップ回避）**
   顔復元処理 (`gfpgan_onnx.py`) を行う ONNXRuntime にて、CUDA実行プロバイダーを使用しようとした際に DLL や cuDNN のバージョン不整合でプロセスが無限待機（デッドロック）するケースがあります。これを回避するため、実行プロバイダーを `CPUExecutionProvider` のみに変更し、確実に処理が完了するようにします。

---

## 提案される変更

### メインアプリケーション

#### [MODIFY] [main.py](file:///h:/My%20Apps/upscaler ex/main.py)

`main.py` の `setup_drag_and_drop` メソッドを、以下のように CustomTkinter の構造および `DnDWrapper` に対応した記述へ修正します。

```python
    def setup_drag_and_drop(self):
        """ドラッグ＆ドロップのイベントをウィンドウとプレースホルダーにバインドする"""
        if TKDND_AVAILABLE:
            try:
                # ウィンドウ自身への登録 (DnDWrapper経由)
                TkinterDnD.DnDWrapper.drop_target_register(self, DND_FILES)
                self.dnd_bind("<<Drop>>", self.on_file_drop)
                
                # placeholder_frame のインナーウィジェット (Canvas) への登録
                inner_widget = self.placeholder_frame
                if hasattr(self.placeholder_frame, "_canvas"):
                    inner_widget = self.placeholder_frame._canvas
                
                TkinterDnD.DnDWrapper.drop_target_register(inner_widget, DND_FILES)
                inner_widget.dnd_bind("<<Drop>>", self.on_file_drop)
            except Exception as e:
                print(f"Failed to setup drag and drop: {e}")
```

### 顔復元モジュール

#### [MODIFY] [gfpgan_onnx.py](file:///h:/My%20Apps/upscaler ex/gfpgan_onnx.py)

`gfpgan_onnx.py` の `load_model` 内の `providers` 指定（134行目付近）を `CPUExecutionProvider` のみに制限します。

```python
        # CUDAExecutionProvider のロード時に cuDNN との不整合でハングするのを防ぐため、CPUプロバイダのみを使用
        providers = ['CPUExecutionProvider']
```

---

## 検証計画

### 手動検証
1. `run_app.bat` を実行して起動し、ドラッグ＆ドロップで画像ファイルをドロップした際に画像が正常に読み込めることを確認します。
2. 画像をロード後、「人物の顔・瞳を自然に修復」をONにして「画像を高画質化する」を実行し、処理が停止せずに100%まで進行して完了することを確認します。
3. 高画質化完了後、人物の顔が検出され、適切に復元処理が施されているかプレビュー画面で確認します。
