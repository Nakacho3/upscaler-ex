# プロジェクト引き継ぎ仕様書 (Handover Notes)

本ドキュメントは、画像アップスケーリングおよびAIセグメンテーションマスク生成アプリケーション「**Upscaler EX**」の開発を、他のAIエージェントに引き継ぐための仕様書である。

---

## 1. アプリケーション概要と構成

**Upscaler EX** は、以下の2つの主要機能を提供するデスクトップアプリケーションである。

1. **アップスケール（Upscale）機能**:
   - 画像全体の超解像処理（アップスケーリング）および顔領域の高精細な復元を行う。
2. **AIマスク生成（AI Mask）機能**:
   - セマンティックセグメンテーションを用いて、画像内の構成要素（人物、顔、オブジェクト、テキスト、空、背景など）を自動解析し、領域別のマスク（2値画像）を生成する。
   - 生成したマスクは、将来的にアップスケーリング実行時に領域ごとのパラメータ（人物のみ高精細化、背景はデノイズのみ等）を適用するためのベースデータとして活用される。

### 画面構成（タブUI）
メインウィンドウは `customtkinter` を用いたダークテーマUIで構築されており、最上部のタブコントロール (`ctk.CTkTabview`) で「アップスケール」と「AIマスク生成」を切り替えることができる。

---

## 2. 主要コンポーネントと役割

プロジェクトルート配下の主要コードとその役割は以下の通りである。

* **[main.py](file:///h:/My%20Apps/upscaler%20ex/main.py)**
  - アプリケーションのエントリーポイントおよびGUIの実装。
  - `customtkinter` (ctk) ベース of UIウィジェット構築。
  - ドラッグ＆ドロップ対応 (`tkinterdnd2` 使用)。
  - 解析処理の非同期実行（スレッド分離によるUIフリーズ防止と進捗バーの制御）。
  - **[MaskPreviewCanvas](file:///h:/My%20Apps/upscaler%20ex/main.py#L19)**: マスク表示用のカスタムキャンバス。マウス操作による画像のパン（平行移動）、ホイールによるズームに対応。
* **[ai_mask_engine.py](file:///h:/My%20Apps/upscaler%20ex/ai_mask_engine.py)**
  - AIマスク生成のコアエンジン。
  - 各種AIモデルや画像処理技術を組み合わせてカテゴリ別のマスク画像（NumPy配列）を生成する。
  - マスクをプレビュー表示用にカラーオーバーレイブレンドするヘルパー関数を提供。
* **[upscaler.py](file:///h:/My%20Apps/upscaler%20ex/upscaler.py)**
  - 画像のアップスケーリング処理のコアロジック。
  - CuPyやTensorRT等のGPU高速化技術を利用可能。
* **[gfpgan_onnx.py](file:///h:/My%20Apps/upscaler%20ex/gfpgan_onnx.py)**
  - ONNXランタイムを利用したGFPGANモデルによる顔修復処理の実装。
* **[test_face_detect.py](file:///h:/My%20Apps/upscaler%20ex/test_face_detect.py)**
  - 顔検出およびGFPGANによる顔復元機能の動作検証用テストスクリプト。

---

## 3. 現在の実装状況と検証結果

### AIマスク生成機能（v0.1 - 基本実装完了）
- **人物（Person）/ 背景（Background）**: `models/selfie_segmenter.tflite` (MediaPipe Tasks API) を用いてGPU/CPU上で推論。背景は人物領域の反転として生成。
- **顔（Face）**: `models/blaze_face_full_range.tflite` または `blaze_face_short_range.tflite` (MediaPipe) で顔領域を検出し、楕円としてマスク化。
- **テキスト（Text）**: OpenCV MSER (最大安定極値領域) を利用した文字領域の簡易輪郭抽出。
- **空（Sky）**: HSV色空間（青空・白い空の境界値）および「画像の上部に位置する」という空間プライアに基づく簡易抽出。
- **オブジェクト（Object）**: 前景部のCannyエッジ＋輪郭検出から、人物・空・テキスト領域を除外した領域を簡易抽出。
- **プレビューとエクスポート**: 各マスクは半透明カラーでオーバーレイ表示され、チェックボックスで表示/非表示を切り替え可能。また、「💾 解析結果(マスク)を保存」ボタンから `[画像名]_[カテゴリ名]_mask.png` として一括保存可能。

---

## 4. 進行中のタスク（未着手・最優先タスク）

現在、AIマスクの精度向上およびCUDA高速化のための計画書 **[ai_mask_cuda_and_text_fix/implementation_plan.md](file:///h:/My%20Apps/upscaler%20ex/docs/ai_mask_cuda_and_text_fix/implementation_plan.md)** が作成されており、次のAIエージェントはこの計画の実行を担う。

### 計画の背景と課題
1. **テキスト検出の誤検出**: 従来のMSERによるテキスト検出は、衣服の模様、建物の格子、木の葉の影などの細かなテクスチャを大量に誤検出してしまう課題がある。
2. **GPU (CUDA) の未活用**: セグメンテーション（人物等）にTFLiteのCPUベースモデル（または非効率な推論）を使用しており、RTX GPU（Tensor Core / CUDA Toolkit）の性能を活かしきれていない。また、ObjectやSkyなどの抽出が古典的なヒューリスティック画像処理に依存しているため精度が低い。

### 具体的な変更予定（`ai_mask_engine.py` の修正）
1. **PyTorch & ONNX Runtime の CUDA 連携**
   - `torch.cuda.is_available()` および ONNX Runtime の `CUDAExecutionProvider` を検出し、GPU上での推論を実行する。
2. **PyTorchによる人物（Person）＆オブジェクト（Object）マスクの高度化**
   - `torchvision.models.segmentation` の `deeplabv3_resnet50`（または軽量な `lraspp_mobilenet_v3_large`）を使用。
   - COCOクラスの「person (ID: 15)」を高精度にセグメンテーションして人物マスクとする。
   - 「人物以外の全COCOオブジェクト（乗り物、動物、家具等）」をマージしてオブジェクトマスクとすることで、境界を精密に抽出する。
3. **EASTテキスト検出器 (ONNX) によるテキストマスクの劇的改善**
   - 誤検出の多いMSERを廃止し、ディープラーニングベースの **EASTテキスト検出モデル (ONNX形式)** に置き換える。
   - ONNX Runtime（CUDA）で推論を行い、信頼度スコアと NMS（Non-Maximum Suppression）を適用して、真のテキスト領域のみを矩形/傾いた矩形として正確に抽出する。
4. **顔検出（Face）の安定化**
   - GFPGAN側で使用している実績があり、スレッドのデッドロックやハングを起こしにくい `mp.solutions.face_detection`（MediaPipe Legacy API）に置き換える。
5. **堅牢なフォールバック設計**
   - CUDAエラー、VRAM不足、またはオフライン環境（モデル未ダウンロード等）の例外を捕捉し、従来のMediaPipe Tasks (TFLite CPU) や MSER などの簡易検出へと自動的にフォールバックさせ、クラッシュを防止する。

---

## 5. 開発環境とモデルファイル一覧

### 必要ライブラリ (`requirements.txt` / 追加パッケージ)
- `customtkinter`
- `opencv-python`
- `numpy`
- `pillow`
- `mediapipe`
- `torch` (CUDA版) / `torchvision` (CUDA版)
- `onnxruntime-gpu` (CUDA Toolkitとバージョンが一致していること)

### 使用モデルファイルと設置先
モデルはプロジェクトルート配下の `models/` フォルダに配置または自動ダウンロードされる。

| モデル名 | 用途 | 取得方法 / 配置 | ファイル名 (例) |
|---|---|---|---|
| **Selfie Segmenter** | 人物/背景セグメンテーション (従来のフォールバック用) | `models/` に事前配置済み | `selfie_segmenter.tflite` |
| **BlazeFace** | 顔検出 (従来のフォールバック用) | `models/` に事前配置済み | `blaze_face_full_range.tflite` |
| **DeepLabV3 / LRASPP** | 高精度な人物・オブジェクト抽出 (PyTorch) | 初回推論時に PyTorch サーバーから `~/.cache/torch/` に自動ダウンロード | - |
| **EAST Text Detector** | テキスト検出 (ONNX) | 初回推論時に `models/` へ自動ダウンロード (約94MB) | `east_text_detection.onnx` (または `frozen_east_text_detection.onnx`) |

---

## 6. 開発上の制約事項とルール

次の開発を行うAIエージェントは、以下のプロジェクトルールを厳守すること。

- **日本語でのコミュニケーション・ドキュメント管理**:
  - すべてのチャット応答、実装計画書、動作確認書、タスク管理は**日本語**で行う。
- **コード編集時の限定的修正とバックアップ**:
  - 不要なリファクタリングや最適化（変数名の変更やインデントスタイルの変更など）は行わず、指示された箇所以外は一切書き換えない。
  - 大幅な書き換え（100行超）を行う前には、必ず対象ファイルのバックアップ（`.bak`）を作成すること。既に `.bak` が存在する場合は、`filename.py.bak1` などの連番を付与し、常に無印の `.bak` が最新のバックアップになるようにリネームする。
- **承認プロセスの順守**:
  - 変更を行う前に必ず `implementation_plan.md` を作成し、ユーザーの「Approve」ボタンの押下を待ってから実行（修正コードの適用）を行う。
- **フォールバックの徹底**:
  - PyTorch や ONNX Runtime で発生し得る「VRAM不足」「CUDAライブラリのミスマッチ」「ネットワーク切断によるダウンロード失敗」に対し、必ず `try-except` で囲み、CPUやMediaPipe Tasksなどのレガシー実装へ自動フォールバックさせる。

---

## 7. 次のステップ（実装手順の推奨）

1. **単体検証スクリプトの作成**:
   - `ai_mask_engine.py` を修正する前に、動作検証用のスクリプト `test_ai_mask_cuda.py` を新規作成する。
   - このスクリプトは、任意の画像を入力して PyTorch (CUDA) や EAST ONNX (CUDA) による推論が通るか、およびエラー発生時のフォールバックが機能するかを標準出力ログとマスク画像の保存で確認できるものとする。
2. **`ai_mask_engine.py` の実装と修正**:
   - `implementation_plan.md` に従い、PyTorch セグメンテーションおよび EAST ONNX を組み込んだ推論ロジックを実装する。
3. **`main.py` との結合および手動テスト**:
   - アプリを起動し、UIから「AIマスク解析」を実行し、進捗バーの動作、誤検出の減少、およびVRAM使用量が想定範囲内であるかを確認する。
