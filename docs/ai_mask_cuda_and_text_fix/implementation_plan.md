# AIマスク精度の向上およびCUDA Toolkit連携の実装計画

本計画は、AIマスク生成エンジンの精度を向上させ、NVIDIA RTXシリーズのCUDAテクノロジー（CUDA Toolkit）を最大限活用して処理を高速・高精度化するとともに、関係のないテクスチャを文字・テキスト領域として誤検出する確率を劇的に低減させるためのものです。

---

## ユーザー確認事項

> [!IMPORTANT]
> **1. 新しいモデルファイルの自動ダウンロード**
> テキスト検出を従来の古典的画像処理（MSER）から、ディープラーニングベースの「EASTテキスト検出モデル（ONNX形式）」に変更します。このモデル（約94MB）は、初回の解析実行時にインターネットから自動的にダウンロードされます（ミラーURLをサポート）。
>
> **2. Torchvisionによるセグメンテーションモデルの導入**
> 人物（Person）およびオブジェクト（Object）の抽出精度を極限まで高めるため、PyTorch CUDA上で直接動作する `torchvision` の `deeplabv3_resnet50`（または軽量な `lraspp_mobilenet_v3_large`）を使用します。初回ロード時にPyTorchの公式サーバーからモデル（約160MB / 13MB）が自動的にダウンロードされます。
>
> **3. 堅牢なフォールバック設計**
> ネットワークがオフラインの場合や、CUDA Toolkitが適切に動作しない環境（VRAM不足含む）では、自動的に従来のMediaPipe（CPU）やMSERによる簡易抽出に切り替わるフォールバック処理を実装し、アプリのクラッシュを完全に防ぎます。

---

## 提案する変更内容

### AIマスクエンジンコンポーネント

#### [MODIFY] [ai_mask_engine.py](file:///h:/My%20Apps/upscaler%20ex/ai_mask_engine.py)

主な変更内容は以下の通りです：

1. **デバイスとランタイムのCUDA対応チェック**
   - PyTorch (`torch.cuda.is_available()`) および ONNX Runtime の CUDAプロバイダ (`'CUDAExecutionProvider'`) を検出し、RTXシリーズのGPU（CUDA Toolkit）を使用して推論を高速化します。

2. **人物（Person）およびオブジェクト（Object）マスクの高度化 (PyTorch CUDA)**
   - `torchvision.models.segmentation` のセグメンテーションモデル（デフォルトで `lraspp_mobilenet_v3_large` または `deeplabv3_resnet50`）を採用。
   - 人物（COCOクラス15: person）を高精度に切り抜きます。
   - オブジェクト（Object）の判定に、従来の「Cannyエッジ＋輪郭検出」ではなく、セグメンテーションモデルが認識した「人物以外の全COCOオブジェクト（乗り物、動物、家具、日用品等）」をマージして使用します。これにより、実用的なオブジェクトマスクが極めて精密に生成されます。

3. **テキスト（Text）マスクの誤検出対策 (EAST ONNX + CUDA)**
   - 従来の誤検出が非常に多かった OpenCV MSER を廃止し、ディープラーニングベースの **EASTテキスト検出器 (ONNXモデル)** に置き換えます。
   - ONNX Runtime を使用し、CUDA Execution Provider で実行します。
   - 推論結果に NMS（Non-Maximum Suppression）を適用してテキストのバウンディングボックスを精密に抽出し、マスクに描画します。これにより、衣服の模様、ビルの窓、木々の影などの誤検出を劇的に削減します。

4. **顔検出（Face）の安定化**
   - `gfpgan_onnx.py` と同じ安定したレガシーAPI `mp.solutions.face_detection`（スレッドハングを起こさない実績のある手法）に置き換え、顔のバウンディングボックスから楕円マスクを正確に生成します。

5. **フォールバック処理の実装**
   - 全てのディープラーニング処理（PyTorch, ONNX Runtime）において、モデルロード失敗やCUDA/VRAMエラー、オフライン時の例外を `try-except` で捕捉し、従来のMediaPipe Tasks APIや簡易処理へ自動的にCPUフォールバックさせます。

---

## 検成・検証計画

### 自動・手動テストの実行
1. `test_face_detect.py` の動作に影響がないことを確認します。
2. 新しい `ai_mask_engine.py` を単体テストするスクリプト `test_ai_mask_cuda.py` をルートに作成し、画像を入力して各マスク（Person, Face, Object, Text, Sky）が正しく生成されること、およびCUDAが使用されていることをログで確認します。
3. `main.py` からアプリを起動し、「AIマスク生成」タブで実際に解析を実行し、UI上で誤検出が減っていること、及びプレビューが正常にオーバーレイされるか確認します。
