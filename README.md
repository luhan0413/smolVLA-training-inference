# SmolVLA training

獨立環境：Python 3.12、LeRobot 0.6.1、SmolVLA base fine-tuning。
資料使用 LeRobotDataset v3.0。資料夾可自行命名，執行時指定資料集根目錄（包含 `meta/info.json` 的那一層）。
專用入口：`DATASET_ROOT=/path/to/dataset bash train_ur7e.sh`；資料結構與操作流程見 [TRAINING_GUIDE.md](TRAINING_GUIDE.md)。

## 環境

請將 `/path/to/smolvla-training` 換成你的專案路徑。

```bash
cd /path/to/smolvla-training
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=2 python -c 'import torch; print(torch.__version__); assert torch.cuda.is_available(), "CUDA unavailable"; print(torch.cuda.get_device_name(0))'
```

需要重建時（需已安裝 uv）：

```bash
uv venv .venv --python python3.12
uv pip install --python .venv/bin/python -r requirements.txt
```

依賴統一保存在 `requirements.txt`，包含固定套件版本與 SmolVLA／training 額外依賴，供同類 Linux/GPU 環境重建。
訓練腳本預設使用 GPU 2；執行前請用 `nvidia-smi` 確認可用 GPU，並透過 `GPU` 環境變數指定。環境檢查則使用 `CUDA_VISIBLE_DEVICES` 指定 GPU。

## 兩個訓練腳本的差別

`train.sh` 負責共用訓練流程；`train_ur7e.sh` 準備 UR7e 專用設定，再呼叫 `train.sh`。兩個檔案都需要保留在同一個資料夾。

| 項目 | `train.sh` | `train_ur7e.sh` |
|---|---|---|
| 用途 | 通用 SmolVLA 訓練入口 | UR7e 專用設定入口 |
| 資料來源 | Hugging Face 資料集，或透過 `DATASET_ROOT` 指定本機資料 | 必須透過 `DATASET_ROOT` 指定本機資料，資料夾名稱不限 |
| 輸入設定 | 未額外指定 UR7e 輸入格式 | 7 維 `observation.state`、`observation.images.left` RGB 影像，形狀為 `[3, 480, 640]` |
| 預設 batch size | 8 | 64 |
| 預設 GPU | 2 | 2 |
| 預設訓練步數 | 20,000 | 20,000 |
| 模型存檔頻率 | 每 5,000 步 | 沿用 `train.sh`，每 5,000 步 |
| 額外設定 | 可傳入額外的 `lerobot-train` 參數 | 關閉 ALOHA 適配與 ALOHA delta joint actions，每 10 步記錄日誌 |

執行流程：

```text
train_ur7e.sh（準備 UR7e 設定）
    → train.sh（檢查環境與資料、設定輸出位置）
    → lerobot-train（開始訓練）
```

符合上述輸入格式的 UR7e 資料可使用 `train_ur7e.sh`。其他資料可使用 `train.sh`，並依資料格式調整訓練參數。兩個入口都會先執行 `check_dataset.py`，檢查失敗就停止。

例如，指定本機資料並調整 GPU、batch size：

```bash
DATASET_ROOT=/path/to/dataset GPU=0 BATCH_SIZE=8 bash train_ur7e.sh
```

先跑兩步確認 UR7e 訓練流程：

```bash
DATASET_ROOT=/path/to/dataset GPU=0 BATCH_SIZE=1 STEPS=2 bash train_ur7e.sh
```

`/path/to/dataset` 請換成實際資料集根目錄，該目錄需包含 `meta/info.json`。`GPU`、`BATCH_SIZE`、`STEPS`、`SAVE_FREQ` 等環境變數可覆蓋預設值。

## 資料到位後

UR7e 訓練必須透過 `DATASET_ROOT` 指定資料集，沒有預設資料夾：

```bash
DATASET_ROOT=/path/to/dataset bash train_ur7e.sh
```

Hub dataset（私有資料先執行 `.venv/bin/hf auth login`）：

```bash
bash train.sh YOUR_HF_NAME/YOUR_DATASET
```

本機 dataset，路徑下需有 meta/info.json 及其引用的 data/videos：

```bash
DATASET_ROOT=/absolute/path/to/dataset bash train.sh local/my_dataset
```

先用兩步檢查完整訓練流程：

```bash
GPU=2 STEPS=2 BATCH_SIZE=1 bash train.sh YOUR_HF_NAME/YOUR_DATASET
```

正式訓練：

```bash
GPU=2 BATCH_SIZE=8 STEPS=20000 bash train.sh YOUR_HF_NAME/YOUR_DATASET
```

執行 `bash train.sh --help` 查看可調整參數。預設不啟用 W&B、不上傳模型。
首次訓練會下載 pretrained model，需網路和足夠磁碟空間。
每次使用獨立 output directory，checkpoint 保存在 outputs/train/ 下。
兩步測試只驗證程式流程，不代表模型已學會任務。

## TCP / joint 資料檢查

模型的 state/action 欄位由資料集決定，這份啟動腳本不預設 TCP 或 joint 語意。
資料到位後需檢查 observation.state、action、RGB 相機、task、FPS、維度及統計量。
相機名稱與 pretrained config 不同時，需設定 rename map。
TCP action 需固定座標系、TCP offset、位置單位、旋轉表示、gripper 定義，
並區分 absolute target、相對於 chunk 起點及逐步 delta。
相對 pose 轉換與部署控制器需另行配對；腳本不會自動做 FK/IK 或 TCP 轉換。
部署時使用訓練 checkpoint 對應的前後處理與 normalization。

官方參考：https://huggingface.co/docs/lerobot/v0.6.1/en/smolvla

## 訓練前資料檢查

檢查資料集時須指定路徑：`.venv/bin/python check_dataset.py /absolute/path/to/dataset`。
train.sh 與 train_ur7e.sh 會自動先做全量檢查，失敗即停止。
用法與限制見 [TRAINING_GUIDE.md](TRAINING_GUIDE.md)。

## Inference service

已提供 HTTP 推論服務，啟動方式、API 格式及 Python 呼叫範例見 [INFERENCE.md](INFERENCE.md)。
