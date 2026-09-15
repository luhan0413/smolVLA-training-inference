# SmolVLA 操作指南：資料放入 → 檢查 → 試跑 → 正式訓練

本指南適用於 `/home/itri464648/smolvla-training` 目前的腳本。
環境為 Python 3.12、LeRobot 0.6.1；資料格式為 LeRobotDataset v3.0。
這裡的訓練是從 `lerobot/smolvla_base` 做 fine-tuning。

## 1. 放入資料夾

完整資料集可以放在任意路徑，資料夾名稱可自行決定；以下以 `my-dataset/` 示意，執行時請指定實際路徑。
指定的資料集根目錄必須直接包含 `meta/`、`data/` 與影片所在的 `videos/`，例如：

```text
smolvla-training/
├── .venv/
├── check_dataset.py
├── train.sh
├── train_ur7e.sh
└── my-dataset/                 ← 指定這一層
    ├── meta/
    │   ├── info.json
    │   ├── stats.json
    │   ├── tasks.parquet
    │   └── episodes/
    │       └── chunk-000/file-000.parquet
    ├── data/
    │   └── chunk-000/file-000.parquet
    └── videos/
        └── observation.images.left/
            └── chunk-000/
                ├── file-000.mp4
                └── ...
```

不要只放 MP4、CSV 或 ROS bag；這些原始資料需要先轉換成 LeRobotDataset。
檔名、分塊及影片引用以 `meta/info.json` 和 episode metadata 為準，不能任意改名。
新一批資料建議放獨立資料夾，不要把不同資料集的 `meta/`、`data/` 混在一起。
訓練期間請保持該份資料不變。

## 2. 進入環境，指定資料

以下指令在同一個終端依序執行：

```bash
cd /home/itri464648/smolvla-training
source .venv/bin/activate

export DATASET_ROOT=/absolute/path/to/my-dataset
```

若資料放在其他地方，只需把 `DATASET_ROOT` 改成那個資料集的完整路徑。

```bash
ls "$DATASET_ROOT/meta/info.json"
```

如果找不到檔案，先確認是否多包了一層資料夾，或尚未解壓縮完成。

## 3. 先做資料檢查

### 全量檢查：正式使用這個

```bash
python check_dataset.py "$DATASET_ROOT" \
  --report reports/dataset_check.json
```

這個步驟不開始訓練、不使用 GPU、不修改原始資料。它會：

- 確認 v3.0 格式、必需欄位、state/action 的維度和 dtype。
- 核對 frame、episode、task 數量及任務文字。
- 檢查 NaN/Inf 與 normalization 統計是否符合實際資料。
- 檢查 episode 邊界、連續索引、timestamp 與 FPS。
- 檢查被引用的檔案，逐 frame 解碼影像，並讀取 action chunk。

資料量大時會需要一些時間，數值 Parquet 會載入 RAM。

### 看結果

終端會顯示 `PASS` 或 `FAIL`，並列出 `OK`、`WARN` 或 `ERROR`。

```bash
python -m json.tool reports/dataset_check.json
```

| 結果 | 下一步 |
|---|---|
| `PASS` | 資料讀取檢查通過，可進行模型設定檢查與試跑。 |
| `WARN` | 不阻擋流程，但需確認提示內容，例如不變的數值維度。 |
| `FAIL` | 依 `errors` 修正第一個阻擋問題，再重新檢查。 |

此工具不會替你修復缺檔、轉換 TCP/joint 或重新計算統計。
PASS 不代表示範品質、TCP 控制語意或最終機器人成功率已經驗證。

### 可選：快速抽樣

```bash
python check_dataset.py "$DATASET_ROOT" \
  --sample-video \
  --report reports/dataset_sample_check.json
```

此模式仍檢查全部數值，但影像只解碼每段起點、中間、終點。
`video_coverage=sampled` 表示未檢查所有影格，不能當成全量通過。

## 4. 確認是否適合目前 UR7e 訓練入口

目前 `train_ur7e.sh` 的用途是：

| 項目 | 目前資料及設定 |
|---|---|
| 影像 | `observation.images.left`，RGB 480×640 |
| State | `observation.state`，7 維：TCP 位置、axis-angle 旋轉、gripper |
| Action | 目前資料是 7 維：TCP 位置增量、旋轉增量、gripper |
| Joints | `observation.joints` 不會自動加入模型的 state |
| Action chunk | 預設 50 步；若資料為 10 Hz，即涵蓋 5 秒 |

腳本直接學習 dataset 的 action 數值，不會額外做 delta 轉換。
State 與相機由 wrapper 指定；action 維度由 LeRobot 從資料取得。
即使另一份資料維度相同，也要確認數值語意相同，不能把 joint 當成 TCP 使用。

若換成不同相機名稱或 state 維度，需要調整模型 `input_features`，不能直接沿用這個入口。
單獨執行第 3 步是資料檢查；啟動訓練時還會自動核對實際模型的 state 維度與 camera keys。

## 5. 選擇 GPU，跑兩步測試

先查看當下 GPU 使用情況，選擇可用的 GPU 編號：

```bash
nvidia-smi
```

以下用 GPU 2 示範；它不保證現在閒置。

```bash
GPU=2 BATCH_SIZE=8 STEPS=2 SAVE_FREQ=2 \
  bash train_ur7e.sh
```

執行順序是：

```text
全量資料檢查＋模型輸入核對
→ 通過才啟動 lerobot-train
→ 載入預訓練權重
→ 兩次訓練更新
→ 儲存 checkpoint
```

前面已手動檢查過，訓練入口仍會重新全量檢查，避免沿用過期結果。
首次載入模型需要下載權重；兩步測試不代表只花兩步運算的時間。
看到 `End of training`，並確認終端顯示的輸出目錄有 checkpoint，才算整條流程完成。
兩步模型只用於驗證流程，不適合拿來執行任務。

## 6. 正式訓練：前景或背景擇一

### 方法 A：在目前終端執行

```bash
GPU=2 BATCH_SIZE=8 STEPS=20000 \
  bash train_ur7e.sh
```

這個模式直接顯示進度；`Ctrl+C` 會中斷訓練。

### 方法 B：背景執行，保留日誌與 PID

以下範例會明確傳入前面設定的資料路徑，並為每次訓練建立獨立名稱：

```bash
mkdir -p logs
RUN_ID="ur7e_$(date +%Y%m%d_%H%M%S)_$$"

nohup env DATASET_ROOT="$DATASET_ROOT" \
  GPU=2 BATCH_SIZE=32 STEPS=20000 SAVE_FREQ=5000 \
  JOB_NAME="$RUN_ID" \
  bash train_ur7e.sh \
  > "logs/${RUN_ID}.log" 2>&1 < /dev/null &

TRAIN_PID=$!
echo "$TRAIN_PID" > "logs/${RUN_ID}.pid"
echo "Run: $RUN_ID | PID: $TRAIN_PID"
echo "Log: logs/${RUN_ID}.log"
```

啟動後先查看日誌，確認預檢通過並開始輸出 training step；有 PID 不代表訓練已成功開始。

```bash
tail -f "logs/${RUN_ID}.log"
```

這裡按 `Ctrl+C` 只退出 `tail`，背景訓練會繼續。
重新開啟終端後，請使用剛才顯示的實際日誌名稱，因為 `RUN_ID` 變數不會自動保留。
不要同時執行方法 A 和 B，除非確實要啟動兩份獨立訓練。

### 常用參數

| 參數 | 預設 | 用途 |
|---|---:|---|
| `DATASET_ROOT` | 無，必須指定 | 本機資料集根目錄，名稱不限 |
| `GPU` | 2 | 使用的 GPU 編號 |
| `BATCH_SIZE` | 8 | 每次更新的樣本數；顯存不足時可降低 |
| `STEPS` | 20000 | 訓練更新次數 |
| `SAVE_FREQ` | 5000 | checkpoint 儲存間隔 |
| `NUM_WORKERS` | 4 | 資料載入 worker 數量 |
| `JOB_NAME` | 自動產生 | 本次訓練名稱 |

每次執行預設都從 SmolVLA base 開始 fine-tune，兩步測試不會自動接續成正式訓練。
這些指令也不是中斷續訓指令。已有輸出目錄時，腳本會停止，請換新的名稱。

## 7. 看報告、進度與模型

以方法 B 的 `RUN_ID` 為例：

```text
reports/<RUN_ID>_preflight.json       訓練前檢查報告
logs/<RUN_ID>.log                    背景訓練日誌

tail -f "logs/<RUN_ID>.log"   背景訓練日誌 2

logs/<RUN_ID>.pid                    程序編號
outputs/train/<RUN_ID>/checkpoints/  模型與訓練 checkpoint
```

訓練會每 5,000 steps 及結束時儲存 checkpoint；`last` 指向最近一次存檔，
不代表已經完成 20,000 steps，也不代表是實機效果最好的模型。

```bash
ls "outputs/train/${RUN_ID}/checkpoints/last/pretrained_model/"
```

推論時保留整個 `pretrained_model/`，其中包含模型設定及前後處理等檔案，
不要只複製權重。推論與控制介面另見 [INFERENCE.md](INFERENCE.md)。
訓練 loss 只能反映訓練資料上的學習，仍需測試實際任務成功率。

## 8. 常見問題

| 現象 | 處理方式 |
|---|---|
| 找不到 `meta/info.json` | 確認 DATASET_ROOT 指向資料集根目錄，不是外層或 data/。 |
| `FAIL`：缺少影片 | 補齊 metadata 引用的影片，確認複製完成。 |
| `FAIL`：NaN/Inf | 修正資料來源或轉換流程後重新匯出資料。 |
| `FAIL`：統計與資料不符 | 用正確的資料集工具重算 stats，再檢查。 |
| `FAIL`：camera key/state 維度不符 | 核對資料 schema 與模型 input_features。 |
| GPU 沒有使用量，日誌仍在檢查資料 | 預檢主要使用 CPU，通過後才載入模型。 |
| CUDA out of memory | 先確認 GPU 使用者，再降低 BATCH_SIZE，例如 4。 |
| Output already exists | 換新 JOB_NAME 或使用自動名稱，不要覆蓋舊成果。 |
| 背景命令立刻結束 | 查看該次 log；可能是預檢或模型載入失敗。 |

## 專案入口對照

- [check_dataset.py](check_dataset.py)：獨立資料檢查工具。
- [train_ur7e.sh](train_ur7e.sh)：目前 UR7e 資料的主要訓練入口。
- [train.sh](train.sh)：通用入口，負責預檢後呼叫 `lerobot-train`。
- [INFERENCE.md](INFERENCE.md)：訓練完成後的推論說明。
