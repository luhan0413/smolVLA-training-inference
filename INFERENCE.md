# SmolVLA 推論 Server 使用指南

本文件供 server 管理者與遠端 client 使用者操作。Server 接收相機影像、TCP state 和任務文字，回傳預測動作；不會直接控制機器人。

## 1. 目前配置

| 項目 | 設定 |
| --- | --- |
| 主機專案目錄 | `/home/itri464648/smolvla-training` |
| Python 環境 | `.venv/bin/python`，現有環境為 Python 3.12、LeRobot 0.6.1 |
| Server 程式 | `inference_service.py` |
| Client 範例 | `inference_client.py` |
| 本文件使用的 GPU | 實體 GPU 2，以 `CUDA_VISIBLE_DEVICES=2` 指定 |
| 區網 IP | `192.168.50.159`，若主機網路變更需重新確認 |
| Port | `8085` |
| 預設 checkpoint | `outputs/train/ur7e_20260910_085801_3951068/checkpoints/020000/pretrained_model` |
| 健康檢查 | `GET /health` |
| 推論 | `POST /infer` |

以下 server 指令都在這台主機執行。遠端使用者只需要知道 server URL 與 API，不需要下載模型或有 GPU。Server 沒有網頁操作介面；瀏覽器可以查看 `/health`，推論需發送 POST。

## 2. 啟動前檢查

```bash
cd /home/itri464648/smolvla-training
nvidia-smi
hostname -I
ss -ltnp 'sport = :8085'
```

確認 GPU 2 有足夠資源、主機仍有 `192.168.50.159`，且 8085 沒有其他服務監聽。若已有本服務，直接使用或先按第 4 節關閉，避免重複載入模型。

現有 `.venv`、checkpoint 和模型／tokenizer cache 已供本機使用。下方啟動方式啟用離線模式，不會自動下載缺少的資源。若搬到新主機，需先準備相容環境、完整 checkpoint 與所需 cache，不能只複製 `inference_service.py`。

## 3. 啟動 Server

### 3.1 前景執行：適合手動測試

在主機終端執行：

```bash
cd /home/itri464648/smolvla-training
CUDA_VISIBLE_DEVICES=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  .venv/bin/python -u inference_service.py \
  --host 192.168.50.159 --port 8085
```

等到出現以下訊息才算完成，載入模型期間連線可能被拒絕：

```text
INFO:root:Ready at http://192.168.50.159:8085
```

終端會持續顯示請求與錯誤紀錄。保持該終端開啟；要停止時按 **Ctrl+C**。

### 3.2 背景執行：適合登出 SSH 後繼續提供服務

在一般主機 Bash／SSH 終端執行以下指令；先確保沒有另一個 server 使用同一 port：

```bash
cd /home/itri464648/smolvla-training
mkdir -p logs
nohup env CUDA_VISIBLE_DEVICES=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  .venv/bin/python -u inference_service.py \
  --host 192.168.50.159 --port 8085 \
  >> logs/inference_service.log 2>&1 < /dev/null &
printf '%s\n' "$!" > logs/inference_service.pid
```

查看啟動進度：

```bash
tail -n 50 -f logs/inference_service.log
```

此處按 Ctrl+C 只會退出 `tail`，不會關閉背景 server。日誌採追加寫入，請看最新時間與最新啟動結果。PID 檔案存在不代表 server 已啟動成功，仍須檢查 `/health`。

`nohup` 不提供開機自動啟動或程序崩潰後自動重啟；主機重開後要再啟動。在會清理子程序的受管理終端中，也不能只憑背景指令成功就判定 server 持續存活。

### 3.3 確認服務可用

在主機或同區網電腦執行：

```bash
curl --noproxy '*' --fail --max-time 5 http://192.168.50.159:8085/health
```

成功範例：

```json
{
  "status": "ready",
  "checkpoint": "/home/itri464648/smolvla-training/outputs/train/ur7e_20260910_085801_3951068/checkpoints/020000/pretrained_model",
  "device": "cuda"
}
```

`ready` 表示模型已載入且 HTTP 可用，不代表抓取任務成功。請確認回應的 checkpoint 是預期版本。

## 4. 關閉與重新啟動

### 前景服務

回到啟動 server 的終端，按 **Ctrl+C**。

### 背景服務或找不到原本終端

先查看 PID 檔案對應程序：

```bash
cd /home/itri464648/smolvla-training
cat logs/inference_service.pid
ps -p "$(cat logs/inference_service.pid)" -o pid,args
ss -ltnp 'sport = :8085'
```

確認該 PID 的命令是要停止的 `inference_service.py`，host／port 也正確，再執行：

```bash
kill -TERM "$(cat logs/inference_service.pid)"
```

若 PID 檔案不存在、程序已結束，或指向其他程序，不要直接 kill。改查實際程序：

```bash
pgrep -af '[p]ython.*inference_service.py'
ss -ltnp 'sport = :8085'
```

確認後用 `kill -TERM <實際PID>` 停止；請把 `<實際PID>` 換成數字。避免使用 `pkill python`，以免停止訓練或其他服務。

關閉後檢查：

```bash
ss -ltnp 'sport = :8085'
```

沒有 LISTEN 列表示此 port 已停止監聽。確認關閉後可刪除舊 PID 檔案：

```bash
rm -f logs/inference_service.pid
```

**重新啟動**：先完成關閉，再執行第 3 節任一啟動方式，最後檢查 `/health`。修改參數或程式後也需要重新啟動才會生效。

## 5. 如何調整啟動參數

| 參數 | 程式預設值 | 用途 |
| --- | --- | --- |
| `--checkpoint PATH` | 第 1 節的 020000 checkpoint | 指向完整的 `pretrained_model` 目錄，不是單一權重檔案 |
| `--device DEVICE` | `cuda` | 使用可見 GPU；可改成 `cpu`，但推論速度需另測 |
| `--host ADDRESS` | `127.0.0.1` | 本機限定；本文件以 `192.168.50.159` 開放區網介面 |
| `--port PORT` | `8085` | HTTP port；改動後 client URL 也要修改 |

| 環境變數 | 本文件使用值 | 用途 |
| --- | --- | --- |
| `CUDA_VISIBLE_DEVICES` | `2` | 選擇實體 GPU；程式本身沒有預設選 GPU 2 |
| `HF_HUB_OFFLINE` | `1` | Hugging Face Hub 使用本機資源 |
| `TRANSFORMERS_OFFLINE` | `1` | Transformers 使用離線模式 |

`CUDA_VISIBLE_DEVICES=2` 時，程序內第一張可見 GPU 是實體 GPU 2；使用 `--device cuda` 即可，不需要改成 `cuda:2`。

例如：改用實體 GPU 0、15000-step checkpoint 與 port 8086。先依第 2 節檢查該 GPU 和 port，再啟動：

```bash
cd /home/itri464648/smolvla-training
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  .venv/bin/python -u inference_service.py \
  --checkpoint outputs/train/ur7e_drawer_20k/checkpoints/015000/pretrained_model \
  --device cuda --host 192.168.50.159 --port 8086
```

此時 client URL 為 `http://192.168.50.159:8086`。

若只給本機使用，將 host 改成 `127.0.0.1`。遠端 client 的 `127.0.0.1` 代表遠端電腦自己，並不是 server。若 server 綁定本機位址，可透過 SSH tunnel 使用：

```bash
ssh -N -L 8085:127.0.0.1:8085 itri464648@192.168.50.159
```

保持 tunnel 開啟，在遠端電腦以 `http://127.0.0.1:8085` 呼叫。這個 tunnel 範例適用於 server 綁定 `127.0.0.1` 的情況。

本服務沒有驗證或 TLS。區網使用請綁定預期的區網 IP；`0.0.0.0` 會監聽所有 IPv4 介面，不是「只允許同子網」。綁定區網 IP 本身也不會過濾來源，實際存取範圍由路由與防火牆決定。

### 哪些不是啟動參數？

目前沒有 `--fps`、`--chunk-size`、`--temperature` 或 gripper 閾值參數。服務要求輸出為 **50×7**，回應的 `fps` 固定為 **10**，輸入影像固定為 **640×480**。更換 checkpoint 必須與這些條件及特徵／processor 設定相容，不能任意替換其他模型。

模型、前處理、反正規化使用同一 checkpoint 的設定與統計量。不要混用其他資料集的 normalization 檔案。若要調整 action 維度、影像規格、chunk 長度或 gripper 後處理，需要修改程式並驗證，不是單純更改 CLI。

## 6. 遠端使用者快速呼叫

先依第 3.3 節確認 `/health`。將本專案的 `inference_client.py` 複製到 client 電腦；它只使用 Python 標準函式庫，不需要安裝 LeRobot 或 CUDA。

```bash
python3 inference_client.py \
  --url http://192.168.50.159:8085 \
  --image /path/to/left.png \
  --state -0.06 -0.29 0.49 1.26 1.19 -1.32 0.0 \
  --task 'grasp the black handle on the target drawer and pull the drawer open.'
```

請換成真實圖片路徑及與該影像同步的 TCP state；上方數值只是格式示例。圖片是 client 電腦上的檔案，client 會將內容編碼傳送，不是要求 server 讀取這個路徑。

| Client 參數 | 說明 |
| --- | --- |
| `--url` | Server 根網址，不要加 `/infer`；預設為 `http://127.0.0.1:8085` |
| `--image` | 必填，640×480 PNG 或 JPEG |
| `--state` | 必填，依序提供 7 個數值：x、y、z、rx、ry、rz、gripper |
| `--task` | 必填，非空任務文字 |

範例 client 每次執行只送一個請求，輸出 JSON，不會控制機器人；請求 timeout 為 120 秒，且程式會略過環境中的 HTTP proxy。

## 7. HTTP API 與動作解讀

### POST /infer

發送 JSON，使用 `Content-Type: application/json` 與正確的 `Content-Length`。不支援 chunked transfer encoding。以下 `image_base64` 是佔位文字，實際需換成圖片原始檔案的 Base64：

```json
{
  "state": [-0.06, -0.29, 0.49, 1.26, 1.19, -1.32, 0.0],
  "task": "grasp the black handle on the target drawer and pull the drawer open.",
  "image_base64": "<PNG 或 JPEG 原始檔案的 Base64>"
}
```

- `state`：恰好 7 個有限數值，順序見上表；傳入訓練資料的原始尺度，不要自己先做 mean/std normalization。
- `task`：非空字串，最長 2000 字元；實際 tokenizer 長度由 checkpoint 設定決定。
- `image_base64`：不含 `data:image/...;base64,` 前綴。圖片必須是寬 640、高 480；server 轉 RGB、CHW、[0,1] 後交給模型處理。
- 整個 request body 上限 8 MiB，包含 Base64 與 JSON。

### 成功回應

| 欄位 | 意義 |
| --- | --- |
| `actions` | 50 列、每列 7 個數值，已使用 checkpoint 統計量反正規化 |
| `shape` | `[50, 7]` |
| `fps` | `10`，表示資料頻率，不是 server 吞吐量保證 |
| `action_names` | `delta_x, delta_y, delta_z, delta_rx, delta_ry, delta_rz, gripper` |
| `inference_ms` | server 端本次預測流程耗時，不含整段網路傳輸與機器人移動時間 |

50 步對應資料時間尺度的 5 秒預測；如何選取、排程、執行這些 action 由 client 負責。Server 串行處理請求，多人可以呼叫但不會同時執行多個推論；繁忙時會增加等待時間。每次推論會 reset policy，沒有跨請求的 action queue。健康檢查也可能等待正在處理的推論。

### 座標系與機器人控制

資料收集端的定義如下（由資料提供者說明）：

| 原始欄位 | 平移 delta | 旋轉 delta |
| --- | --- | --- |
| `action_tcp`（tool frame） | `R_current.T @ (p_next - p_current)` | `R_current.T @ R_next` |
| `action_tcp_base`（base frame） | `p_next - p_current` | `R_next @ R_current.T` |

目前訓練資料只將動作命名為 `action`，metadata 未標明由上述哪個欄位匯入。**部署前需確認資料轉換時選用的來源**，不能只依 `delta_x` 等欄位名稱推定 frame。

若確定使用 tool-frame delta，執行端應按以下方式合成：

```python
p_target = p_current + R_current @ delta_p_tool
R_target = R_current @ delta_R_tool
```

這裡 `delta_R_tool` 是旋轉矩陣；若 action 的三個旋轉值儲存為旋轉向量，需先依相同定義轉換，不能直接加到目前姿態的三個分量。State 的座標系、位置／角度單位、TCP offset、gripper 開閉方向，以及 delta 的時間基準都需和資料收集端一致。Server 不做座標變換、FK／IK 或機器人控制，也不會因換成 UR5 自動完成適配。

## 8. 常見問題

| 現象 | 檢查與處理 |
| --- | --- |
| `Connection refused` | Server 未啟動、仍在載入、已退出，或 URL／port 錯誤；查看日誌與 `ss` |
| 本機能連，其他電腦無法連 | 確認綁定區網 IP、client 使用正確 IP，並檢查路由／防火牆是否允許 TCP 8085 |
| `Address already in use` | 同一 port 已被佔用；確認現有程序，停止自己的舊服務或換 port |
| `Cannot assign requested address` | `--host` 不是這台主機目前的 IP；以 `hostname -I` 確認 |
| 缺少模型／tokenizer／cache | 離線模式不會下載；補齊相容 checkpoint 與 cache 後再啟動 |
| CUDA／顯存錯誤 | 用 `nvidia-smi` 檢查 GPU；選可用 GPU，並檢查環境相容性 |
| HTTP 400 | State、task、圖片格式不符，或使用不支援的 transfer encoding；查看回應 `error` |
| HTTP 413 | Body 為空、缺少有效 Content-Length，或超過 8 MiB |
| HTTP 404 | 檢查路徑是否為 `/health` 或 `/infer`，不要加尾端 `/` |
| HTTP 500 | 模型推論失敗，查看 server traceback |
| HTTP 200 但 client 拋出 gripper 範圍錯誤 | 見下方已知限制；HTTP 成功不代表 client 動作驗證必定通過 |

### 已知限制：gripper 可能超出 0～1

訓練資料 gripper 的 min/max 為 0／1，但模型的連續輸出仍可能超界，例如曾回傳 `-0.0305`。**目前 server 只檢查 action shape 和有限值，沒有 clamp gripper。** 若 client 要求 `0 <= gripper <= 1`，就可能拋出 `gripper_position must be finite and between 0 and 1` 而停止。

重新啟動不會修正這個問題。若部署約定 gripper 為 0～1，可在確認開閉語意後加入明確的範圍處理，並保留異常記錄與 client 驗證；這是待實作的變更，不是目前服務已提供的功能。不要將前六個位移／旋轉欄位一起限制為 0～1。

## 9. 維護與驗證

不啟動 server 的單元測試：

```bash
cd /home/itri464648/smolvla-training
.venv/bin/python -m unittest test_inference_service -v
```

測試程式通過、`/health` ready、`/infer` 回傳有效陣列，分別驗證不同層次；都不能單獨證明實機能抓住把手或完成任務。

過去的推論 smoke test 結果保存在 `logs/inference_smoke_result.json`，僅代表當時的流程驗證。背景啟動日誌位置為 `logs/inference_service.log`；前景啟動則直接看終端。
