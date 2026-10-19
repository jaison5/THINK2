# 有交易成本、動作遮罩與 walk-forward 的量子強化學習匯率交易評估

**暫定題目**：Cost-Aware, Risk-Adjusted Quantum Reinforcement Learning for FX Trading: A Masked, Walk-Forward Re-evaluation of QA3C

資料夾放用到的所有程式、資料、結果和紀錄，由 `B:\qfin-main\sync_papers.py` 從原本的實驗資料夾複製過來。資料夾結構和原本相同（`qfin-main`、`B`、`C` 並排），程式不用改路徑就能直接執行。

## 研究目的

1. **檢驗原主張**：原論文（arXiv 2509.09176）說 QLSTM＋QA3C 的報酬比古典 A3C 高 0.45%。在規格完整、有交易成本、有動作遮罩、walk-forward 的條件下，這個結論是否成立？
2. **找出結果的來源**：原論文的獎勵設計會被代理人鑽漏洞（例如用「無效的賣出」來空手、頻繁小賺就賣），結果主要由獎勵與環境設定決定，而不是模型。
3. **量化預測器的價值**：狀態裡的 QLSTM 預測，到底對交易有沒有幫助（沒有預測器／QLSTM／理想預測器的消融）。
4. **確保公平與嚴謹**：加入參數量相近的古典 A3C（238 個參數，QA3C 為 244 個）與論文的古典 A3C（3,332 個參數）；用 PennyLane 後端做交叉驗證。

## 資料夾對照

| 位置 | 內容 | 對應論文章節 |
|---|---|---|
| `qfin-main/` | 原論文實驗二的重現：`exp2.py`、`pl_exp2.py`（QLSTM 預測器、A3C、QA3C）、Mac 與 Windows 的結果、貨幣 ETF 資料、重現報告 `REPRODUCTION_WIN.md` | 重現原論文 |
| `C/cdata.py` | 資料清理、技術指標、walk-forward 切分（測試年 2020～2024）、每個 fold 重新訓練的 QLSTM 預測器 | 資料與預測器 |
| `C/env_c.py` | 交易環境：動作遮罩、交易成本（單邊比例）、兩種獎勵（論文的 Table III／扣成本後的對數報酬）；評估與規則型基準 | 環境設計 |
| `C/train_c.py`、`C/run_c.py` | 非同步 A3C（8 個 worker、論文的超參數），在驗證年挑模型、在測試年用 0／0.05%／0.1% 三種成本評估；代理人 QA3C、A3C_small、A3C、QA3C_PL（PennyLane 後端） | 實驗 |
| `C/check_np_policy.py` | 嚴謹性檢查：NumPy 選動作和 PennyLane 版 QA3C 是否完全一致 | 方法驗證 |
| `C/report_c.py`、`C/results/`、`C/logs/` | 彙整與所有結果、紀錄（`results/rl_old_sampler/` 是改程式前的 9 次 QA3C，留作對照） | 結果 |
| `B/fastsim.py`、`B/check_fastsim.py` | C 使用的批次模擬器，以及和 PennyLane 的等價性檢查；細節見 `B/README.md` | 方法 |
| `C/REFERENCES.md`、`B/REFERENCES.md` | 參考文獻與重疊程度判斷 | 相關研究 |

## 實驗設計

| 組別 | 內容 |
|---|---|
| E1 主實驗 | 獎勵 logret、QLSTM 預測；QA3C vs. 古典 A3C h=9 vs. 古典 A3C h=128；5 個 fold × 3 個種子 |
| E2 獎勵消融 | 論文的獎勵（有遮罩與成本）；QA3C、古典 A3C h=9 |
| E3 預測器消融 | 古典 A3C h=9 搭配「沒有預測器（0.5）」與「理想預測器（用未來資訊，當上限）」 |
| 規則型基準 | 買進持有、均線 MA5／MA20、隨機（20 次平均） |
| 交叉驗證 | PennyLane 後端完整訓練 3 次 QA3C（2024 年、種子 0～2），和新程式比較分布 |

所有代理人在 0.05% 成本下訓練，測試時用 0、0.05%、0.1% 三種成本評估。

## 目前狀態（2026-10-19 20:00，全部實驗已完成）

| 項目 | 狀態 |
|---|---|
| 原論文實驗二的重現（Mac、Windows） | 完成 |
| 每個 fold 的 QLSTM 預測器、規則型基準 | 完成 |
| C 主實驗（E1～E3，105 次訓練，含用新程式重跑的 12 次 QA3C） | 完成 |
| PennyLane 交叉驗證（3 次） | 完成 |
| C 的報告 | 完成：`C/README.md`、`C/results/summary_c.md` |

## 目前的主要發現

- **原設定下的重現沒有成功**：Windows 重跑的 QA3C 3 個種子報酬 2.54%、0%、0%（後兩個在測試期間一次都沒交易），平均 0.85%，低於古典 A3C 的 9.25%，和論文「QA3C 比 A3C 高 0.45%」相反。
- **沒有任何強化學習代理人贏過簡單策略**（E1，成本 0.05%，2020～2024 年 5 年複利）：古典 A3C h=128 −1.49%、古典 A3C h=9 −1.80%、QA3C −2.81%；買進持有 +8.94%、均線 +7.88%。
- **QA3C 沒有勝過古典 A3C**：15 對同 fold、同種子的比較，三種成本下差異都不顯著（Wilcoxon p = 0.064～0.978）。
- **結果主要由獎勵設計決定**（E2）：原設定下代理人會用無效的「賣出」來空手；加上動作遮罩後，用論文的獎勵交易次數大增（古典 A3C h=9 每年 74 次、QA3C 24.7 次），用對數報酬獎勵則 QA3C 幾乎不交易（每年 1.9 次，15 次訓練中 8 次完全不交易）。
- **QLSTM 預測器幾乎沒有幫助**（E3）：古典 A3C h=9 的 5 年報酬，沒有預測器 −3.77%、QLSTM −1.80%、理想預測器 +2.19%；預測器準確率 55.2%（87 筆，±10.5 個百分點）。
- **PennyLane 交叉驗證**：PennyLane 後端完整訓練 3 次 QA3C，結果和新程式完全相同，訓練時間 140.8 vs. 14.4 分鐘。
- 詳細數字見 `C/README.md`。
- **論文的 ETF 基準期間對不上**：Table V 的 ETF 報酬和它寫的測試期間不符，較接近到 2025-07-21 為止的 5 年。
- **計算後端不影響結果**：NumPy 選動作和 PennyLane 版 QA3C 在 9,969 步中選出的動作完全相同（logits 最大差 1.6e-15）。

## 重跑方式

環境（Windows，在本資料夾用 Git Bash）：

```bash
py -3.11 -m venv qfin-main/.venv-pl
qfin-main/.venv-pl/Scripts/python.exe -m pip install -r qfin-main/requirements-pl.txt yfinance
cd C
../qfin-main/.venv-pl/Scripts/python.exe run_c.py --parallel 3          # 預測器、基準、105 次訓練
../qfin-main/.venv-pl/Scripts/python.exe check_np_policy.py             # NumPy 與 PennyLane 的一致性
../qfin-main/.venv-pl/Scripts/python.exe train_c.py --fold 2024 --agent QA3C_PL --reward logret --probs qlstm --seed 0   # PennyLane 交叉驗證
../qfin-main/.venv-pl/Scripts/python.exe report_c.py                    # 彙整
```

原論文的重現步驟見 `qfin-main/README.md`、`qfin-main/REPRODUCTION_WIN.md`。

