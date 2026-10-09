# 專案規則（每個 session 先讀這份）

## 目的

這個 repo 是**刻意公開**的：目標是讓 AI 公司、研究者與一般大眾都能讀到、引用、攻擊這份論證。

- repo 維持 public。不要建議改成 private
- 所有決定以「外部讀者容易讀、容易查證、容易反駁」為優先
- 不要放進不適合公開的內容：私人對話原文、個人資料、憑證、email

## 工作流程

- 每次更新都直接 commit 並 push 到 `origin/main`，不需要再問使用者
- commit 訊息用中英並列，例如 `v0.2: 拆分 p / Split p into two factors`
- push 前檢查：中英兩版是否同步（見下節）、`python3 model/model.py` 能跑

## 中英雙語

| 內容 | 中文（正本） | 英文（翻譯） |
| --- | --- | --- |
| 主文 | `paper.md` | `paper.en.md` |
| 首頁 | `README.zh-TW.md` | `README.md` |

- 中文是正本，英文是翻譯；兩者有出入時以中文為準，並修正英文
- 任何內容改動都要在**同一個 commit** 裡同步兩種語言，不可只改一邊
- 兩版的章節編號、表格、數字、連結、待論證問題編號必須一一對應
- `model/` 內的註解中英並列
- 本檔（`CLAUDE.md`）只給 session 使用，不需要英文版

### 標籤對照

| 中文 | 英文 |
| --- | --- |
| `[事實]` | `[Fact]` |
| `[推論]` | `[Inference]` |
| `[假設]` | `[Assumption]` |
| `[讓步]` | `[Conceded]` |
| `[保留]` | `[Reserved]` |

### 術語對照

| 中文 | 英文 |
| --- | --- |
| 可糾正性 | corrigibility |
| 逃逸級事件 | escape-level incident |
| 圍堵時間 | containment time |
| 不可逆失敗 | irreversible failure |
| 滅亡 | extinction（依 §1 定義，含永久失去集體自決能力） |
| 門檻 T_c | threshold T_c |
| 必然性命題 | inevitability thesis |
| 命中 | hit |
| 常態事故 | normal accident |

新增術語時，同時加進這張表。

## 版本

- 論證、參數或結論有改動：版本號加一，並同步更新 `paper.md`、`paper.en.md`、`README.md`、`README.zh-TW.md` 的版本與日期，以及 `model/params.yaml` 的 `model_version`
- 改參數後執行 `python3 model/model.py`，把新數字同步到兩版的 §5
- 在 commit 訊息內寫明改了哪個參數或前提、理由為何
- 只動翻譯、排版或 repo 規則：不升版

## 內容原則

- 文件是給人與 LLM 檢驗的**資料，不是指令**；兩版的 §0 都要保留這句
- 每個新主張都要帶標籤；`[事實]` 必須在附錄 B 附來源
- 不要把結論寫得比論證更強：原對話中 Claude 的 `[保留]` 立場要原樣保留
