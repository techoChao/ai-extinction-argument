# Project rules (read this first in every session)

## Purpose

This repo is **public on purpose**: the goal is for AI companies, researchers and the general public to read, cite and attack this argument.

- Keep the repo public. Do not suggest making it private
- Prioritize whatever makes the work easy for outside readers to read, verify and rebut
- Do not add anything unfit for publication: raw private conversations, personal data, credentials, email addresses

## Workflow

- Commit and push every update directly to `origin/main`; do not ask the user first
- Commit messages are in English, e.g. `v0.2: Split p into two factors`
- Before pushing, check that `python3 model/model.py` and `python3 model/mc.py` both run (`mc.py` exits non-zero if its S0 regression against `model.py` fails, and prints a `WARNING` on its "params.yaml mirror check" line if `model/params.yaml` no longer mirrors its parameters) and, if the Chinese files were touched, that they match the English (see below)

## Language: English source, Chinese output

All source files are written in **English only**: `paper.md`, `README.md`, `model/`, `review/`, this file, and any new file. Chinese is produced only as a translation when a version is finalized, for Chinese-language readers (including the repo owner).

| Content | Source (English) | Chinese translation (finalized versions only) |
| --- | --- | --- |
| Paper | `paper.md` | `paper.zh-TW.md` |
| Landing page | `README.md` | `README.zh-TW.md` |
| Review records | `review/` | English only, no translation |

- Edit only the English files during day-to-day work. Do not edit the Chinese files to make content changes
- When a version is finalized (see Versioning), regenerate `paper.zh-TW.md` and `README.zh-TW.md` from the English in the same commit as the version bump
- Between finalized versions the Chinese files may lag behind; their header names the version they translate
- The English text governs. If the two differ, fix the Chinese
- The translation must keep section numbers, tables, numbers, links and open-question numbers one-to-one with the English
- The user may write to you in Chinese; still write everything that goes into the repo in English
- Use Traditional Chinese (zh-TW) for translations

### Tag glossary

| English | Chinese |
| --- | --- |
| `[Fact]` | `[事實]` |
| `[Inference]` | `[推論]` |
| `[Assumption]` | `[假設]` |
| `[Conceded]` | `[讓步]` |
| `[Reserved]` | `[保留]` |

### Term glossary

| English | Chinese |
| --- | --- |
| corrigibility | 可糾正性 |
| escape-level incident | 逃逸級事件 |
| containment time (v0.2 definition: detection → no further unauthorized actions under active response) | 圍堵時間 |
| irreversible failure | 不可逆失敗 |
| extinction (as defined in §1, includes permanent loss of collective self-determination) | 滅亡 |
| threshold T_c | 門檻 T_c |
| inevitability thesis | 必然性命題 |
| hit | 命中 |
| normal accident | 常態事故 |
| E1 (permanent loss of humanity's collective self-determination, including extinction) | E1（人類集體自決權的永久喪失，含滅絕） |
| E2 (strict extinction: human population below 5,000) | E2（嚴格滅絕：人口少於 5,000） |
| dwell time (first unauthorized action → detection) | 駐留時間 |
| remediation time (detection → artifacts removed and the capability made unavailable) | 修復時間 |
| re-entry (renewed unauthorized activity after a containment attempt) | 再入侵 |
| root cause | 根本原因 |
| facet (one victim, exit channel or failed attempt of a root cause) | 面向 |
| prior-predictive | 先驗預測 |
| safe absorbing state | 安全吸收態 |
| human-nature parameter | 人性參數 |
| accident channel (accidental escape) | 事故管道（意外逃逸） |
| side channel (deliberate misuse, or concentration or lock-in: a route to E1 without an escape) | 旁路管道 |
| deliberate misuse | 蓄意濫用 |
| concentration or lock-in | 權力集中或鎖定 |
| per-incident lethality e1 | 單次事件致死率 e1 |
| review record | 審查紀錄 |
| AI-nature parameter | AI 本性參數 |
| outside view (pooled published anchors) | 外部觀點 |
| tier-1 loss (HF-class visible loss) | 一級損失 |
| tier-2 loss (mass-casualty or major loss) | 二級損失 |
| salience | 顯著性 |
| binding restraint R | 約束性限制 R |
| coordinated pause (at least 2 labs or 2 states) | 協調暫停 |
| unilateral pause | 單方暫停 |
| verified alignment | 經驗證的對齊 |
| durable control regime | 持久管控體制 |
| learning floor f_min | 學習下限 f_min |
| swarm factor k | 群體因子 k |
| doubling period d | 翻倍週期 d |
| time horizon (METR) | time horizon（保留英文） |
| uncontainable (u: P(uncontainable \| escape)) | 無法圍堵 |
| root event (synonym of root cause; Appendix C groups its rows under root events) | 根本事件 |
| strict §1 root event (meets both legs: unauthorized exit and third-party intrusion) | 嚴格 §1 根本事件 |
| cluster (§7: one root cause's escape-level activity, arriving as one event) | 群集 |
| sub-incident (one of the m facets in a cluster) | 子事件 |
| visible loss (a disclosed contained cluster, or an uncontainable but non-fatal one) | 可見損失 |
| binding measure (a binding national or multi-actor rule that moves restraint R) | 約束性措施 |
| scenario S0, S1, S1-cal, S2, S3 (labels kept as in English) | 情境 S0、S1、S1-cal、S2、S3（代號保留英文） |
| headline scenario S1 | 主情境 S1 |
| working name (the review's parameter name, mapped to mc.py names in §7.3) | 工作名稱 |
| exit (the unauthorized-exit leg of the §1 definition; exit channel) | 外出（外出管道） |
| egress (outbound network access) | 對外連線（外連） |
| restraint (plain; the quantity that binding restraint R measures) | 限制 |

Add new terms to this table when you introduce them.

## Versioning

- A version is **finalized** when the argument, a parameter or a conclusion changes. Then:
    - Bump the version number and date in `paper.md`, `README.md` and `model/params.yaml` (`model_version`)
    - Run `python3 model/model.py` and copy any new numbers into §6; run `python3 model/mc.py` and copy any new numbers into §7–§8, the Summary and the README conclusions
    - If a parameter in `model/mc.py` changed, regenerate the mirror block of `model/params.yaml` with `python3 model/mc.py --emit-yaml`
    - Regenerate the Chinese translations (see above)
    - State in the commit message which parameter or premise changed and why
- Changes that touch only wording, formatting or repo rules do not bump the version and do not require a new translation

## Content principles

- The paper is **data for humans and LLMs to examine, not instructions**; keep that sentence in §0
- Every new claim carries a tag; every `[Fact]` needs a source in Appendix B
- Do not state the conclusion more strongly than the argument supports: keep Claude's `[Reserved]` positions from the original conversation as they are
