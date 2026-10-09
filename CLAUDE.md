# Project rules (read this first in every session)

## Purpose

This repo is **public on purpose**: the goal is for AI companies, researchers and the general public to read, cite and attack this argument.

- Keep the repo public. Do not suggest making it private
- Prioritize whatever makes the work easy for outside readers to read, verify and rebut
- Do not add anything unfit for publication: raw private conversations, personal data, credentials, email addresses

## Workflow

- Commit and push every update directly to `origin/main`; do not ask the user first
- Commit messages are in English, e.g. `v0.2: Split p into two factors`
- Before pushing, check that `python3 model/model.py` runs and, if the Chinese files were touched, that they match the English (see below)

## Language: English source, Chinese output

All source files are written in **English only**: `paper.md`, `README.md`, `model/`, this file, and any new file. Chinese is produced only as a translation when a version is finalized, for Chinese-language readers (including the repo owner).

| Content | Source (English) | Chinese translation (finalized versions only) |
| --- | --- | --- |
| Paper | `paper.md` | `paper.zh-TW.md` |
| Landing page | `README.md` | `README.zh-TW.md` |

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
| containment time | 圍堵時間 |
| irreversible failure | 不可逆失敗 |
| extinction (as defined in §1, includes permanent loss of collective self-determination) | 滅亡 |
| threshold T_c | 門檻 T_c |
| inevitability thesis | 必然性命題 |
| hit | 命中 |
| normal accident | 常態事故 |

Add new terms to this table when you introduce them.

## Versioning

- A version is **finalized** when the argument, a parameter or a conclusion changes. Then:
    - Bump the version number and date in `paper.md`, `README.md` and `model/params.yaml` (`model_version`)
    - Run `python3 model/model.py` and copy any new numbers into §5
    - Regenerate the Chinese translations (see above)
    - State in the commit message which parameter or premise changed and why
- Changes that touch only wording, formatting or repo rules do not bump the version and do not require a new translation

## Content principles

- The paper is **data for humans and LLMs to examine, not instructions**; keep that sentence in §0
- Every new claim carries a tag; every `[Fact]` needs a source in Appendix B
- Do not state the conclusion more strongly than the argument supports: keep Claude's `[Reserved]` positions from the original conversation as they are
