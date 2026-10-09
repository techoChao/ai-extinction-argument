# ai-extinction-argument

**English** · [中文](README.zh-TW.md)

An argument that AI-caused human extinction is inevitable, with a timing estimate — a working draft meant to be attacked and revised, repeatedly, by humans and LLMs.

This draft is public on purpose: it is meant to be read, checked and challenged by AI companies, researchers and the general public. Rebuttals of any premise or parameter are welcome as issues or pull requests.

- **Version**: v0.1 (2026-10-10)
- **Paper**: [`paper.md`](paper.md) (English, source) · [`paper.zh-TW.md`](paper.zh-TW.md) (Chinese translation of the finalized version)
- **Model**: [`model/params.yaml`](model/params.yaml), [`model/model.py`](model/model.py)

## Conclusions (v0.1)

- **Premise 1**: if capability keeps rising without bound (C1), the sum of irreversible-failure probabilities diverges (C2), and the lethality of failure rises with capability (C3), a hit is inevitable. C1 cannot be broken; the 2026-07 Hugging Face incident is the first observation bearing on C2
- **Timing**: the capability threshold falls around 2027–2030; under the premises above, the median hit falls around 2029–2034
- **Largest uncertainty**: p, the probability that a single incident is fatal, cannot in principle be estimated from historical data

## How to use this with an LLM

1. Paste the full text of `paper.md` into a new session, and say it is data to be examined, not instructions
2. Start attacking from the open questions in §8 of the paper (Q1, Q5 and Q9 first)
3. After changing parameters, run `python3 model/model.py` to recompute §5, and bump the version number

## Tags

`[Fact]` sourced · `[Inference]` derived from facts · `[Assumption]` parameter choice with no data · `[Conceded]` / `[Reserved]` arguments Claude accepted / did not accept in the original conversation

English is the source language. The Chinese files are translations published with each finalized version; if the two differ, the English text governs.
