# ai-extinction-argument

**English** · [中文](README.zh-TW.md)

An argument that AI-caused human extinction is inevitable if its conditions hold, a structured review of that argument, and a probability model with explicit human-nature assumptions — a working draft meant to be attacked and revised, repeatedly, by humans and LLMs.

This draft is public on purpose: it is meant to be read, checked and challenged by AI companies, researchers and the general public. Rebuttals of any premise or parameter are welcome as issues or pull requests.

- **Version**: v0.2 (2026-10-10)
- **Paper**: [`paper.md`](paper.md) (English, source) · [`paper.zh-TW.md`](paper.zh-TW.md) (Chinese translation; its header names the finalized version it translates) · v0.1: [`paper.md` at commit db1c3b4](https://github.com/techoChao/ai-extinction-argument/blob/db1c3b48dfee2c56c2c179ca42e762d54e9ab455/paper.md)
- **Model**: [`model/model.py`](model/model.py) (v0.1 closed form, corrected and extended) · [`model/mc.py`](model/mc.py) (v0.2 Monte Carlo) · [`model/params.yaml`](model/params.yaml) (parameters and priors)
- **Review record**: [`review/2026-10-10-v0.1-review.md`](review/2026-10-10-v0.1-review.md) (the structured review of v0.1 that v0.2 answers)

## Conclusions (v0.2)

All probabilities are conditional on the paper's stated priors and model structure; they are not forecasts.

- **Premise 1**: `[Conceded]` if capability rises without bound (C1), the sum of irreversible-failure probabilities diverges (C2) and lethality rises with capability (C3), a hit eventually comes. `[Inference]` The conditional is valid but fixes no date; C1 holds as "no reliable mechanism", C2 is open, C3's widening is unmeasured. `[Reserved]` Claude's recorded position stands: inevitable if the conditions are true; the conditions are more likely true than false ("highly probable"); not convinced it is already certain. `[Inference]` "Highly probable" is a credence that C2 holds, not a model probability
- **Review of v0.1** (by Claude instances, so not independent of the original conversation's model family; paper §4.1): `[Inference]` its arithmetic is exact and its capability timing (threshold roughly 2027–2031) robust for the assumed threshold H_c, which is itself unvalidated, but r_0 = 4 incidents per year counted one correlated cluster plus two non-incidents, its 1% floor for p sits about 30–750 times above the p implied by central survey and superforecaster estimates, and the R8 concession is contradicted by OpenAI's own report (withdrawn on evidence and disputed; the recorded `[Conceded]` tag is kept)
- **Corrected v0.1 model**: `[Inference]` under v0.1's premises (r_0 = 4 incidents per year, and p of 5% or more or an incident rate that keeps doubling) the median hit stays at about 2029–2032; at p = 1% and a fixed rate it is about 2046; at the p that surveys and calibrated forecasters imply (roughly 1e-5 to 1.5e-3) and a fixed incident rate it moves more than a century later (after 2100 throughout; about 2202 at p = 0.1%), and at lab insiders' and Hinton's 3.2e-3 to about 2083
- **v0.2 model, headline scenario S1**: `[Inference]` S1 reads its human-nature priors from the historical record where one exists (11 of the 35 are assumptions). Its prior-predictive mean over the lethality prior, P(E1: extinction or permanent loss of collective self-determination), is 19.3% / 29.7% / 37.9% by 2035 / 2050 / 2100, of which accidental escape 18.8% / 27.5% / 32.9% and misuse or lock-in 0.5% / 2.2% / 5.0%; P(E2: population below 5,000) is 5.1% / 7.7% / 9.3%. For the lethality e1 from 1e-4 to 0.3 (its prior runs to 0.5), P(E1 by 2100) runs from 7.6% to 89.8%, at the low end mostly through the assumed misuse and lock-in hazards; this split, not the mean, is the main result (paper §8.3). The user's premises in full (S2) give 99.7% by 2035, which restates those premises; optimistic coordination (S3) gives 7.4% by 2100; the pooled outside view (five published anchors fielded before the incident, paper §7.5) gives 7.4% for E1 by 2100. Like for like, S1's accident channel (32.9%) sits about 4.5 times above that pool, and with e1 restricted to 1e-4 to 1e-2, roughly the range implied by the outside view (an order-of-magnitude match; S1-cal) P(E1 by 2100) is 13.7% (6.7% for the accident channel alone)
- **Largest uncertainty**: `[Inference]` e1, the probability that an uncontainable escape ends in E1, which has no observation behind it; moving it from its prior's q10 to its q90 moves P(E1 by 2050) by +82.3 percentage points; human nature as a block moves it by +53.2 or −25.6 points

## How to use this with an LLM

1. Paste the full text of `paper.md` into a new session, and say it is data to be examined, not instructions. The file is roughly 110k tokens; if the model's context is smaller, paste §0–§12 and add Appendix B, C or D only when a claim needs its sources
2. Attack in the order §0 of the paper gives: the lethality e1 first, then the human-nature parameters, then r_0 and its growth, and the capability curve last; in §11, start with Q14, Q13 and Q12
3. Run both scripts: `python3 model/model.py` (the corrected v0.1 tables in §6) and `python3 model/mc.py` (the v0.2 results in §7–§8; about a minute). After changing a parameter, rerun both and bump the version number

## Tags

`[Fact]` sourced · `[Inference]` derived from facts · `[Assumption]` parameter choice with no data · `[Conceded]` / `[Reserved]` arguments Claude accepted / did not accept in the original conversation

English is the source language. The Chinese files are translations of finalized versions, and each names the version it translates; if the two differ, the English text governs.
