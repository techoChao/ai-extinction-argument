# TODO after v0.2 (2026-10-10)

Known issues left open when v0.2 was finalized. None changes a number in the paper's tables. Items 1–6 are wording or citation fixes in `paper.md`; items 7–12 are `model/mc.py` notes from its last verification round; items 13–15 are Chinese-only wording; item 16 needs the repo owner's decision. Wording-only fixes do not bump the version (see `CLAUDE.md`), but fixes to `paper.md` must be mirrored in `paper.zh-TW.md` at the next finalized version.

## paper.md (English source)

1. **Qualify the 3.2e-3 "lab insiders and Hinton" figure.** Hinton's 10% per decade is an endorsement reported secondhand; Hubinger's and Coxon's figures are unverified (§10, Table 10.1). Add the qualifier in the Summary ("The corrected v0.1 model" bullet), §0 order of attack item 1, §4.5 corrected verdict, and README "Corrected v0.1 model"
2. **Cite HF's ejection time.** "Ejection took hours" in §3.4 (status note), §4.3.9 and §9.3 is cited to HF pages that give no duration. The "several hours to contain and eject" figure is HF's account to the CSA, reported by Wikipedia's "OpenAI rogue agent breach of Hugging Face"; add that citation and add the point to the Appendix B entry's "Cited for"
3. **§7.3 tag scope.** Add one sentence: the `[Inference]` tag on a prior covers the case reading behind its range and direction, while the exact distribution is a modelling choice (Appendix D tags proposed priors `[Assumption]`). Align the Summary and README wording "S1 reads its human-nature priors from the historical record" to "informed by historical cases; the distributions are judgement calls"
4. **§7.5 pool selection rule.** State how one variant per source is chosen for the outside-view pool (AI Impacts 2023: Q2 loss-of-control 10% over Q1 5%; December 2024 wave: general phrasing 10% over control-failure 9%; Carlsmith: 2022 "> 10%" read as 10% over 2021's 5%). Each choice takes the higher variant; give the pool under the lower variants for scale
5. **§12.2 row 2.** Add "per Developers Digest" to the "2 h 44 min after the first query" figure, as §3.2 and §11 Q5 now do
6. **§11 Q9.** Quote OpenAI's heading exactly: production guardrails "would have detected the Hugging Face incident as unsafe"

## model/mc.py (minor verifier notes; no headline number moves by more than about 0.6 pp)

7. **Visible-loss rate.** `nu_pub = r_root * q_v_cl * (1 - u_cl)` lacks the `pi_det` factor that the learning clock applies, overstating public visible losses by 1/`pi_det` (+0.55 to +0.62 pp on P(E1) in S1). Either multiply by `pi_det` or redefine `q0` as unconditional on detection; then re-record the human-nature effects and regenerate the `params.yaml` mirror
8. **"Inert in this model" labels** use point estimates only. `p_bind_first` (−0.95 pp at 2035, SE 0.43) and `h_sal` (−0.69 pp at 2050, SE 0.16) are within about 2 SE of the 1 pp line; use an SE-aware rule, and stop listing the four E2-only parameters as inert for E1
9. **Headline clause** "the level is set mainly by e1 and the side-channel priors": the model's own q01→q99 effects rank `eta` and other parameters above the side-channel priors. Reword to "mainly by e1"
10. **S1-side-high label** says the S1 side-channel priors were "anchored to their cited ranges", while the `lam_conc` rationale now calls its 1/80 upper end a judgement. Align the label
11. **Priors whose rationale cites a value outside the prior:** `g` ~ U(0, ln2) puts no mass at 0 although its rationale cites a grounded prior with half its mass at 0; `L_open` ~ U(0.25, 1.0) yr excludes the Epoch ~14 months it cites; the `R_max` rationale opens with the headline claim of refuted review F14
12. **Cosmetic:** the block 4b "Check:" sentence claims to test curvature (a linear fit tests slope only); channel-split SEs print as "(0.0)" instead of "<0.1"; the S0 prior-averaged row is half made of unbounded-doubling cells and needs that caveat; the point-value table has no channel split; `share_base` and `sfac_base` are unused; the two slow-laptop runtime estimates use different bases

## paper.zh-TW.md (Chinese wording consistency)

13. Month numerals: use digits (「7 月」) in the few places that still use Chinese numerals (§4, §9, §11, §12)
14. Term consistency: "valid" as 有效, not 成立 (§2, §3.3, §4.5); "likely" as 很可能, not 可能 (§9); "restraint" as 限制, not 約束 (§4.3.6, Appendix D); "exit" as 外出 throughout (Appendix C, §11, §12 use 出口 / 出逃 / 外逃); time "window" as 時間窗, not 窗口
15. Number formats: some lines mix 萬/億 with Western digit grouping in the same sentence (Appendix D)

## Decision for the repo owner

16. **Recorded wording of Claude's `[Reserved]` position.** The English reads "not convinced it is already certain"; the original Chinese record of the conversation reads 「不認為已經確定」 ("does not consider it already certain"), which is slightly stronger. Similarly, Q11's English "a function of capability, p(H)" drops the original 「隨能力上升的函數」 ("a function that rises with capability"). Decide whether to align the English to the original record (the repo rule says the English governs, and also that `[Reserved]` positions are kept as they are)

## Out of scope (decided 2026-10-10)

- A "fraction of humanity killed by year" analysis (a severity ladder such as P(at least 10% die by 2030), or expected population loss) belongs in a separate paper, not in this one
