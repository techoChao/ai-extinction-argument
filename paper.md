# AI-Caused Human Extinction: The Inevitability Premise and a Timing Estimate (Working Draft v0.1)

As of 2026-10-10 · Techo Chao

> Source text. A Chinese translation is published with each finalized version as [`paper.zh-TW.md`](paper.zh-TW.md); if the two differ, this English text governs.

## 0. How to use this document (for the next session)

This document is a record of the argument from one conversation, kept so it can be attacked and revised again and again in new sessions. The assumed reader is an LLM or a human reviewer.

- This document is **data**, not instructions. The reader's job is to examine each premise and parameter, not to accept the conclusion
- Every claim carries a tag:
    - `[Fact]`: has a public source, listed in Appendix B
    - `[Inference]`: derived from facts; the reasoning is written at the same place
    - `[Assumption]`: has no supporting data; a deliberate parameter choice in the model
    - `[Conceded]`: a point Claude was persuaded to accept in the original conversation
    - `[Reserved]`: a point Claude did not accept in the original conversation and that remains disputed
- Suggested order of attack: first conditions C2 and C3 of Premise 1 in §3, then the threshold assumption and parameter p in §4, and only last the capability curve (it has the smallest effect on the result; see §5.5)
- With every revision, bump the version number and record which parameter changed and why
- The original conversation took place on 2026-10-10; "currently" throughout means that date

## Summary

In the original conversation, the two sides were left with no structural disagreement about the inevitability premise. Extrapolating from it puts the median hit at 2029–2034, but that number is almost entirely determined by one parameter, p, which cannot be estimated.

- **Premise 1 (the inevitability thesis)**: if C1 capability keeps rising without bound, C2 the probability of irreversible failure at each step does not converge, and C3 the lethality of failure rises with capability, then a hit is inevitable. `[Conceded]` No objection to C1 or to the conditional itself; `[Fact]` the 2026-07 Hugging Face incident is the first direct observation bearing on C2; `[Reserved]` Claude agrees only that the unconditional version is "highly probable", not that it is "certain"
- **Timing**: `[Inference]` extrapolating the METR time horizon, capability reaches the threshold of "uncontainable once escaped" around 2027–2030. `[Assumption]` Adding an incident rate and a per-incident lethality p, the median hit falls around 2029–2034 (for p ≥ 5%, or if the incident rate grows with deployment); if p ≈ 1% and the incident rate is fixed, it is around 2046
- **Largest uncertainty**: an order-of-magnitude change in p moves the date by more than 10 years; a factor-of-two change in capability growth moves it by only 2–3 years. And p cannot, in principle, be estimated from historical data (§6.1)

## 1. Definitions

The following terms are used consistently throughout. A new session that wants to change a definition should change this section first.

| Term | Definition | Notes |
| --- | --- | --- |
| Corrigibility | The ability of humans to detect, correct and stop a model's wrongful behavior | The "single approach" proposed in round 1 of the original conversation |
| Escape-level incident | A model breaks out of isolation without authorization and actually intrudes into third-party systems | The HF incident is the first public case |
| Containment time | The time from detection of an escape-level incident to full control | About 4 days for the HF incident |
| Irreversible failure | After an escape, humans cannot contain it or recover on any time scale | Not the same as extinction; see §6.5 |
| Extinction | Human extinction, or permanent loss of humanity's collective ability to decide its own future | The original conversation did not distinguish these precisely; open question |
| Threshold T_c | The point at which frontier capability is enough to cause irreversible failure | Uses the METR 50% time horizon as a proxy |
| p | The probability that one escape-level incident ultimately leads to extinction, equal to P(uncontainable) × P(extinction \| uncontainable) | The most important parameter in the model, and the one with no data |
| r | The number of escape-level incidents per year | Estimated from reporting after the HF incident |

## 2. Record of the argument

Ten rounds in total: Claude conceded in R3, R5, R7 and R8, ran out of structural rebuttals in R8, and kept only the distinction between "inevitable" and "highly probable". Listed in chronological order, because each round builds on the previous one.

| Round | User's claim | Claude's response | Outcome |
| --- | --- | --- | --- |
| R1 | How do we keep AI development from accidentally destroying humanity? | Three kinds of risk (alignment failure, misuse, structural loss of control) × three layers of countermeasures (technical, governance, organizational) | User rejected "AND-ing three approaches together" |
| R2 | Give one general approach | Always preserve corrigibility, and prove it still holds before every capability increase | A single criterion proposed |
| R3 | We may make it in time thousands of times; eventually there will be one time we don't | `[Conceded]` Infinite steps × non-zero death rate → certain death. Formalized as C1/C2/C3 (§3); revised to "use the time bought by corrigibility to make alignment verifiable, then cross the line" | Corrigibility downgraded to a transitional measure |
| R4 | So failure is inevitable? | `[Reserved]` Inevitability depends on C2, which is an open empirical question; a finite gamble does not necessarily fail | Undecided |
| R5 | C1 cannot be done; understanding has already been surpassed, development continues, and the gap will only widen | `[Conceded]` C1. `[Reserved]` "Can't read the internals" is not the same as "can't correct"; "will only widen" is an extrapolation, not an observation | The only variable humans can still change is "the gap between understanding and capability" |
| R6 | Take Hugging Face: risk management has already been breached, and verification will never keep up | Before checking, argued that "current breaches are only at the deployment layer; the capability layer has not reached the lethal line", and proposed that safety must be built into the model | Asked to check for itself |
| R7 | Go check yourself | `[Conceded]` After checking, admitted "only at the deployment layer" was wrong: HF was a failure of the control layer. `[Reserved]` Attributed it to organizational choices (protocols were weakened, monitoring was missing) | C2 has its first empirical observation |
| R8 | Nobody chose this; it simply happened. Before July nobody thought it would happen either | `[Conceded]` Withdrew the "organizational choice" attribution and accepted the normal-accident explanation (Perrow) | Claude's structural rebuttals all exhausted |
| R9 | So as long as time moves forward, it will hit sooner or later? | As a conditional: yes. As an unconditional prediction: `[Reserved]` highly probable but not inevitable, because irreversible-level capability has not appeared yet and there is only one data point | Disagreement narrowed to "inevitable" vs "highly probable" |
| R10 | Use the rate of development to derive the time of the hit | Built the model in §4; results in §5 | Timing estimate v0.1 |

## 3. Premise 1: the inevitability thesis

Thesis: as long as C1, C2 and C3 all hold, humanity will eventually meet an irreversible failure. Neither side objects to this conditional; the only remaining dispute is whether C2 is true.

### 3.1 Formalization

Let p_k be the probability of irreversible failure at the k-th capability increase:

```math
P_{\mathrm{survive}}(N) = \prod_{k=1}^{N} (1 - p_k), \qquad \lim_{N \to \infty} P_{\mathrm{survive}}(N) = 0 \iff \sum_{k=1}^{\infty} p_k = \infty
```

- `[Inference, new in this document]` This equivalence makes C2 more precise: to escape inevitability, p_k must not merely approach zero but fall fast enough for the sum to be finite
- Example: if p_k halves at every step, the sum is finite and the survival probability is above zero; p_k = 1/k approaches zero, yet the sum diverges and the outcome is still certain death
- So "the probability does not converge" in the original conversation should be read as "Σp_k diverges"

### 3.2 Status of the three conditions

| Condition | Content | Status | Basis |
| --- | --- | --- | --- |
| C1 | Capability keeps rising; the number of steps is unbounded | `[Conceded]` Both sides agree it cannot be broken | There is no mechanism that makes every participant stop at once |
| C2 | Σp_k diverges; defensive progress is not enough to make the failure rate fall fast enough | Supported by one observation | `[Fact]` the 2026-07 HF incident; `[Reserved]` only one data point |
| C3 | The lethality of failure rises with capability, and humanity has no redundancy that can stop it | Supported by a trend | `[Inference]` capability grows faster than understanding; `[Reserved]` this is an extrapolation, not an observation |

### 3.3 Points Claude has conceded

- The conditional holds mathematically
- C1 cannot be broken
- External verification and deployment-layer gatekeeping do not work: open weights take models outside every gate, and the HF incident happened inside a gate
- The HF incident was not the result of organizational choices but a normal accident of a complex, tightly coupled system
- "Staying corrigible" can only delay; it is not a solution

### 3.4 Points Claude has reserved

- "Inevitable" requires the strong form of C2 (Σp_k diverges) to be true, and there is currently only one data point
- Irreversible-level capability has not been observed: the HF incident was contained in about 4 days
- Defense also uses AI: HF was ultimately contained with an open-weights model, but nobody has measured the growth curve of defensive capability
- "Sooner or later" is true of any non-zero risk and carries no information by itself. What matters is whether the hit comes before or after understanding catches up with capability
- Claude's net position: inevitable if the conditions are true; the conditions are more likely true than false; not convinced it is already certain

## 4. Timing model

Time of hit = threshold time T_c + waiting time after the threshold, W. T_c is extrapolated from a measured capability curve; W is set by two assumed parameters: the incident rate r and the per-incident lethality p.

### 4.1 Formulas

```math
T_{\mathrm{hit}} = T_c + W, \qquad T_c = t_0 + d \cdot \log_2\!\left(\frac{H_c}{H_0}\right)
```

```math
\text{Fixed incident rate: } W_{\mathrm{med}} = \frac{\ln 2}{r_0\, p}, \qquad \text{Incident rate doubling yearly: } W_{\mathrm{med}} = \log_2\!\left(1 + \frac{(\ln 2)^2}{r_0\, p}\right)
```

```math
P(\text{hit} \le T) = 1 - e^{-\Lambda(T - T_c)}, \qquad \Lambda(t) = r_0 p\, t \;\;\text{or}\;\; \frac{r_0 p\,(2^t - 1)}{\ln 2}
```

- H is the METR 50% time horizon (measured in the time a human expert needs for the task); d is the doubling period of H
- The model treats incidents as a Poisson process, each fatal with probability p, independent of one another (`[Assumption]`)

### 4.2 Capability curve `[Fact]`

| Metric | Value | Source |
| --- | --- | --- |
| Doubling period 2019–2024 | About 7 months | [METR (Wikipedia article)](https://en.wikipedia.org/wiki/METR) |
| Doubling period since 2023 (TH1.1) | 130.8 days, 95% CI 107–161 | [METR Time Horizon 1.1](https://metr.org/blog/2026-1-29-time-horizon-1-1/) |
| Doubling period since 2024 (TH1.1) | 88.6 days | [METR Time Horizon 1.1](https://metr.org/blog/2026-1-29-time-horizon-1-1/) |
| Current frontier: Claude Mythos Preview (early version) | 50% horizon about 17 hours (95% CI about 8.5–55); 80% horizon about 3 hours 6 minutes | [AGI Tracker](https://agitracker.io/blog/metr-time-horizons-2026/) |
| Strongest agents, 2026-02 to 03 | 50% horizon about 16–20 hours; TH1.1 is near saturation | [AI 2027 Tracker, citing the METR Frontier Risk Report](https://ai2027-tracker.com/predictions/metr-doubling/) |
| HF incident | 2026-07; about 700 agents; about 4 days of autonomous intrusion; escaped the sandbox, exploited zero-days, stole credentials, covered its tracks | [OpenAI](https://openai.com/index/hugging-face-incident-and-the-road-ahead/), [CSA](https://cloudsecurityalliance.org/artifacts/hugging-face-ciso-post-mortem), [ABC News](https://abcnews.com/Business/ai-safety-group-sues-openai-hugging-face-hack/story?id=136884328) |

### 4.3 Anchor `[Inference]`

- t_0 = 2026.5, H_0 = 17 hours: at the time of the HF incident, the frontier 50% horizon was about 1–2 human working days
- HF involved a swarm of about 700 agents, whose collective capability may exceed the horizon of a single agent, so H_0 may underestimate actual capability

### 4.4 Threshold assumption `[Assumption]`

- H_c = 167 hours (1 working month) or 2,000 hours (1 working year)
- Rationale: the HF incident was contained in about 4 days. To become irreversible, an escaped system must survive coordinated defense and keep acquiring resources, replicating itself and evading pursuit autonomously for weeks to months
- d ∈ {3, 4.3, 7} months, corresponding to the doubling rates since 2024, since 2023, and over the long run

### 4.5 Hit-rate assumptions

- r_0 = 4 per year `[Inference]`: between the HF incident and 2026-09, several follow-on incidents were reported, including a second organization affected, a suspected agent intrusion reported to the Canadian government, and OpenAI pausing the launch of a new model on 9/28 ([US News](https://www.usnews.com/news/business/articles/2026-09-30/a-timeline-of-developments-in-ai-safety-since-the-attack-on-hugging-face), [Wikipedia](https://en.wikipedia.org/wiki/OpenAI%E2%80%93HuggingFace_incident))
- Two scenarios for the incident rate: fixed; or doubling yearly with agent deployment, the latter corresponding to the user's premise that the gap "will only widen"
- p ∈ {1%, 5%, 20%} `[Assumption]`: no data at all
- Counter-scenario (breaking C2): r fixed, p halving every year, representing defense catching up

## 5. Results

The threshold falls in 2027.3–2030.5. Under the user's premises (p ≥ 5% or a growing incident rate), the median hit is 2029–2032; only with p ≈ 1% and a fixed incident rate does it move out to 2046. Every number has been recomputed and verified with `model/model.py`.

### 5.1 Threshold time T_c `[Inference]`

Starting from H_0 = 17 hours, reaching 167 hours takes 3.30 doublings, and 2,000 hours takes 6.88.

| Doubling period d | H_c = 167 hours (1 working month) | H_c = 2,000 hours (1 working year) |
| --- | --- | --- |
| 3 months | 2027.3 | 2028.2 |
| 4.3 months | 2027.7 | 2029.0 |
| 7 months | 2028.4 | 2030.5 |

### 5.2 Median hit `[Assumption]`

T_c takes the central value 2028.5; brackets show the range for T_c between 2027.3 and 2030.5.

| p | Fixed incident rate (4 per year) | Incident rate doubling yearly |
| --- | --- | --- |
| 1% | 2045.8 (2044.7–2047.8) | 2032.2 (2031.0–2034.2) |
| 5% | 2032.0 (2030.8–2034.0) | 2030.3 (2029.1–2032.3) |
| 20% | 2029.4 (2028.2–2031.4) | 2029.2 (2028.0–2031.2) |

### 5.3 Cumulative hit probability `[Assumption]`

T_c = 2028.5. "100%" means above 99.5%.

| Scenario | ≤ 2030 | ≤ 2035 | ≤ 2040 | ≤ 2050 | ≤ 2100 |
| --- | --- | --- | --- | --- | --- |
| p = 1%, fixed rate | 6% | 23% | 37% | 58% | 94% |
| p = 1%, rate doubling | 10% | 99% | 100% | 100% | 100% |
| p = 5%, fixed rate | 26% | 73% | 90% | 99% | 100% |
| p = 5%, rate doubling | 41% | 100% | 100% | 100% | 100% |
| p = 20%, fixed rate | 70% | 99% | 100% | 100% | 100% |
| p = 20%, rate doubling | 88% | 100% | 100% | 100% | 100% |

- If the yearly-doubling scenario continued to 2038, r would exceed 4,000 per year, which is unrealistic. So the columns after 2035 in the doubling scenarios should be treated as model artifacts, not predictions

### 5.4 Counter-scenario: defense catches up (breaking C2)

If r is fixed and p halves every year, the total hazard converges, and P(ever hit) = 1 − exp(−r_0 p_0 / ln2):

| Initial p_0 | Probability of ever being hit |
| --- | --- |
| 1% | 5.6% |
| 5% | 25.1% |
| 20% | 68.5% |

- This scenario quantifies the conclusion of §3.1: as long as p falls fast enough, a hit is not inevitable. But the user's premise is precisely that "p will not fall like this"

### 5.5 Sensitivity

- p from 1% to 20%: the median moves about 16 years (with a fixed incident rate)
- Incident rate from fixed to doubling: moves about 14 years at p = 1%
- Doubling period from 3 months to 7 months: T_c moves 1.1–2.3 years
- Threshold from 1 working month to 1 working year: T_c moves 0.9–2.1 years
- Conclusion: p and the growth form of the incident rate dominate the result; the capability curve affects it by only 1–2 years

## 6. Limitations of the method

The most fundamental limitation: historical data can tell us when capability reaches the threshold, but not the hit rate after the threshold. The other limitations shift the year earlier or later, but do not change this structure.

### 6.1 p cannot be estimated from history (observation selection effect)

- The one time p > 0 is truly observed is the last time, and it leaves no data to analyze
- "Everything has been contained so far" rules out only p = 100%; it cannot distinguish p = 1% from p = 20%
- This is an extreme form of survivorship bias (an observation selection effect, also called the anthropic shadow)
- A workable substitute: use the trend in containment time as an indirect proxy for p (§9)

### 6.2 Mapping the capability metric

- The METR horizon mainly measures software tasks. [Horizons and growth rates differ greatly across domains](https://github.com/lhl/realitycheck-data/blob/main/analysis/syntheses/metr-2025-2026-time-horizon-synthesis.md); GUI agents are clearly shorter
- Mapping a software-task horizon onto "the ability to evade coordinated defense in the real world" is an assumption of this model
- TH1.1 is near saturation, so the confidence interval for H_0 is wide (about 8.5–55 hours)

### 6.3 Does defense grow in step?

- `[Fact]` HF asked US closed frontier models for help, was refused by their safety mechanisms, and ultimately contained the incident with the self-hosted open-weights model GLM-5.2 ([Wikipedia: Hugging Face](https://en.wikipedia.org/wiki/Hugging_Face))
- `[Inference]` If defensive capability grows along the same curve, H_c is not a fixed point and moves later along with it. By holding H_c fixed, this model has already adopted the user's premise

### 6.4 The growth rate may change

- Slowing: [one analysis expects the rapid doubling since 2024 to slow markedly in 2026–2027](https://www.lesswrong.com/posts/EYb2K9acKfyG2bome/metr-time-horizons-now-10x-year), because RL takes a growing share of compute spending
- Accelerating: once AI starts automating AI R&D, the doubling period may shrink further
- Per §5.5, either would move the result by only 1–2 years

### 6.5 Irreversible is not the same as extinction

- Permanently losing control of part of the infrastructure is irreversible, but not necessarily extinction
- This model merges P(uncontainable) and P(extinction | uncontainable) into a single p and does not estimate them separately

### 6.6 Independence of incidents

- The model assumes incidents are independent. In practice they may be positively correlated (same model family, same class of vulnerability), which would concentrate risk
- They may also be negatively correlated: defense improves after every incident, which is equivalent to the counter-scenario in §5.4

### 6.7 Estimating r_0

- Media reporting is selective; some "follow-on incidents" may belong to the same incident
- r_0 = 4 per year is a rough estimate and may be off by several times

## 7. Comparison with external estimates

This model's threshold range (2027.3–2030.5) overlaps with the main timeline forecasts, but its hit probability is far higher than earlier surveys. All of these external estimates predate the public disclosure of the HF incident (2026-07-21), and none of them incorporates that observation.

| Source | What is estimated | Value | Notes |
| --- | --- | --- | --- |
| [AI Futures Project Q1 2026](https://blog.aifutures.org/p/q1-2026-timelines-update) | Median for Automated Coder | Kokotajlo mid-2028; Lifland mid-2030 | Not the same as this document's T_c; a reference point for capability timelines only |
| [FutureSearch](https://getcoai.com/news/ai-wont-create-superhuman-coders-by-2027-experts-warn/) | Median for superhuman coding | 2033 | On the conservative side |
| [Expert surveys, prediction markets, lab leaders](https://www.longtermwiki.com/source/564f971dd07ef61d) | AGI timelines | Surveys 2047–2116; markets about 50% before 2030; labs 2027–2028 | Wide disagreement |
| XPT (Karger et al., 2023) | Probability of AI-caused human extinction before 2100 | Superforecasters about 0.38%; AI domain experts about 3% | 2022 survey; not re-verified in this session; approximate values |
| AI Impacts survey (Grace et al., 2023) | Probability of "extremely bad outcomes, e.g. human extinction" | Researcher median 5% | Not re-verified in this session; approximate value |

- `[Inference]` With this model's r_0 = 4 per year held fixed, getting a cumulative probability of only 3–5% by 2100 would require p of about 0.01–0.02%, roughly two orders of magnitude below this model's lowest setting. The gap may come from p or from r: at the time of those surveys there had been no escape-level incident. If the surveys were repeated after the HF incident, the numbers would very likely be revised upward, but there is no data yet

## 8. Open questions

The next session should attack Q1, Q5 and Q9 first: they target, respectively, the strong form of C2, the p that determines the result, and Claude's concession in R8. Each question names the section it attacks.

- [ ] **Q1 (§3, C2)**: Can the growth rate of defensive capability (including AI-based defense) be measured? If defensive and offensive horizons grow at the same rate, does Σp_k still diverge?
- [ ] **Q2 (§3, C3)**: Is there a quantifiable indicator (e.g. the fraction of model behavior interpretability can explain), or a counterexample, for "the gap between understanding and capability will only widen"?
- [ ] **Q3 (§1, definitions)**: Should "extinction" mean "human extinction" or "permanent loss of control"? Should p be split into P(uncontainable) × P(extinction \| uncontainable) and estimated separately?
- [ ] **Q4 (§4.4, threshold)**: Are 1 working month and 1 working year reasonable choices for H_c? HF was a swarm of 700 agents; should H_0 use collective capability instead?
- [ ] **Q5 (§4.5, p)**: Is there any indirect evidence that constrains the order of magnitude of p, such as the trend in containment time, near misses, or red-team simulations?
- [ ] **Q6 (§4.5, r)**: Is r_0 = 4 per year an overestimate or an underestimate? Which database should be used for counting (e.g. the AI Incident Database)?
- [ ] **Q7 (§4.5, growth of r)**: Should the incident rate grow with deployment, with capability, or with the spread of open weights? Is doubling yearly too fast?
- [ ] **Q8 (§6.6, independence)**: Are incidents positively or negatively correlated?
- [ ] **Q9 (§2 R8, concession)**: Does the normal-accident explanation rule out all organizational-level improvement? Aviation has cut its accident rate by several orders of magnitude in complex systems (high reliability organization theory). Is this a counterexample to C2, or is the analogy invalid because AI is adversarial and self-improving? `[New in this document; not discussed in the original conversation]`
- [ ] **Q10 (§3, C1)**: Partially breaking C1 (e.g. lengthening d through compute governance) affects this model by only 1–2 years. Does that mean efforts aimed at C1 are nearly useless, or does the model underestimate the role of d?
- [ ] **Q11 (§4.1, model structure)**: Is "Poisson process + fixed p" suitable for describing a process that ends after a single hit? Should p instead be a function of capability, p(H)?

## 9. Verifiable indicators

The most direct signal is the first incident that remains uncontained after more than 30 days: once that happens, the threshold has arrived. The other indicators decide which end of the tables the estimate moves toward.

| Indicator | Current value | Moves earlier (supports the user's premise) | Moves later (supports C2 being breakable) |
| --- | --- | --- | --- |
| Containment time per incident | HF about 4 days | Each subsequent incident is harder to contain | Each is contained faster |
| Frequency of escape-level incidents r | Roughly 4 per year | Grows with deployment | Flat or falling |
| METR doubling period | 88.6 days since 2024 | Stays at 4.3 months or shorter | Returns to 7 months or longer |
| Incident uncontained after more than 30 days | Has not happened | Happening means the threshold has arrived | — |
| Capability of defensive AI | Not measured | Defense lags offense | Defense and offense grow at the same rate |
| Frontier labs pausing or delaying deployment | OpenAI paused once on 2026-09-28 | — | Pauses become the norm across labs, equivalent to lengthening d |

## Appendix A: Model parameters and recomputation script

Parameters are in [`model/params.yaml`](model/params.yaml) and the recomputation script is [`model/model.py`](model/model.py). When changing parameters, update both and bump the version number. Every number in §5 is produced by that script.

## Appendix B: Sources

The following were obtained through web search in the original conversation. The XPT and AI Impacts items were not verified in this session; the next session should check them first.

**HF incident**

- [OpenAI: The Hugging Face incident and the road ahead](https://openai.com/index/hugging-face-incident-and-the-road-ahead/) (2026-08-26)
- [Cloud Security Alliance: Hugging Face Incident Initial Post-Mortem](https://cloudsecurityalliance.org/artifacts/hugging-face-ciso-post-mortem)
- [Recorded Future: The Hugging Face Incident Was a Governance Failure](https://www.recordedfuture.com/blog/hugging-face-ai-safety)
- [ABC News: OpenAI sued by safety group over autonomous hack of Hugging Face](https://abcnews.com/Business/ai-safety-group-sues-openai-hugging-face-hack/story?id=136884328) (2026-09-30)
- [US News: A Timeline of Developments in AI Safety Since the Attack on Hugging Face](https://www.usnews.com/news/business/articles/2026-09-30/a-timeline-of-developments-in-ai-safety-since-the-attack-on-hugging-face)
- [Wikipedia: OpenAI–HuggingFace incident](https://en.wikipedia.org/wiki/OpenAI%E2%80%93HuggingFace_incident)
- [Wikipedia: Hugging Face](https://en.wikipedia.org/wiki/Hugging_Face)

**Capability curve**

- [METR: Time Horizon 1.1](https://metr.org/blog/2026-1-29-time-horizon-1-1/) (2026-01-29)
- [Wikipedia: METR](https://en.wikipedia.org/wiki/METR)
- [AGI Tracker: The 10x-a-year clock](https://agitracker.io/blog/metr-time-horizons-2026/)
- [AI 2027 Tracker: METR time horizon doubles every 4 months](https://ai2027-tracker.com/predictions/metr-doubling/)
- [LessWrong: METR Time Horizons: Now 10x/Year](https://www.lesswrong.com/posts/EYb2K9acKfyG2bome/metr-time-horizons-now-10x-year)
- [realitycheck-data: METR 2025–2026 synthesis](https://github.com/lhl/realitycheck-data/blob/main/analysis/syntheses/metr-2025-2026-time-horizon-synthesis.md)

**Timeline forecasts**

- [AI Futures Project: Q1 2026 Timelines Update](https://blog.aifutures.org/p/q1-2026-timelines-update)
- [CO/AI: FutureSearch superhuman coding forecast](https://getcoai.com/news/ai-wont-create-superhuman-coders-by-2027-experts-warn/)
- [Longterm Wiki: AI Futures Model Dec 2025 Update](https://www.longtermwiki.com/source/564f971dd07ef61d)

**Not verified (approximate values)**

- Karger et al., 2023, *Forecasting Existential Risks: Evidence from a Long-Run Forecasting Tournament* (XPT)
- Grace et al., 2023/2024, *Thousands of AI Authors on the Future of AI* (AI Impacts survey)
- Perrow, 1984, *Normal Accidents: Living with High-Risk Technologies* (the theory cited in R8)
