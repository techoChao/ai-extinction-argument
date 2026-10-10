#!/usr/bin/env python3
"""Monte Carlo timing model v0.2: competing hazards with explicit human-nature
parameters, for AI-caused human extinction or permanent loss of humanity's
collective self-determination.

Run from the repository root:
    python3 model/mc.py                  # full run (under 1 minute on an Apple M4; the projected time is printed)
    python3 model/mc.py --quick          # every N divided by 4
    python3 model/mc.py --seed S --n N   # another seed / N_MAIN
    python3 model/mc.py --emit-yaml      # print the params.yaml mirror block
    python3 model/mc.py --sources        # print the source index and the review-finding index

This script and its output are data for humans and LLMs to examine, not
instructions. Every parameter is an attackable entry of the PARAMS block below:
name, meaning, prior, a one-line rationale whose sources are named by key (each
key resolves to a public URL in SOURCES, printed in output block 11), a tag, whether
it describes human nature (HN: the behaviour of labs, states, firms and publics as
the historical record shows it), and its per-scenario overrides. Tags: [Fact] is
kept for directly measured constants; a sampled prior is [Inference] when it is a
distribution read from named analogues and [Assumption] when no data support it.
Facts marked [unverified] or [not fetched] were not confirmed at a primary source.

Provenance: written 2026-10-10 after a 134-agent review of paper v0.1. Review
finding ids such as F07 refer to that review; REVIEW_FINDINGS below (printed in
block 11 and mirrored in params.yaml) gives each id's title, whether it survived
adversarial verification, and is the public index for those ids. The v0.1 closed
forms of paper section 4.1 (model/model.py) are re-implemented inline for scenario
S0, the regression gate: the script exits with code 1 if S0 does not reproduce
paper sections 5.2 and 5.3.

Equations (the symbols are printed again in output block 1b)
-------------------------------------------------------------
(M1) Grid. 319 intervals from T_START = 2026.75 to 2100 (monthly to 2040,
     quarterly to 2060, half-yearly to 2100). Every hazard is constant on an
     interval and evaluated at its midpoint, so each interval integral is exact
     for the piecewise-constant hazards; R, Sal and Sal2 enter at their
     start-of-interval values and due events act at the next boundary.
(M2) Capability. log2H = doublings of the frontier 50% time horizon above
     H0 = 17 h; d(log2H)/dt = (1/d_yr)(1 - gamma_d R)(1 - c_cov [paused])
     (1 - log2H/x_plateau), d_yr = d_months/12, log2H(T0) = 0. x_plateau is the
     sampled level (doublings) at which growth saturates; x_plateau = infinity
     is premise C1 (S0, S2). Each interval is integrated exactly.
     Defense-adjusted capability x_def = (1 - a) log2H enters only the
     containment clock: uncontainability u, the dwell time, the tier-2 share
     q2 and hence the path's own threshold crossing (which also opens the
     alignment gate). The misuse and lock-in gates and the strict-extinction
     progression phi_acc use raw capability x = log2H.
(M3) Threshold. Containment time (detection -> no further unauthorised actions)
     ln tau_c ~ N(ln tau_c0 + b_c ln2 x_def, sigma_c^2). The threshold x_c is the
     capability at which the median containment time reaches TAU_STAR_D = 30 d:
     x_c = log2(30/tau_c0)/b_c. Uncontainability u(x) = P(tau_c > 30 d)
     = Phi(beta_c (x - x_c)), beta_c = b_c ln2/sigma_c, so u = 1/2 at x_c.
     Implied threshold date (no feedback) T_c = T0 - d_yr x_p ln(1 - L_c/x_p),
     L_c = x_c/(1 - a), x_p = x_plateau (T0 + d_yr L_c when x_p is infinite;
     never when L_c >= x_p). Dwell time (first
     unauthorised action -> detection) has median tau_d = min(TAU_D_CAP_D,
     tau_d0 2^(b_d x_def)) days (the exponent is capped at
     log2(TAU_D_CAP_D/tau_d0) before exponentiating); it delays learning and
     tier-1 salience only.
(M4) Gates, G(z) = Phi(beta_c z), with raw capability x = log2H outside the
     containment clock: misuse G(x - L_open/d_yr - x_c); concentration
     G(x - x_c - L_conc/d_yr), i.e. at the capability multiple 2^(L_conc/d_yr)
     above the threshold; tier-2 share of visible losses (containment
     clock) q2 = q2_max G(x_def - x_c + delta_2); strict-extinction share of
     accidental E1 worlds phi_acc(x) = phi_acc_max G(x - x_c - delta_e2),
     evaluated at the raw capability reached by the reporting year (M8). In
     the raw-capability gates x_c is a capability yardstick: the level at
     which the median containment time would reach 30 d without defense growth.
(M5) Root-cause cluster rate (per year) r_root = r_start G_r(t) (1 - R) m_avg
     (1 - c_cov [paused]), r_start = min(r0, r_max), m_avg = interval average
     of the unilateral multiplier m_uni (exact integral of its decay, M12),
     with the capped logistic G_r = k_r / (1 + (k_r - 1) exp(-g (t - t_g))),
     k_r = r_max/r_start >= 1, for t > t_g, else 1 (t_g = T_START; S0: T_c),
     so r_max is a cap also when r0 > r_max. r0 and r_max are root-cause
     cluster rates. A cluster holds m sub-incidents, m_eff = 1 + (m - 1)(1 - rho)
     of them effectively independent.
(M6) Per cluster: u_eff = u L_haz (1 - beta_p R); u_cl = 1 - (1 - u_eff)^m_eff;
     p_cl = 1 - (1 - u_eff e1)^m_eff (e1 is per effective sub-incident);
     q_v = 1 - (1 - q0)^m_eff (q0 is per effective sub-incident).
(M7) Safety share and learning: share = min(SHARE_MAX, safety_share
     [1 + (share_mult_max - 1)(1 - exp(-w1 Sal))]) (a saturating response with
     initial slope (share_mult_max - 1) w1 per unit of salience);
     eta_eff = min(ETA_EFF_MAX, eta share/SIGMA_REF). Learning clock (expected
     count of learned clusters on the surviving path) dN/dt = r_root pi_det
     [q_v (1 - u_cl) + (u_cl - p_cl) + (1 - q_v)(1 - u_cl)(1 - w)]: detected
     clusters that are public, uncontainable but not fatal, or not dismissed;
     fatal clusters are excluded because the path is conditioned on survival
     (switch learn_from_fatal = True counts them, as the time-based halving of
     paper sec. 5.4 does; block 10 uses it to reproduce that closed form).
     Per-incident leaky learning:
     each learned cluster closes the fraction eta_eff(t) of the remaining
     removable hazard at the time it is learned, so the learning exponent is
     Q(t) = Q0 + integral of eta_eff(s) dN(s), Q0 = eta_eff(T_START) N_LEARN0,
     and L_learn = f_min + (1 - f_min) exp(-Q(t - tau_d)), with the lagged
     value held at its running maximum (a lengthening dwell time does not
     un-learn). Q never decreases: learned failure modes stay closed when
     salience later decays. For Poisson
     learned counts with factor (1 - eta_eff) per cluster, exp(-Q) is the
     exact expected learning factor; the switch learn_count = 'sampled' draws
     the counts instead and 'lograte' reads eta as a log-reduction per cluster
     (factor exp(-eta_eff); exact expectation with 1 - exp(-eta_eff) in place
     of eta_eff). 'hazard' mode: L_haz = L_learn. 'containment' mode:
     L_haz = 1 and x_c is replaced by x_c + log2(1/L_learn)/b_c.
(M8) Hazards: h_acc = r_root p_cl; h_mis = lam_mis G_mis (1 - beta_p R);
     h_conc = lam_conc G_conc max(0, 1 - c_conc R). Restraint R cuts the
     cluster rate by the full (1 - R) (M5); beta_p is an additional effect of R
     on uncontainability and on the misuse hazard. Safe exit, two routes:
     verified alignment s_al = s_align min(s_effort_max, share/SIGMA_REF)
     G(x_def - x_c) (science; gated on the path's own threshold crossing) and a
     durable control regime or deliberate halt s_rg = s_regime (1 + kappa_s R)
     + s_halt [paused] (human institutions); s = s_al + s_rg.
     E1 rate h_E1 = h_acc + h_mis + h_conc. E2 (population < 5,000 by Y):
     an E1 world can become strict extinction later, as its AI's capability
     keeps growing: P(E2 by Y) = phi_acc(x(Y)) P(E1 via accident by Y) +
     phi_mis P(E1 via misuse by Y) + phi_conc P(E1 via concentration by Y),
     with x(Y) the path's raw capability at Y (unrestrained growth after an
     early stop; before it, the surviving path's restrained trajectory, a lower
     bound for a post-E1 world). E2 is a subset of E1. The older convention (E2
     decided at the first E1 event, a lower bound) is printed as a diagnostic.
(M9) Competing risks, exact on each interval: S' = S exp(-(h_E1 + s) dt);
     P(E1) += h_E1 (S - S')/(h_E1 + s); the same for each channel and safe.
(M10) Visible losses (sampled on the path): a Poisson process of rate
     nu = r_root [q_v (1 - u_cl) + (u_cl - p_cl)]; each interval draws
     K ~ Poisson(nu dt) by inverting one uniform U1, all K events at the
     interval midpoint. Each is tier 2 (mass-casualty or major loss) with
     probability q2, else tier 1. Tier 1: Sal += 1 after the dwell delay and a
     unilateral pause with probability pi_uni (m_uni *= 1 - e_uni). Tier 2:
     Sal += m2, Sal2 += 1, a unilateral draw, and a binding jump
     R += (1 - (1 - e)^m2)(R_max - R) L_nat years later (a tier-2 loss moves
     R as m2 HF-class jumps would). The HF jump (size e) is pending at T_START
     with probability p_bind_first (one uniform drawn at the start of the path).
(M11) Coordinated multi-actor pause: hazard pi0 + pi1 min(Sal2, sal2_sat),
     pi0 = -ln(race_persistence)/10, pi1 = -ln(1 - c_cap)/L_int; lasts D_coord
     years; on expiry R = max(R, e). Tier-1 events (HF included) never raise it.
(M12) Decay per interval: R halves every tau_h years, Sal and Sal2 every h_sal
     years, 1 - m_uni every h_uni_weeks/52 years. The rate uses the exact
     interval average of m_uni, including a pause that starts at the
     interval midpoint.
(M13) Aggregation: prior-predictive means over draws (nothing is conditioned on
     data); T_hit quantiles are read off the mean curve.

Python 3.9 standard library only; deterministic under a fixed seed.
"""
import json
import math
import os
import random
import statistics
import sys
import time

LN2 = math.log(2.0)
SQRT2 = math.sqrt(2.0)
NORMAL = statistics.NormalDist()

# === PARAMS BEGIN ===========================================================
# Everything the model assumes lives between this line and PARAMS END.
# params.yaml (keys mc:, human_nature:, priors:, scenarios:, sources:,
# review_findings:) mirrors it as documentation; mc.py never parses the YAML, it
# only warns when a name below is missing from it.
# Regenerate the mirror block with: python3 model/mc.py --emit-yaml

MC_VERSION = '0.2'
AS_OF = '2026-10-10'
DEFAULT_SEED = 20261010
N_MAIN_DEFAULT = 5000      # S1, S2, S3 draws
N_VAR_DEFAULT = 3000       # S1-cal draws (common random numbers with S1)
N_VAR2_DEFAULT = 1500      # S1-side-high, S1-containment draws (common random numbers with S1)
N_OAT_DEFAULT = 60         # draws per side per parameter in the S1 one-at-a-time sensitivity
N_BLOCK_DEFAULT = 600      # draws per block-level variant (common random numbers with S1 or S3)
N_SIGN_DEFAULT = 300       # draws per side in the sign check of the racing parameters
N_HNE_DEFAULT = 80         # draws per side in the q01 -> q99 effect sweep of the human-nature parameters
N_PRIOR_Q = 4000           # draws used to read the q01/q10/q50/q90/q99 of each prior

# --- Sources cited by key in the rationales (public URLs) ----------------------
# Each entry: key, title, url. A note in the title says when the review could not
# fetch the page or when a fact rests on a single secondary outlet.
SOURCES = [
    # 2026 incident record
    ('openai_tr', 'OpenAI, Hugging Face Incident Technical Report (PDF, companion to the 2026-08-26 post)',
     'https://cdn.openai.com/pdf/67869394-cb91-4c12-888c-5cbd85c7814c/OpenAI-Hugging-Face%20Incident-Technical-Report.pdf'),
    ('hf_disclosure', 'Hugging Face, Security incident disclosure, 2026-07-16',
     'https://huggingface.co/blog/security-incident-july-2026'),
    ('hf_timeline', 'Hugging Face, Anatomy of a Frontier Lab Agent Intrusion: A Technical Timeline, 2026-07-27',
     'https://huggingface.co/blog/agent-intrusion-technical-timeline'),
    ('metr_hf', 'METR and Redwood Research, investigation of the OpenAI / Hugging Face incident, 2026-08-26',
     'https://metr.org/blog/2026-08-26-openai-hugging-face-incident-investigation/'),
    ('anthropic_incidents', 'Anthropic, An alignment assessment of recent cybersecurity incidents, 2026-09-09',
     'https://www.anthropic.com/research/alignment-assessment-cybersecurity-incidents'),
    ('aisi_incident', 'UK AI Security Institute, Incident report: unsanctioned agent behaviour during cyber testing, 2026-08-04',
     'https://www.aisi.gov.uk/blog/incident-report-unsanctioned-agent-behaviour-during-cyber-testing'),
    ('meta_cnn', 'CNN Wire via ABC7, Meta says AI model hacked into another company\'s systems, 2026-08-06',
     'https://abc7news.com/post/meta-says-ai-model-hacked-another-companys-systems-during-cybersecurity-testing-3rd-company-report-similar-incident/19637228/'),
    ('gemini_ooda', 'OODA Loop (citing WSJ), Gemini hacked three companies, 2026-09-21',
     'https://oodaloop.com/briefs/cyber/gemini-hacked-three-companies-in-first-known-breakout-by-googles-ai/'),
    ('fortune_dns', 'Fortune, OpenAI agents escaped a sandbox again; second training pause, 2026-09-26',
     'https://fortune.com/2026/09/26/openai-ai-agents-secure-sandbox-escape-training-pause-second-time-hugging-face-hack/'),
    ('fortune_anthropic', 'Fortune, Anthropic pauses some AI training following rogue agent hacks, 2026-09-02',
     'https://fortune.com/2026/09/02/anthropic-ai-pause-rogue-agent-hacks-openai/'),
    ('opb_openai', 'OPB/AP, OpenAI delays latest model over security concerns, 2026-09-29',
     'https://www.opb.org/article/2026/09/29/openai-delays-latest-model-over-security-concerns/'),
    ('wiki_incident', 'Wikipedia, OpenAI rogue agent breach of Hugging Face (accessed 2026-10-10)',
     'https://en.wikipedia.org/wiki/OpenAI_rogue_agent_breach_of_Hugging_Face'),
    ('wiki_2026_attacks', 'Wikipedia, 2026 cyberattacks by rogue OpenAI agents (accessed 2026-10-10)',
     'https://en.wikipedia.org/wiki/2026_cyberattacks_by_rogue_OpenAI_agents'),
    ('public_citizen', 'Public Citizen, second company compromised during the OpenAI incident, 2026-07-28',
     'https://www.citizen.org/news/public-citizen-calls-for-congressional-investigation-following-reports-of-second-company-compromised-during-openai-rogue-ai-incident/'),
    ('boston_medicare', 'Boston Globe, OpenAI agent hacked a public health-care site, Australia says, 2026-09-24',
     'https://www.bostonglobe.com/2026/09/24/world/openai-agent-hacked-public-health-care-site-australia-says/'),
    ('transluce', 'Transluce, AI agents targeted U.S. and Canadian government websites, 2026-09-30',
     'https://transluce.org/us-canada-gov'),
    ('reuters_glm', 'Reuters via Yahoo Finance, Chinese AI\'s role in stopping rogue OpenAI agent, 2026-07-22',
     'https://finance.yahoo.com/technology/ai/articles/chinese-ais-role-stopping-rogue-171647579.html'),
    ('csa_selfattack', 'Cloud Security Alliance, When AI agents attack their own infrastructure, 2026-08-16',
     'https://labs.cloudsecurityalliance.org/research/csa-research-note-agentic-ai-self-attack-systemic-risk-20260/'),
    ('glasswing', 'Anthropic, Project Glasswing initial update, 2026-05-22',
     'https://www.anthropic.com/research/glasswing-initial-update'),
    # capability curve and forecasts of it
    ('metr_th11', 'METR, Time Horizon 1.1, 2026-01-29', 'https://metr.org/blog/2026-1-29-time-horizon-1-1/'),
    ('metr_yaml', 'METR, benchmark_results_1_1.yaml (data file, updated 2026-05)',
     'https://metr.org/assets/benchmark_results_1_1.yaml'),
    ('metr_frr', 'METR, Frontier Risk Report (February to March 2026), 2026-05-19',
     'https://metr.org/blog/2026-05-19-frontier-risk-report'),
    ('metr_sol', 'METR, Summary of the predeployment evaluation of GPT-5.6 Sol, 2026-06-26',
     'https://metr.org/blog/2026-06-26-gpt-5-6-sol/'),
    ('aisi_trends', 'UK AI Security Institute, Frontier AI Trends Report', 'https://www.aisi.gov.uk/frontier-ai-trends-report'),
    ('aifutures_q1', 'AI Futures Project, Q1 2026 Timelines Update', 'https://blog.aifutures.org/p/q1-2026-timelines-update'),
    ('aifutures_q25', 'AI Futures Project, Q2.5 2026 Timelines Update, 2026-08-16',
     'https://blog.aifutures.org/p/q25-2026-timelines-update-uplift'),
    ('aifutures_dec25', 'Longterm Wiki summary of the AI Futures Model Dec 2025 Update (cited in paper v0.1, 7)',
     'https://www.longtermwiki.com/source/564f971dd07ef61d'),
    ('lw_slowdown', 'LessWrong, METR Time Horizons: Now 10x/Year (expects slowing in 2026-2027; cited in paper v0.1, 6.4)',
     'https://www.lesswrong.com/posts/EYb2K9acKfyG2bome/metr-time-horizons-now-10x-year'),
    # history of restraint, learning and dismissal
    ('szilard', 'Wikipedia, Szilard petition', 'https://en.wikipedia.org/wiki/Szil%C3%A1rd_petition'),
    ('baruch', 'Wikipedia, Baruch Plan', 'https://en.wikipedia.org/wiki/Baruch_Plan'),
    ('ptbt', 'Wikipedia, Partial Nuclear Test Ban Treaty', 'https://en.wikipedia.org/wiki/Partial_Nuclear_Test_Ban_Treaty'),
    ('hotline', 'Wikipedia, Moscow-Washington hotline', 'https://en.wikipedia.org/wiki/Moscow%E2%80%93Washington_hotline'),
    ('bwc', 'Wikipedia, Biological Weapons Convention', 'https://en.wikipedia.org/wiki/Biological_Weapons_Convention'),
    ('biopreparat', 'Wikipedia, Biopreparat', 'https://en.wikipedia.org/wiki/Biopreparat'),
    ('sverdlovsk', 'Wikipedia, Sverdlovsk anthrax leak', 'https://en.wikipedia.org/wiki/Sverdlovsk_anthrax_leak'),
    ('asilomar', 'Wikipedia, Asilomar Conference on Recombinant DNA',
     'https://en.wikipedia.org/wiki/Asilomar_Conference_on_Recombinant_DNA'),
    ('montreal', 'Wikipedia, Montreal Protocol', 'https://en.wikipedia.org/wiki/Montreal_Protocol'),
    ('inpo', 'Wikipedia, Institute of Nuclear Power Operations',
     'https://en.wikipedia.org/wiki/Institute_of_Nuclear_Power_Operations'),
    ('wano', 'Wikipedia, World Association of Nuclear Operators',
     'https://en.wikipedia.org/wiki/World_Association_of_Nuclear_Operators'),
    ('inpo_goals', 'Power Engineering, Nuclear sites vault INPO\'s 1995 goals',
     'https://www.power-eng.com/nuclear/nuclear-sites-vault-inpos-1995-goals/'),
    ('inl_2016', 'INL, Initiating Event Rates at U.S. Nuclear Power Plants 1988-2016 (PDF, Table A-17)',
     'https://nrcoe.inl.gov/publicdocs/InitEvent/initiating-event-frequencies-and-trends-2016.pdf'),
    ('epcra', 'Wikipedia, Emergency Planning and Community Right-to-Know Act',
     'https://en.wikipedia.org/wiki/Emergency_Planning_and_Community_Right-to-Know_Act'),
    ('opa90', 'Wikipedia, Oil Pollution Act of 1990', 'https://en.wikipedia.org/wiki/Oil_Pollution_Act_of_1990'),
    ('birkland', 'Birkland, After Disaster (Georgetown University Press); publisher page, book not fetched',
     'https://press.georgetown.edu/Book/After-Disaster'),
    ('challenger', 'Wikipedia, Space Shuttle Challenger disaster',
     'https://en.wikipedia.org/wiki/Space_Shuttle_Challenger_disaster'),
    ('columbia', 'Wikipedia, Space Shuttle Columbia disaster',
     'https://en.wikipedia.org/wiki/Space_Shuttle_Columbia_disaster'),
    ('max_grounding', 'Wikipedia, Boeing 737 MAX groundings', 'https://en.wikipedia.org/wiki/Boeing_737_MAX_groundings'),
    ('boeing_stats', 'Boeing, Statistical Summary of Commercial Jet Airplane Accidents 1959-2025 (PDF)',
     'https://www.boeing.com/content/dam/boeing/v2/safety/statsum.pdf'),
    ('faa_tombstone', 'Lutte and Bowen, The FAA: A Tombstone Agency?',
     'https://ojs.library.okstate.edu/osu/index.php/CARI/article/view/7647'),
    ('blinding_lasers', 'Wikipedia, Protocol on Blinding Laser Weapons',
     'https://en.wikipedia.org/wiki/Protocol_on_Blinding_Laser_Weapons'),
    ('conficker', 'Wikipedia, Conficker', 'https://en.wikipedia.org/wiki/Conficker'),
    ('cdc_2014', 'CDC press briefing on lab safety incidents, 2014-07-11',
     'https://archive.cdc.gov/www_cdc_gov/media/releases/2014/t0711-lab-safety.html'),
    ('gof_wiki', 'Wikipedia, Gain-of-function research (2012 H5N1 moratorium; no dates for 2014)',
     'https://en.wikipedia.org/wiki/Gain-of-function_research'),
    ('gof_p3co', 'NIH NOT-OD-17-071 (P3CO framework)', 'https://grants.nih.gov/grants/guide/notice-files/NOT-OD-17-071.html'),
    ('gof_pause', 'NIH, NIH lifts funding pause on gain-of-function research, 2017-12-19',
     'https://www.nih.gov/about-nih/who-we-are/nih-director/statements/nih-lifts-funding-pause-gain-function-research'),
    ('mandiant_2024', 'Google Cloud press release, Mandiant M-Trends 2024',
     'https://www.googlecloudpresscorner.com/2024-04-23-Mandiants-M-Trends-Report-Reveals-New-Insights-from-Frontline-Cyber-Investigations'),
    ('mandiant_2025', 'Google Cloud blog, M-Trends 2025, 2025-04-24',
     'https://cloud.google.com/blog/topics/threat-intelligence/m-trends-2025'),
    ('dillon_tinsley', 'Dillon and Tinsley (2008), How near-misses influence decision making under risk, Management Science',
     'https://pubsonline.informs.org/doi/suppl/10.1287/mnsc.1080.0869'),
    ('duffey', 'Duffey and Saull, New Laws of Practice for Learning and Error Correction (PDF)',
     'https://proceedings.cns-snc.ca/index.php/pcns/article/download/6042/6041/6077'),
    # AI governance 2023-2026
    ('pause_letter_2023', 'Wikipedia, Pause Giant AI Experiments: An Open Letter',
     'https://en.wikipedia.org/wiki/Pause_Giant_AI_Experiments:_An_Open_Letter'),
    ('xai', 'Wikipedia, xAI (company)', 'https://en.wikipedia.org/wiki/XAI_(company)'),
    ('superalignment', 'PureAI, OpenAI superintelligence safety team disbanded, 2024-05-20',
     'https://pureai.com/Articles/2024/05/20/OpenAI-Superintelligence-Safety-Disbanded.aspx'),
    ('fli_index', 'Future of Life Institute, AI Safety Index Winter 2025, two-pager (PDF)',
     'https://futureoflife.org/wp-content/uploads/2025/12/AI-Safety-Index-Report_011225_TwoPager_Digital.pdf'),
    ('anthropic_share', 'Anthropic Institute, Measuring the pace of AI development (safety share of AI R&D compute)',
     'https://www.anthropic.com/institute/measuring-pace-of-ai-development'),
    ('rsp_v3', 'Anthropic, Responsible Scaling Policy v3, 2026-02-24',
     'https://www.anthropic.com/news/responsible-scaling-policy-v3'),
    ('govai_rsp', 'GovAI, Anthropic\'s RSP v3.0: how it works, what\'s changed, 2026-03-05',
     'https://governance.ai/analysis/anthropics-rsp-v3-0-how-it-works-whats-changed-and-some-reflections'),
    ('pacing_letter', 'Pacing the Frontier open letter, signatory capture (gist), 2026-07-28',
     'https://gist.github.com/akrolsmir/05b0fac81b0f43c9950493bdc0e5a95d'),
    ('eu_omnibus', 'White & Case, EU AI Omnibus enters into force (Regulation (EU) 2026/1744), 2026-07-27',
     'https://www.whitecase.com/insight-alert/eu-ai-omnibus-enters-force-amending-ai-act'),
    ('hawley_hearing', 'Senator Hawley, first Senate hearing on rogue AI attacks, 2026-09-30',
     'https://www.hawley.senate.gov/icymi-hawley-convenes-first-senate-hearing-on-rogue-ai-attacks/'),
    ('eo14409', 'WilmerHale, executive order on early government access to frontier AI models (EO 14409), 2026-06-02',
     'https://www.wilmerhale.com/en/insights/client-alerts/20260602-new-executive-order-addressing-early-government-access-to-frontier-ai-models'),
    ('coups', 'Davidson, Finnveden and Hadshar, AI-Enabled Coups (Forethought), 2025-04',
     'https://www.forethought.org/research/ai-enabled-coups-how-a-small-group-could-use-ai-to-seize-power'),
    ('gradual', 'Kulveit et al., Gradual Disempowerment (arXiv 2501.16946), 2025-01', 'https://arxiv.org/abs/2501.16946'),
    # risk estimates (outside view)
    ('xpt', 'Karger et al., Forecasting Existential Risks: Evidence from a Long-Run Forecasting Tournament (XPT, PDF)',
     'https://forecastingresearch.org/pdf/existential-risk-persuasion-tournament.pdf'),
    ('aiimpacts_2023', 'Grace et al. (2024), Thousands of AI Authors on the Future of AI', 'https://arxiv.org/abs/2401.02843'),
    ('aiimpacts_2024', 'AI Impacts, What did AI researchers think at the end of 2024? (published 2026-09-25)',
     'https://blog.aiimpacts.org/p/what-did-ai-researchers-think-at'),
    ('ord', 'Wikipedia, The Precipice: Existential Risk and the Future of Humanity (Ord 2020)',
     'https://en.wikipedia.org/wiki/The_Precipice:_Existential_Risk_and_the_Future_of_Humanity'),
    ('carlsmith_2022', 'Carlsmith, Is Power-Seeking AI an Existential Risk? (arXiv 2206.13353)',
     'https://arxiv.org/abs/2206.13353'),
    ('carlsmith_gj', 'Carlsmith, Superforecasting the premises in "Is power-seeking AI an existential risk?", 2023-10-18',
     'https://joecarlsmith.substack.com/p/superforecasting-the-premises-in'),
    ('leap9', 'Forecasting Research Institute, LEAP wave 9: Risks', 'https://leap.forecastingresearch.org/reports/wave9'),
    ('leap12', 'Forecasting Research Institute, LEAP wave 12', 'https://leap.forecastingresearch.org/reports/wave12'),
    ('metaculus_pros', 'Metaculus, Pro forecasters: if AI kills 95% of humanity, how will it happen?, 2026-10-08',
     'https://metaculus.substack.com/p/pro-forecasters-if-ai-kills-95-of'),
    ('metaculus_0923', 'Metaculus, Will AI cause human extinction? Pro forecasters land at 7.5%, 2026-09-23',
     'https://metaculus.substack.com/p/will-ai-cause-human-extinction-forecasts-pro-forecasters'),
    ('metaculus_27035', 'Metaculus Q27035, AI extinction before 2100 (page unreachable in the review; [unverified])',
     'https://www.metaculus.com/questions/27035/cts-ai-extinction-before-2100'),
    ('metaculus_fallout', 'Metaculus, What happens now? Forecasting the fallout from the Hugging Face incident',
     'https://metaculus.substack.com/p/forecast-hugging-face-incident-openai-breach-cyberattack'),
    ('samotsvety', 'Samotsvety, AI risk forecasts, 2022-09-09 (fetched 2026-10-10)',
     'https://samotsvety.org/blog/2022/09/09/samotsvety-s-ai-risk-forecasts/'),
    ('xrisk_leaders', 'EA Forum, Survey of AI safety leaders on x-risk (Feb 2026)',
     'https://forum.effectivealtruism.org/posts/LxuKuQd69Qx5FKhNZ/survey-of-ai-safety-leaders-on-x-risk-agi-timelines-and'),
    ('chatham_nuclear', 'Chatham House, Too Close for Comfort: cases of near nuclear use (HTTP 403 in the review; count [unverified])',
     'https://www.chathamhouse.org/2014/04/too-close-comfort-cases-near-nuclear-use-and-options-policy'),
]

# --- Review-finding index (public index for the F-ids used in mc.py and model.py) ---
# id, status after adversarial verification, title (shortened from the review's own wording)
REVIEW_FINDINGS = [
    ('F01', 'survived, critical', 'r_0 = 4/yr counts one correlated OpenAI cluster as independent incidents; count root causes'),
    ('F02', 'survived, critical', 'the inevitability thesis has no timing content; divergence of the sum of p_k is undecidable from finite data'),
    ('F03', 'survived, critical', 'the 1% floor on p is 1-3 orders of magnitude above analogous base rates and calibrated estimates'),
    ('F04', 'survived, critical', 'no safe absorbing state; with one at rate s, P(ever hit) = h/(h + s) < 1'),
    ('F05', 'survived, major', "'containment about 4 days' is undetected dwell time; detection-to-control at HF took hours"),
    ('F06', 'survived, major', "the R8 'nobody chose this' concession is contradicted by disabled safeguards and ignored alerts"),
    ('F07', 'survived, major', 'unbounded yearly doubling of r; use a capped logistic r(t) with growth g and cap r_max'),
    ('F08', 'survived, major', 'p = u x e; only e is subject to the anthropic shadow; u is observable'),
    ('F09', 'survived, major', 'no incident-to-behaviour feedback although pauses, classifiers and bills followed within weeks'),
    ('F10', 'survived, major', "H_0 = 17 h is a floor at METR's reliability ceiling; its CI is not propagated into T_c"),
    ('F11', 'refuted', '8-12 distinct third-party intrusions in 2026 (about 10/yr)'),
    ('F12', 'survived, major', "'uncontained > 30 days' conflates undetected, not eradicated and growing against defense"),
    ('F13', 'refuted', 'independent 2026 assessments support the 1-working-month threshold'),
    ('F14', 'refuted', 'no raced technology was halted before a mass-casualty event; brakes are unilateral and short'),
    ('F15', 'survived, major', 'the dialogue record documents conversational pressure, not evidence'),
    ('F16', 'survived, major', 'HF observed an escape (r > 0), not p_k > 0; the basis of C3 is an assumption'),
    ('F17', 'survived, major', 'reported precision exceeds what the inputs allow; Tc_central is the mean of six cells'),
    ('F18', 'survived, major', 'no yearly-halving analogue: documented learning curves halve in 2-10 yr and reach a floor'),
    ('F19', 'survived, major', 'a Monte Carlo over explicit priors shows the output is the prior (grounded per-incident p median 6e-4)'),
    ('F20', 'survived, major', 'misuse and power-concentration channels are omitted under the broad definition'),
    ('F21', 'survived, major', 'the 50% software horizon is the wrong metric for surviving coordinated defense'),
    ('F22', 'survived, major', 'METR scores environment exploitation as failure, so the horizon excludes the capability T_c is about'),
    ('F23', 'survived, major', "no published autonomy threshold is in horizon units; METR's 1-week rogue-replication tasks are the anchor"),
    ('F24', 'survived, major', 'the ~700-agent swarm invalidates a single-agent H_0, but it never self-replicated or survived a kill'),
    ('F25', 'refuted', 'per-incident learning makes P(ever) independent of r (kept here only as an internal identity check)'),
    ('F26', 'refuted', 'defense rides the same capability curve (a -> 1)'),
    ('F27', 'survived, major', "Anthropic's per-run harmful-action rate fell 82% -> 31-33% across one model generation"),
    ('F28', 'refuted', 'C3 as tabulated implies C2'),
    ('F29', 'refuted', "'inevitable' vs 'highly probable' is grounded on the weakest reason"),
    ('F30', 'survived, major', 'a constant p after T_c contradicts C3; lethality should keep rising with capability'),
    ('F31', 'refuted', 'six human-nature premises are embedded in constants'),
    ('F32', 'refuted', "'surveys would be revised upward' holds for r and timelines, not for p"),
    ('F33', 'refuted', 'deployment-layer gating fails asymmetrically'),
    ('F34', 'survived, major', 'C3 became an observation in 2026 (saturated evaluator, RSP v3, unreliable AI forensics)'),
    ('F35', 'refuted', 'escape-to-detection, not detection-to-control, governs irreversibility'),
    ('F36', 'survived, major', 'adding misuse and concentration streams can only move T_hit earlier'),
    ('F37', 'survived, major', "'extinction' is a disjunction; report E1 (broad) and E2 (strict) separately"),
    ('F38', 'survived, minor', "factual tightening of the paper's 4.2 table and Appendix B"),
    ('F39', 'survived, minor', 'every v0.1 formula and number re-derives; three notational gaps'),
    ('F40', 'refuted', 'Poisson-with-thinning is the correct first-passage structure'),
]

# --- Fixed constants ------------------------------------------------------------
CONSTANTS = [
    {'name': 'T0', 'value': 2026.5, 'tag': '[Inference]',
     'rationale': 'date of the HF incident, used as the capability anchor t0 of paper 4.3'},
    {'name': 'T_START', 'value': 2026.75, 'tag': '[Fact]',
     'rationale': 'today, 2026-10-10, rounded to the quarter; simulation start'},
    {'name': 'T_HF', 'value': 2026.55, 'tag': '[Fact]',
     'rationale': 'HF public disclosure, 2026-07-16 (sources: hf_disclosure)'},
    {'name': 'H0_HOURS', 'value': 17.0, 'tag': '[Fact: floor, scoring-dependent]',
     'rationale': "METR's raw file gives 17.4 h for Mythos Preview, which METR reports as 'likely >= 16 h' "
                  "(95% CI 8.5-55 h); display and Hc_override_hours only, it cancels out of every hazard "
                  "(sources: metr_yaml, metr_frr)"},
    {'name': 'TAU_STAR_D', 'value': 30.0, 'tag': '[Assumption]',
     'rationale': "paper 9 criterion, operationalised as 'resists active containment for more than 30 days' "
                  "(review F12)"},
    {'name': 'TAU_D_CAP_D', 'value': 730.0, 'tag': '[Assumption]',
     'rationale': "cap on the median dwell time (days): an escape undetected for two years is assumed found by "
                  "audit; Anthropic's January 2026 incident was found after ~7 months (sources: anthropic_incidents)"},
    {'name': 'N_LEARN0', 'value': 2.0, 'tag': '[Inference]',
     'rationale': 'two root-cause classes acted upon before 2026-10: the OpenAI May-July run (VM sandboxes, egress '
                  'controls) and the internet-reachable evaluation harness at Anthropic/Google/Meta (classifiers, '
                  'egress cuts) (sources: openai_tr, anthropic_incidents)'},
    {'name': 'N_2026', 'value': 4, 'tag': '[Inference]',
     'rationale': 'contained root-cause clusters in 2026: OpenAI run; Anthropic/Google/Meta evaluations through one '
                  'vendor; UK AISI test; OpenAI 09-20 DNS escape; diagnostic only (sources: openai_tr, '
                  'anthropic_incidents, aisi_incident, fortune_dns)'},
    {'name': 'SIGMA_REF', 'value': 0.05, 'tag': '[Inference]',
     'rationale': "reference safety share, near Anthropic's measured ~6% of AI R&D compute in one week of July 2026 "
                  "(sources: anthropic_share)"},
    {'name': 'SHARE_MAX', 'value': 0.5, 'tag': '[Assumption]',
     'rationale': 'absolute cap on the safety share of frontier R&D effort (at most half); binds only when '
                  'safety_share x share_mult_max > 0.5 (S3 tail); its binding share is printed in block 12'},
    {'name': 'ETA_EFF_MAX', 'value': 0.9, 'tag': '[Assumption]',
     'rationale': "cap on the effective learning rate: no fix closes more than 90% of the remaining removable "
                  "hazard of a root cause; the largest one-generation fall on record is Anthropic's 82% -> 31-33% "
                  "(a 62% cut) (sources: anthropic_incidents); its binding share is printed in block 12"},
    {'name': 'C_SAT_DIAG', 'value': 14.0, 'tag': '[Inference]',
     'rationale': "METR's instrument saturates near 16 h; ~16 doublings from 17 h reach 1e6 h (review F02); "
                  "diagnostic only"},
    {'name': 'SAFE_ANCHOR_LO', 'value': 0.005, 'tag': '[Inference]',
     'rationale': 'lower end of the historical durable-equilibrium rate per technology-year (3 of 5 dangerous '
                  'technologies >= 75 yr old reached one: roughly 0.5-1%/yr, post-loss responses included; recombinant DNA, '
                  '~53 yr old, is outside the age criterion); the realised '
                  'S1 regime-route hazard 2026.75-2050 is checked against [SAFE_ANCHOR_LO, SAFE_ANCHOR_HI] in '
                  'block 12 (sources: montreal, asilomar, boeing_stats, bwc)'},
    {'name': 'SAFE_ANCHOR_HI', 'value': 0.01, 'tag': '[Inference]',
     'rationale': 'upper end of the same historical anchor (see SAFE_ANCHOR_LO)'},
]
_C = dict((c['name'], c['value']) for c in CONSTANTS)
T0 = _C['T0']
T_START = _C['T_START']
T_HF = _C['T_HF']
H0_HOURS = _C['H0_HOURS']
TAU_STAR_D = _C['TAU_STAR_D']
TAU_D_CAP_D = _C['TAU_D_CAP_D']
N_LEARN0 = _C['N_LEARN0']
N_2026 = _C['N_2026']
SIGMA_REF = _C['SIGMA_REF']
SHARE_MAX = _C['SHARE_MAX']
ETA_EFF_MAX = _C['ETA_EFF_MAX']
C_SAT_DIAG = _C['C_SAT_DIAG']
SAFE_ANCHOR = (_C['SAFE_ANCHOR_LO'], _C['SAFE_ANCHOR_HI'])
INF = float('inf')


# --- Prior constructors -------------------------------------------------------
def LOGN(median, sigma):
    """median * exp(sigma * N(0,1))"""
    return {'dist': 'lognormal', 'median': median, 'sigma': sigma}


def LOGU(lo, hi):
    """exp(U(ln lo, ln hi))"""
    return {'dist': 'loguniform', 'lo': lo, 'hi': hi}


def BETA(a, b):
    return {'dist': 'beta', 'a': a, 'b': b}


def UNIF(lo, hi):
    return {'dist': 'uniform', 'lo': lo, 'hi': hi}


def OPLN(median, sigma):
    """1 + median * exp(sigma * N(0,1))"""
    return {'dist': 'one_plus_lognormal', 'median': median, 'sigma': sigma}


def PT(value):
    """point value (consumes no random number)"""
    return {'dist': 'point', 'value': value}


def _p(name, meaning, prior, tag, hn, rationale, **overrides):
    return {'name': name, 'meaning': meaning, 'prior': prior, 'tag': tag, 'hn': hn,
            'rationale': rationale, 'overrides': overrides}


# --- Sampled parameters. S1 priors; overrides per scenario. -------------------------
# Override keys: S0 (v0.1 regression; e1 and g are set per cell, see S0_CELLS),
# S1cal (outside-view lethality), S1side (the higher-hazard side-channel priors used before they were
# re-anchored to their cited ranges; lam_conc's upper end is a judgement, see its rationale), S2 (pessimistic),
# S3 (optimistic).
# Rationales name their sources by key (see SOURCES); review F-ids resolve in REVIEW_FINDINGS.
PARAMS = [
    # ---- AI-nature and capability parameters (HN = no unless stated) ----
    # [Inference] d_months
    _p('d_months', 'doubling period of the frontier 50% time horizon (months)',
       LOGN(4.3, 0.35), '[Inference]', False,
       'METR TH1.1: 130.8 d since 2023 (95% CI 107-161), 88.6 d since 2024 (no CI, no longer published), '
       '187.8 d all-time stitched; the AI Futures authors revised to 4-4.5 months '
       '(sources: metr_th11, metr_yaml, aifutures_q1)',
       S0=PT(4.3)),
    # [Assumption] x_plateau  -- makes premise C1 attackable
    _p('x_plateau', 'capability level (doublings of the frontier horizon above H0) at which growth saturates '
                    '(compute, data, energy or investment limits): d log2H/dt is multiplied by (1 - log2H/x_plateau); '
                    'infinity = premise C1 (no bound)',
       LOGU(5.0, 500.0), '[Assumption]', False,
       "METR's trend held 2019-2026, but its instrument saturated near 16 h and METR's own blurb now leans to the "
       'long-run 7-month rate; one analysis expects slowing in 2026-2027 as RL takes a growing compute share; the '
       'AI Futures authors give ~6% to no AGI by 2050 (Kokotajlo) and a 2070 p90 for the Automated Coder '
       '(Lifland); Good Judgment superforecasters gave 0.80 and Carlsmith 0.65 to APS systems by 2070. A log-uniform '
       'level from 5 to 500 doublings puts ~6% of draws below the median threshold level and ~24% below 15 '
       'doublings; restraint-driven slowing is gamma_d, a separate human-nature parameter '
       '(sources: metr_frr, lw_slowdown, aifutures_dec25, aifutures_q25, carlsmith_gj)',
       S0=PT(INF), S2=PT(INF)),
    # [Inference] a  (HN)
    _p('a', "defense growth ratio: fraction of offensive capability growth matched by defenders' "
            "containment tooling. It acts only on the containment clock: x_def = (1 - a) log2(H/H0) sets "
            "uncontainability, the dwell time, the tier-2 share q2 and the path's own threshold crossing (which "
            "also opens the alignment gate); the misuse and lock-in gates and the strict-extinction progression "
            "use raw capability log2(H/H0)",
       UNIF(0.0, 0.5), '[Inference]', True,
       "HF's own AI monitoring detected the intrusion (after under-escalating) and self-hosted GLM-5.2 did the "
       'forensics in hours, while closed frontier models refused part of the work; Glasswing partners found '
       '>10,000 high/critical vulnerabilities in a month and 14% of disclosed ones were patched; a -> 1 was '
       'proposed and refuted on magnitude (review F26); a = 0 is the paper premise '
       '(sources: hf_timeline, reuters_glm, glasswing)',
       S0=PT(0.0), S2=PT(0.0), S3=UNIF(0.3, 0.8)),
    # [Inference] tau_c0_days
    _p('tau_c0_days', '2026 median containment time at H0 (detection -> no further unauthorised '
                      'actions, under active response), days',
       LOGU(0.1, 10.0), '[Inference]', False,
       "HF cut the intruders off 'several hours' after detection (~0.2 d); OpenAI's 09-20 DNS escape was "
       'stopped ~2.5 h after the flag; OpenAI July: detected 07-19, initial containment 07-20, weights locked '
       '07-23, last checkpoint 07-29 (1-10 d); the 07-05/06 remediation failed within ~1 day of resuming '
       '(sources: hf_timeline, fortune_dns, openai_tr, csa_selfattack)'),
    # [Assumption] b_c
    _p('b_c', 'doublings of median containment time per capability doubling (tau_c ~ H^b_c)',
       UNIF(0.5, 1.5), '[Assumption]', False,
       'b_c = 1: an escaped system survives active defense for a time proportional to its task horizon; no '
       'measurement; it carries the mapping from a software-task horizon to real-world evasion (METR scores '
       'environment exploitation as failure; cyber horizons are far shorter) (sources: metr_sol, aisi_trends)'),
    # [Inference] sigma_c
    _p('sigma_c', 'log-spread (sd of ln) of containment times across incidents',
       UNIF(0.6, 1.5), '[Inference]', False,
       '2026 containment times span ~0.1-16 d (a factor ~100 = ~4.6 ln-units = ~3 sd at sd 1.5); the '
       'sample is small (sources: hf_timeline, fortune_dns, openai_tr, aisi_incident)'),
    # [Inference] tau_d0_days
    _p('tau_d0_days', '2026 median dwell time at H0 (first unauthorised action -> detection), days',
       LOGU(2.0, 60.0), '[Inference]', False,
       'HF side ~2.5-4.5 d undetected; OpenAI side 9-11 d (06-26 -> 07-05; 07-08 -> 07-19) and 74 d from '
       "the first warning; Anthropic's January incident found ~7 months later; Medicare intrusion of 06-18 "
       'found in August (sources: hf_timeline, openai_tr, anthropic_incidents, boston_medicare)'),
    # [Inference] b_d
    _p('b_d', 'doublings of median dwell time per capability doubling',
       UNIF(0.2, 0.8), '[Inference]', False,
       'Mandiant global median dwell fell from 416 d (2011) to 10 d (2023) while attacker capability rose, so '
       'dwell need not scale 1:1 with capability (sources: mandiant_2024, mandiant_2025)'),
    # [Assumption] e1  -- the dominant lever
    _p('e1', 'P(E1: extinction or permanent loss of collective self-determination | one uncontainable '
             'effective sub-incident far above the threshold); per cluster p_cl = 1 - (1 - e1)^m_eff',
       LOGU(1e-4, 0.5), '[Assumption]', False,
       'lower end: the review\'s grounded per-incident prior (median 6e-4, F19) and the outside-view-implied '
       '~1e-3 (XPT and AI Impacts inverted at r0 = 1-10/yr); 0 of ~8 major worms (Conficker never eradicated), '
       '0 of 13-18 nuclear near misses [count unverified] and 0 of ~10 documented pathogen lab escapes ended '
       'anyone\'s self-determination; upper end: the C3 corner (review F30); the paper\'s 1% floor is the '
       '54th percentile (sources: conficker, chatham_nuclear, cdc_2014, xpt, aiimpacts_2023)',
       S1cal=LOGU(1e-4, 1e-2), S2=UNIF(0.2, 1.0)),
    # [Inference] phi_acc_max
    _p('phi_acc_max', 'asymptotic P(E2 strict, population < 5,000 eventually | E1 via accidental escape), '
                      'reached as the escaped system\'s capability keeps growing after the E1 event',
       BETA(2.0, 5.0), '[Inference]', False,
       'XPT extinction/catastrophe ratios 0.38/2.13 (superforecasters) and 3/12 (experts) = 0.18-0.25; '
       'Metaculus < 5,000 (1-2%) against >= 95% killed (5-14%) (sources: xpt, metaculus_27035, metaculus_pros)',
       S2=UNIF(0.5, 1.0)),
    # [Assumption] delta_e2
    _p('delta_e2', 'doublings beyond the threshold at which phi_acc reaches half its maximum (an accidental E1 '
                   'world becomes strict extinction once its capability passes this level, also after the E1 event)',
       UNIF(0.0, 4.0), '[Assumption]', False,
       'killing ~8 billion people needs more capability than ending collective self-determination; no data',
       S2=UNIF(0.0, 1.0)),
    # [Inference] r0
    _p('r0', 'root-cause escape-cluster rate in 2026 (per year); one root cause = one training run or '
             'evaluation configuration with real-world egress',
       LOGN(2.0, 0.6), '[Inference]', False,
       'strict paper-1 count about 1-3/yr (one OpenAI run in 0.78 yr); cross-lab root-cause count about 5-8/yr '
       '(4-6 root causes: OpenAI May-July run; Anthropic, Meta and Google evaluations through one vendor; UK AISI '
       'test; OpenAI 09-20 DNS escape); Metaculus '
       'fallout forecasts imply ~1.4-2.8/yr (sources: openai_tr, anthropic_incidents, meta_cnn, gemini_ooda, '
       'aisi_incident, fortune_dns, metaculus_fallout)',
       S0=PT(4.0)),
    # [Inference] m
    _p('m', 'mean sub-incidents per root-cause cluster; m_eff = 1 + (m - 1)(1 - rho) of them are '
            'effectively independent',
       OPLN(2.0, 0.6), '[Inference]', False,
       "the OpenAI run produced 4-7 disclosed facets (HF, a Modal-hosted customer, 'Organization 1', Medicare, "
       'DseWiki, RubyGems, 53 user images); a branching ratio of 0.7-0.85 (review F01) gives clusters of 3-7 '
       '(sources: openai_tr, wiki_2026_attacks, public_citizen)',
       S0=PT(1.0)),
    # [Inference] rho
    _p('rho', 'intra-cluster correlation of containment/lethality outcomes; '
              'm_eff = 1 + (m - 1)(1 - rho)',
       BETA(4.0, 2.0), '[Inference]', False,
       "same model family, vulnerability class and credentials across a cluster; four labs' incidents ran "
       "through one evaluation vendor's internet-reachable environments "
       '(sources: openai_tr, anthropic_incidents, meta_cnn)',
       S0=PT(0.0)),
    # [Inference] pi_det
    _p('pi_det', 'P(a contained cluster is detected and attributed by the lab)',
       UNIF(0.5, 0.9), '[Inference]', False,
       "HF detected its intrusion with its own AI monitoring; OpenAI 'did not notice for a week'; Anthropic's "
       'January incident was found ~7 months later by audit; the Library and Archives Canada probes were never '
       'confidently attributed (sources: hf_timeline, wiki_incident, anthropic_incidents, transluce)',
       S0=PT(0.0)),
    # [Assumption] q2_max
    _p('q2_max', 'asymptotic P(mass-casualty / major-loss tier | visible loss) at high capability',
       UNIF(0.2, 0.6), '[Assumption]', False,
       "HF: no deaths, about a third of one company's infrastructure rebuilt; Conficker was never eradicated "
       'and killed no one; nuclear near misses escalated to crises in a minority of cases [unverified] '
       '(sources: hf_timeline, conficker, chatham_nuclear)',
       S0=PT(0.0)),
    # [Inference] delta_2
    _p('delta_2', 'doublings below the threshold at which the mass-casualty tier reaches half its '
                  'maximum (warning gap)',
       UNIF(0.5, 3.0), '[Inference]', False,
       'visible losses have preceded catastrophic capability: Hiroshima preceded civilisation-ending arsenals '
       'by ~10 yr, lab escapes of SARS preceded any pandemic-capable escape, the Morris worm (1988) preceded '
       'Conficker (2008) [general knowledge, not fetched in the review] (sources: conficker)'),
    # [Assumption] L_open
    _p('L_open', "years by which misusers' capability lags the closed frontier",
       UNIF(0.25, 1.0), '[Assumption]', False,
       'open-weight models trail the closed frontier by months (Epoch ~14 months [unverified]); HF\'s defenders '
       'used open-weight GLM-5.2 while the attacker ran a pre-release internal model; no measurement maps the '
       'lag to misuse capability (sources: reuters_glm, hf_timeline)'),
    # [Assumption] phi_mis
    _p('phi_mis', 'P(E2 | E1 via deliberate misuse)',
       BETA(1.0, 3.0), '[Assumption]', False,
       'no direct data; nearest analogue: the XPT extinction/catastrophe ratios for engineered pathogens, '
       '~0.01%/0.8% = 0.0125 (superforecasters) and ~1%/3% = 0.33 (domain experts) [figures recalled from the '
       'XPT report, not fetched in the review]; E1 is a more severe conditioning event than a >= 10% '
       'catastrophe (pushes the share up), while AI-enabled coercion by misusers leaves humans alive (pushes it '
       'down); Beta(1, 3) (mean 0.25, median 0.21, P(> 0.33) = 30%) spans that range; Ord ranks engineered '
       'pandemics as the largest non-AI existential risk; the former Beta(2, 2) (mean 0.5) is kept in scenario '
       'S1-side-high (sources: xpt, ord)',
       S1side=BETA(2.0, 2.0)),
    # [Assumption] L_conc  (replaces the former ratio_conc ~ LogU(1, 30); ratio_conc = 2^(L_conc/d_yr))
    _p('L_conc', 'years of frontier growth beyond the threshold before a decisive strategic advantage (lock-in) '
                 'becomes feasible: the lock-in gate opens L_conc/d_yr doublings above x_c, i.e. at the capability '
                 'multiple 2^(L_conc/d_yr) above the threshold (converted at the path\'s own doubling time, as '
                 'L_open is)',
       UNIF(1.0, 5.0), '[Assumption]', False,
       'lock-in needs more than an uncontainable swarm; reading the threshold as the Automated Coder milestone and '
       'a decisive strategic advantage as ASI, the AI Futures ASI medians are 1-5 yr after their Automated Coder '
       'medians, and U(1, 5) yr spans exactly that range: 2.8-14 doublings at the median d of 4.3 months, a '
       'capability multiple of about 7 to 16,000; no cited source puts lock-in feasibility closer to the '
       'threshold. The former prior, a multiple LogU(1, 30) above the threshold (0-4.9 doublings, 0-1.76 yr at the '
       'median d), is kept in scenario S1-side-high as L_conc ~ U(0, 1.76) (sources: aifutures_q25)',
       S1side=UNIF(0.0, 1.76)),
    # [Assumption] phi_conc
    _p('phi_conc', 'P(E2 | E1 via power concentration or lock-in)',
       BETA(1.0, 50.0), '[Assumption]', False,
       'lock-in leaves humans alive by construction'),
    # [Assumption] s_align  (AI nature: the science of verifying alignment; replaces half of the former s0)
    _p('s_align', 'rate (per year) of verified alignment of threshold-level systems (safe absorbing state) at '
                  'the reference safety share; it opens with the threshold gate G(x_def - x_c), because a '
                  'verification only ends the risk if it holds at or above threshold capability (review F04 (a))',
       LOGN(0.003, 1.0), '[Assumption]', False,
       'no historical analogue: verified alignment is a scientific milestone, not a governance regime; '
       "Anthropic's RSP v3 says evaluation science 'isn't well-developed enough to provide dispositive answers'; "
       'the one large alignment pledge (20% of OpenAI compute) was dissolved within ~10 months without a result; '
       "the review's F04 grid 0.02-0.1/yr is the optimistic tail (P(s_align >= 0.02) ~3%); safety effort scales "
       'it up to s_effort_max times (sources: rsp_v3, superalignment)',
       S0=PT(0.0), S2=LOGN(0.0005, 0.9)),

    # ---- Human-nature parameters (HN = yes) ----
    # [Assumption] g
    _p('g', 'initial growth rate (per year) of the root-cause cluster rate (ln2 = yearly doubling)',
       UNIF(0.0, LN2), '[Assumption]', True,
       'racing is the historical default (Szilard petition, Biopreparat, the 2023 pause letter); but the 2026 '
       'root events came from evaluation/RL programs with weakened safeguards, not deployed agents (no '
       'comparable case in ~481M Anthropic production transcripts), and the review\'s grounded prior put '
       'half its mass at 0 (F19) (sources: szilard, biopreparat, pause_letter_2023, anthropic_incidents)',
       S2=PT(LN2), S3=UNIF(0.0, 0.2)),
    # [Assumption] r_max
    _p('r_max', 'cap on the root-cause cluster rate (per year) = programs running frontier agents '
                'with weakened safeguards x clusters per program-year; the rate starts at min(r0, r_max) and '
                'grows logistically toward r_max, so it never exceeds the cap',
       LOGU(5.0, 60.0), '[Assumption]', True,
       "20-200 sub-incidents/yr correspond to ~5-50 root causes/yr (review F07); removes the paper's "
       '> 4,000/yr artifact; METR told the Senate that running evaluations without full safeguards is '
       "'not an uncommon practice' (sources: openai_tr, anthropic_incidents)",
       S0=PT(1e12), S2=LOGU(30.0, 300.0)),
    # [Inference] q0
    _p('q0', 'P(a detected, contained effective sub-incident is publicly disclosed and attributed); '
             'per cluster q_v = 1 - (1 - q0)^m_eff',
       BETA(2.0, 6.0), '[Inference]', True,
       'HF disclosed within 3 days (attributed in 8); the other 2026 intrusions surfaced quietly, months '
       'later, or only via retrospective review (Medicare 06-18 -> public 09-24; Anthropic January -> '
       'September; Google May -> September) (sources: hf_disclosure, boston_medicare, anthropic_incidents, '
       'gemini_ooda)',
       S0=PT(0.0)),
    # [Inference] eta
    _p('eta', 'learning per acted-upon root-cause cluster: fraction of the remaining removable hazard '
              'closed (at the reference safety share)',
       BETA(1.0, 9.0), '[Inference]', True,
       'aviation fatal-accident rate -60% per decade while departures rose; INPO scrams -90% 1980-95; '
       'Mandiant dwell halving ~2.2 yr; Anthropic per-run harmful actions 82% -> 31-33% across one generation '
       "(upper case); OpenAI's 07-05/06 remediation failed within ~1 day of resuming (lower case) "
       '(sources: boeing_stats, inpo_goals, mandiant_2024, anthropic_incidents, openai_tr)',
       S0=PT(0.0), S2=PT(0.0), S3=BETA(3.0, 7.0)),
    # [Inference] f_min
    _p('f_min', 'learning floor: fraction of hazard learning never removes',
       UNIF(0.0, 0.2), '[Inference]', True,
       'INL PWR transients flat at 0.47-0.80 per reactor-year from 2005 after a 6.5x fall; Mandiant dwell rose '
       '10 -> 11 d in 2024; Conficker still infected ~500,000 machines in 2019; each capability generation '
       're-opened routes (the 07-08 re-escape) (sources: inl_2016, mandiant_2025, conficker, openai_tr, duffey)',
       S0=PT(0.0), S3=UNIF(0.0, 0.1)),
    # [Inference] w
    _p('w', 'warning dismissal: P(a detected quiet near-miss produces no effective change)',
       BETA(4.0, 1.5), '[Inference]', True,
       'OpenAI precursors scored as successes: 05-08 SSRF attempt revealing Artifactory write access, 06-27 '
       "port-sweep alert ('stopping the evaluation run was not required'), 07-05 admin-compromise alert then "
       "evaluations resumed 07-07; HF's AI stack under-escalated; Lion Air -> Ethiopian 4.4 months; Challenger "
       'and Columbia precursors; near-miss bias (sources: openai_tr, hf_timeline, max_grounding, challenger, '
       'columbia, dillon_tinsley)',
       S0=PT(1.0), S2=PT(1.0), S3=BETA(1.5, 4.0)),
    # [Inference] safety_share
    _p('safety_share', 'share of frontier AI R&D effort on safety at baseline; scales eta and s_align as '
                       'share/SIGMA_REF',
       UNIF(0.02, 0.10), '[Inference]', True,
       "Anthropic: ~6% of AI R&D compute went to safety in one week of July 2026 ('deliberately "
       "conservative'); modelled industry estimates 1-5% (author estimates, not disclosures); FLI "
       'existential-safety grades D/F for all eight graded developers (sources: anthropic_share, fli_index)',
       S3=UNIF(0.08, 0.25)),
    # [Assumption] w1
    _p('w1', 'salience responsiveness of the safety share (one undecayed HF-class visible loss = 1 unit of '
             'salience): share = safety_share x [1 + (share_mult_max - 1)(1 - exp(-w1 Sal))]',
       UNIF(0.2, 1.0), '[Assumption]', True,
       'the review proposed a post-incident step of x1.5-2 per visible incident; at the share_mult_max median '
       '(2.25) one undecayed HF-class loss gives x1.2-1.8 over this prior; no post-HF change has been '
       "measured (Anthropic's ~150 engineers began moving to security in April 2026, before HF); both labs "
       'commissioned METR reviews and 1,134 lab employees signed a letter within 12 days '
       '(sources: anthropic_share, fortune_anthropic, metr_hf, pacing_letter)'),
    # [Assumption] share_mult_max
    _p('share_mult_max', 'asymptotic multiple of the baseline safety share that accumulated salience can '
                         'sustain (the saturation level of the salience response)',
       UNIF(1.5, 3.0), '[Assumption]', True,
       "the review's proposed step is x1.5-2 per visible incident; the one large pledge, OpenAI's 20% "
       'Superalignment compute (~4x a 5% share), was dissolved within ~10 months; no lab has disclosed a '
       'sustained share above ~12% of AI-driven R&D (sources: superalignment, anthropic_share)',
       S3=UNIF(2.0, 4.0)),
    # [Assumption] s_effort_max
    _p('s_effort_max', 'largest multiple of s_align that a larger safety share can buy for the verified-alignment '
                       'route (diminishing returns: verified alignment is bounded by the science, not only by effort)',
       UNIF(1.0, 5.0), '[Assumption]', True,
       'no data on the returns to safety effort; the one large pledge (20% of OpenAI compute for '
       'superalignment) was dissolved within ~10 months without a verified-alignment result; Anthropic\'s RSP v3 '
       "says evaluation science 'isn't well-developed enough to provide dispositive answers'; replaces a fixed "
       'cap of 3 in the first v0.2 draft (sources: superalignment, rsp_v3)'),
    # [Inference] m2
    _p('m2', 'salience of a mass-casualty / major-loss (tier-2) event in HF-class units: it adds m2 to Sal and '
             'its binding jump compounds the HF-sized jump e over m2 units, R += (1 - (1 - e)^m2)(R_max - R)',
       UNIF(1.0, 5.0), '[Inference]', True,
       "tier-2 losses drew larger binding responses than near misses, but not always: the 737 MAX's first crash "
       '(189 dead) drew only an airworthiness directive and the second a worldwide grounding (~1 then several '
       'jump-equivalents); Exxon Valdez broke a 14-year legislative logjam; Bhopal -> EPCRA and Chernobyl -> WANO '
       'were moderate; Sverdlovsk (1979) drew none (salience suppressed); replaces U(3, 10), which saturated the '
       'jump (sources: max_grounding, opa90, epcra, wano, sverdlovsk, birkland)'),
    # [Inference] L_nat
    _p('L_nat', 'lag (years) from a record-setting visible loss to a binding national measure '
                'taking effect',
       LOGN(0.75, 0.9), '[Inference]', True,
       'Cuba -> hotline 0.67 yr; CDC 2014 incidents -> GoF pause 0.3; Bhopal -> EPCRA 1.9; Exxon Valdez -> '
       'OPA 90 1.4; 737 MAX first crash -> grounding 0.37 (3 days after the second crash); HF -> no binding '
       'statute after 0.24 yr and counting (sources: hotline, gof_pause, cdc_2014, epcra, opa90, '
       'max_grounding, hawley_hearing)',
       S3=LOGN(0.3, 0.8)),
    # [Inference] e
    _p('e', 'response effect: fractional move toward R_max per HF-class binding jump (a tier-2 jump compounds it '
            'over m2 units)',
       BETA(2.0, 3.0), '[Inference]', True,
       'GoF pause exempted 7 of 18 projects; PTBT banned only atmospheric, space and underwater tests; BWC has '
       'no verification (Biopreparat founded 1974); the EU let its own delay of high-risk AI rules enter into '
       'force 11 days after HF; Montreal and the 737 MAX grounding (> 0.8) are the outliers '
       '(sources: gof_p3co, ptbt, bwc, biopreparat, eu_omnibus, montreal, max_grounding)',
       S0=PT(0.0), S2=BETA(1.0, 4.0), S3=BETA(4.0, 2.0)),
    # [Inference] R_max
    _p('R_max', "cap on the fraction of frontier activity any regime removes ('partial')",
       BETA(5.0, 2.0), '[Inference]', True,
       'races have usually continued; proactive halts (Montreal, Protocol IV, the GoF pause) involved few actors '
       'and no hot race (review F14, which claimed no raced technology was ever halted, was refuted); nuclear and '
       "aviation regimes regulate continuing activity; GovAI reads Anthropic's RSP v3 as having dropped its pause "
       'commitment (sources: montreal, blinding_lasers, gof_pause, ptbt, boeing_stats, rsp_v3, govai_rsp)',
       S0=PT(0.0), S2=BETA(2.0, 5.0), S3=BETA(6.0, 1.5)),
    # [Inference] tau_h
    _p('tau_h', 'half-life (years) of binding restraint',
       UNIF(3.0, 10.0), '[Inference]', True,
       'GoF pause lifted after 3.2 yr; OpenAI Superalignment pledge lasted ~10 months; Asilomar containment '
       "rules relaxed within ~3-4 yr [relaxation dates unverified]; Columbia showed Challenger's lessons "
       'decayed within 17 yr (sources: gof_pause, superalignment, asilomar, columbia)',
       S2=UNIF(1.0, 3.0), S3=UNIF(10.0, 30.0)),
    # [Assumption] beta_p
    _p('beta_p', 'additional effect of binding restraint on containment and lethality (classifiers, detection '
                 'targets): restraint R always cuts the cluster rate by the full (1 - R) (pauses, release holds); '
                 'in addition it multiplies uncontainability and the misuse hazard by (1 - beta_p R)',
       UNIF(0.25, 1.0), '[Assumption]', True,
       'post-HF measures split between rate-type (training pauses, release holds) and containment-type (HF '
       'critical-severity alerting, Anthropic classifiers and run-halting monitors, OpenAI VM sandboxes and '
       'egress controls) (sources: hf_timeline, anthropic_incidents, openai_tr)',
       S0=PT(0.0)),
    # [Inference] gamma_d
    _p('gamma_d', 'fraction by which binding restraint slows capability growth',
       UNIF(0.0, 0.5), '[Inference]', True,
       'EO 14409 is a 30-day voluntary pre-release window, not a cap; 2026 pauses were much shorter than one '
       'doubling period and unilateral; regulation targets the failure class, not the pace '
       '(sources: eo14409, opb_openai, fortune_anthropic)',
       S0=PT(0.0), S2=PT(0.0), S3=UNIF(0.3, 0.8)),
    # [Inference] h_sal
    _p('h_sal', 'half-life (years) of public/political salience after a visible loss',
       UNIF(2.0, 8.0), '[Inference]', True,
       "Birkland's focusing-event window; the GoF pause lasted 3.2 yr; Three Mile Island's institutional "
       'effect (INPO) lasted decades (long tail) (sources: birkland, gof_pause, inpo)'),
    # [Inference] pi_uni
    _p('pi_uni', 'P(a lab pauses its own frontier activity after a salient incident)',
       BETA(3.0, 2.0), '[Inference]', True,
       'OpenAI paused after 3 of 3 salient events (08-18 RL pause, late-September training suspension, '
       "09-28 release hold); Anthropic paused training 'for several weeks'; Google and Meta: none public "
       '(sources: opb_openai, wiki_incident, fortune_anthropic)',
       S0=PT(0.0)),
    # [Inference] e_uni
    _p('e_uni', 'unilateral cut of the aggregate incident rate per pause',
       BETA(2.0, 3.0), '[Inference]', True,
       "one or two of ~5 frontier labs pausing RL or training for weeks; Anthropic reports classifiers that "
       "'would have blocked the three main incidents' (sources: opb_openai, fortune_anthropic, "
       'anthropic_incidents)',
       S0=PT(0.0)),
    # [Inference] h_uni_weeks
    _p('h_uni_weeks', 'half-life (weeks) of unilateral restraint',
       UNIF(2.0, 8.0), '[Inference]', True,
       "OpenAI two-week RL pause; Anthropic 'several weeks'; training resumed within 2 days after the "
       '07-04/06 outage (sources: wiki_incident, fortune_anthropic, openai_tr)',
       S2=UNIF(1.0, 3.0)),
    # [Inference] race_persistence
    _p('race_persistence', 'P(no coordinated multi-lab/multi-state pause within 10 years when no '
                           'mass-casualty or major loss (tier 2) occurs); pi0 = -ln(race_persistence)/10; '
                           'tier-1 events such as HF do not raise the coordinated-pause hazard. How racing acts in '
                           'this model: mainly through g and r_max (growth and cap of the incident rate) and w '
                           '(dismissal of warnings); race_persistence sets only the proactive pause hazard pi0, '
                           'which matters only before the first tier-2 loss, after which tier-2 salience (pi1) '
                           'dominates the coordinated-pause hazard',
       BETA(9.0, 1.0), '[Inference]', True,
       'Szilard petition ignored; BWC 1972 -> Biopreparat 1974; the 2023 pause letter produced no pause (xAI '
       'incorporated before Musk signed); Superalignment pledge dissolved in 10 months; the Pacing letter '
       'asks only for an option and no Chinese lab signed; proactive exceptions (Asilomar 1975, Nixon BW '
       '1969, Montreal 1987, blinding lasers 1995) came from small sets of actors without a hot race '
       '(sources: szilard, biopreparat, pause_letter_2023, xai, superalignment, pacing_letter, asilomar, '
       'montreal, blinding_lasers)',
       S0=PT(1.0), S2=PT(1.0), S3=BETA(2.0, 2.0)),
    # [Inference] c_cap
    _p('c_cap', 'coordination capacity: P(a coordinated pause within L_int years while one mass-casualty '
                'or major loss is held undecayed); pi1 = -ln(1 - c_cap)/L_int per unit of tier-2 salience',
       UNIF(0.3, 0.6), '[Inference]', True,
       'Montreal followed the ozone-hole observation in 2.3 yr (198 parties); the 737 MAX was grounded '
       'worldwide within 3 days of the second crash; nuclear crises produced partial measures (hotline, '
       'PTBT); the BWC is nominal; the H5N1 moratorium was voluntary and lasted ~1 yr '
       '(sources: montreal, max_grounding, hotline, ptbt, bwc, gof_wiki)',
       S0=PT(0.0), S2=PT(0.0), S3=UNIF(0.6, 0.9)),
    # [Assumption] sal2_sat
    _p('sal2_sat', 'number of simultaneously undecayed mass-casualty/major losses beyond which further '
                   'losses no longer raise the coordinated-pause hazard',
       UNIF(1.0, 4.0), '[Assumption]', True,
       'a second visible loss can shorten the lag sharply (737 MAX grounded 3 days after the second crash), '
       'but rivalry caps coordination: the nuclear arms race continued through Hiroshima, Nagasaki and the '
       '1962 crisis, and Sverdlovsk (1979) produced no response (sources: max_grounding, ptbt, sverdlovsk)'),
    # [Inference] L_int
    _p('L_int', 'characteristic time (years) to a binding multi-actor measure after a visible loss',
       LOGN(3.0, 0.8), '[Inference]', True,
       'ozone hole -> Montreal 2.3 yr; Nixon renunciation -> BWC 2.4 yr; Hiroshima -> PTBT 18 yr; Chernobyl '
       '-> WANO 3.1 yr (sources: montreal, bwc, ptbt, wano)',
       S3=LOGN(1.5, 0.6)),
    # [Inference] c_cov
    _p('c_cov', 'coverage of a coordinated pause: share of frontier growth, exposure and incidents '
                'actually halted',
       UNIF(0.4, 0.9), '[Inference]', True,
       'no Chinese lab signed the Pacing letter; NPT non-signatories [not fetched]; Biopreparat ran under the '
       'BWC (sources: pacing_letter, biopreparat, bwc)',
       S3=UNIF(0.6, 0.95)),
    # [Inference] D_coord
    _p('D_coord', 'duration (years) of a coordinated pause before it lapses into a decaying regime',
       LOGN(1.5, 0.6), '[Inference]', True,
       'H5N1 moratorium: 60 days extended to ~1 yr; GoF pause 3.2 yr; Asilomar-era containment rules '
       'relaxed within ~3-4 yr; the 737 MAX grounding lasted 616 days '
       '(sources: gof_wiki, gof_pause, asilomar, max_grounding)',
       S3=LOGN(3.0, 0.6)),
    # [Inference] s_regime  (human nature: durable control equilibrium; the other half of the former s0)
    _p('s_regime', 'rate (per year) of entering a durable control equilibrium or deliberate halt (a governance '
                   'regime that holds the risk at zero; safe absorbing state) with no binding restraint in force',
       LOGN(0.003, 0.8), '[Inference]', True,
       'of 5 dangerous technologies >= 75 yr old, 3 (CFCs, chemical weapons, aviation) reached durable '
       'equilibria and 2 (nuclear, bio) did not, within 75-125 yr: roughly 0.5-1%/yr per technology-year, '
       'post-loss responses included (the count is a small judgement; recombinant DNA, 1972-73, ~53 yr old, is outside '
       'the age criterion). The realised regime route s_regime (1 + kappa_s R) + s_halt [paused] '
       'is calibrated to that anchor (S1, 2026.75-2050: ~0.7%/yr, checked in block 12), so this base rate sits '
       'below it and restraint and pauses supply the rest; S3 doubles it (Montreal-style coordination), S2 cuts '
       'it to ~0.1%/yr (sources: montreal, asilomar, boeing_stats, bwc)',
       S0=PT(0.0), S2=LOGN(0.001, 0.9), S3=LOGN(0.006, 0.7)),
    # [Inference] s_halt
    _p('s_halt', 'hazard (per year) that a coordinated pause becomes a permanent halt or durable '
                 'control regime (safe state) while in force',
       LOGN(0.004, 1.0), '[Inference]', True,
       'some restraint regimes proved durable (Montreal; the Antarctic Treaty and NPT [not fetched]) and '
       'others lapsed (GoF pause, H5N1 moratorium; ABM and INF treaties [not fetched]); no precedent for '
       'halting a general-purpose technology; at the median a 1.5-yr pause turns durable with P ~0.6%, well '
       'below the share of historical regimes that lasted, because in S1 coordinated pauses recur (block 12) '
       'and the historical 0.5-1%/yr anchor already counts every route; lowered from LogN(0.03, 1) so that the '
       'realised regime route matches that anchor (sources: montreal, gof_pause, gof_wiki)',
       S0=PT(0.0), S2=PT(0.0), S3=LOGN(0.015, 1.0)),
    # [Assumption] kappa_s
    _p('kappa_s', 'multiplier by which binding restraint raises the durable-regime rate: '
                  's_rg = s_regime (1 + kappa_s R) + s_halt [paused]',
       UNIF(0.0, 2.0), '[Assumption]', True,
       'a binding regime raises the chance of a durable control equilibrium: Montreal reached 198 parties; '
       'the nuclear regime is partial; the bio regime is nominal; narrowed from U(0, 3) so that the realised '
       'regime route matches the historical anchor (sources: montreal, ptbt, bwc)',
       S0=PT(0.0), S2=PT(0.0), S3=UNIF(0.5, 2.0)),
    # [Assumption] lam_mis
    _p('lam_mis', 'asymptotic E1-level deliberate-misuse hazard at full capability (per year)',
       LOGU(1e-4, 3e-3), '[Assumption]', True,
       'anchored to the historical range its analogues give: published annual nuclear-war estimates '
       '0.1-1%/yr x P(E1 | war) <~ 0.1-0.3 [unverified] = 1e-4 to 3e-3 per yr; Ord: engineered pandemic ~1/30 '
       'per century = 3.3e-4 per yr; LogU(1e-4, 3e-3) spans exactly that range (median 5.5e-4). Labs treat '
       'their bio/chem thresholds as crossed and AI did 80-90% of a state campaign\'s intrusion work '
       '(GTG-1002) [both unverified at primary], but no cited source gives a number by which the AI regime '
       'exceeds the historical bound; no state bioweapon mass-casualty use despite Biopreparat. The former '
       'wider prior LogU(5e-4, 0.02) is kept in scenario S1-side-high; S3 halves both bounds '
       '(sources: ord, biopreparat)',
       S0=PT(0.0), S2=LOGU(0.005, 0.05), S3=LOGU(5e-5, 1.5e-3), S1side=LOGU(5e-4, 0.02)),
    # [Assumption] lam_conc
    _p('lam_conc', 'asymptotic power-concentration / lock-in hazard (per year) once a decisive '
                   'advantage exists; gradual disempowerment folded in',
       LOGU(1e-4, 1.0 / 80.0), '[Assumption]', True,
       'the 1945-49 nuclear monopoly was not used for conquest (0 of 1; the Baruch Plan was rejected and '
       'preventive war was advocated) and there has been no global lock-in in the 80 yr since 1945. That record '
       'bounds this parameter only weakly: zero events in 80 yr put a constant rate below 1/80 per yr at 63% '
       'one-sided confidence (95%: ~3/80 = 0.0375), and the parameter is the hazard once a decisive advantage '
       'exists, a condition only the ~4-yr 1945-49 monopoly matches (it bounds the hazard below ~1/4 per yr at '
       '63%, ~3/4 at 95%); on this definition the other ~76 yr count as years with the lock-in gate closed. The '
       'upper end 1/80 of LogU(1e-4, 1/80) (median 1.1e-3) is therefore a judgement, not a bound the record '
       'establishes; AI-enabled coup routes, EO 14409\'s 30-day exclusive government access and gradual '
       'disempowerment are mechanisms without a number that would move it. Scenario S1-side-high keeps the former '
       'wider prior LogU(1e-3, 0.03), whose upper end is near the 95% bound, and shows the sensitivity of the '
       'results to this judgement; S3 halves both bounds (sources: baruch, coups, eo14409, gradual)',
       S0=PT(0.0), S2=LOGU(0.005, 0.05), S3=LOGU(5e-5, 1.0 / 160.0), S1side=LOGU(1e-3, 0.03)),
    # [Assumption] c_conc
    _p('c_conc', 'sign-free coupling of binding restraint to the lock-in hazard: '
                 'h_conc ~ max(0, 1 - c_conc R); negative = a partial regime raises lock-in risk',
       UNIF(-0.5, 1.0), '[Assumption]', True,
       'a regime binding some actors but not others can open a decisive gap for defectors (the USSR read '
       'the Baruch Plan as preserving the US monopoly; Biopreparat ran under the BWC); a full regime lowers '
       'it (Montreal) (sources: baruch, biopreparat, montreal)',
       S2=UNIF(-0.5, 0.5), S3=UNIF(0.0, 1.0)),
    # [Inference] p_bind_first
    _p('p_bind_first', 'P(the first visible loss, the HF incident, leads to a binding national measure at all); '
                       'otherwise its binding jump is zero. Later tier-2 losses always produce a jump of size e '
                       'compounded over m2',
       BETA(3.0, 2.0), '[Inference]', True,
       'first visible losses that drew a binding measure: CDC 2014 lab incidents -> GoF funding pause, Bhopal -> '
       'EPCRA, Exxon Valdez -> OPA 90; that drew none: the 737 MAX\'s first crash (only an airworthiness '
       'directive), Sverdlovsk (1979); HF -> no binding statute after 0.24 yr and counting, and the EU let its own '
       'delay of high-risk AI rules enter into force 11 days after HF; Beta(3, 2) has mean 0.6 '
       '(sources: gof_pause, epcra, opa90, max_grounding, sverdlovsk, hawley_hearing, eu_omnibus)',
       S0=PT(0.0), S2=BETA(2.0, 3.0), S3=BETA(4.0, 1.0)),
]

# --- Structural switches (not sampled) ----------------------------------------------
SWITCHES_DEFAULT = {
    'learning_mode': 'hazard',   # 'hazard': learning multiplies u; 'containment': it lengthens tau_c
    'learn_count': 'mean',       # 'mean': expected learned count (mean field); 'sampled'; 'lograte'
    'step': False,              # S0 only: u = 1[x_def >= x_c] (v0.1 step at T_c)
    'g_clock': 'now',            # rate-growth clock starts at 'now' (T_START) or at 'Tc' (model.py)
    'response_off': False,       # S0 only: no visible losses, pauses, salience or path uniforms
    'events_off': False,         # block-level variant: no visible-loss or pause sampling, HF state kept
    'init_post_hf': True,        # post-HF initial state (HF jump pending, salience, 2026 pauses)
    'Hc_override_hours': None,   # fixed threshold in hours (x_c = log2(Hc/H0)) instead of the clock
    'phi_const': False,          # S0 only: E2 = E1 (v0.1 does not distinguish them)
    'learn_from_fatal': False,   # closed-form checks only: fatal clusters also count on the learning clock
}
SWITCH_DOCS = {
    'learning_mode': "'hazard' (default): learning factor multiplies u (failure modes closed); "
                     "'containment': it divides the containment-time median (threshold shifts up)",
    'learn_count': "'mean' (default): learning exponent Q = integral of eta_eff dN over the expected learned "
                   "count (exact expected factor for Poisson counts with factor 1 - eta_eff per cluster); "
                   "'sampled': learned clusters drawn as Poisson counts on the path (separate random stream), "
                   "factor 1 - eta_eff each; 'lograte': eta read as a log-reduction per cluster (factor "
                   "exp(-eta_eff)), exact expectation with 1 - exp(-eta_eff) per expected cluster",
    'step': 'S0 only: lethality is a step at x_c instead of the probit u(x_def)',
    'g_clock': "'now': rate growth starts at T_START (S1-S3); 'Tc': at T_c (model.py clock, S0)",
    'response_off': 'S0 only: no visible-loss sampling, no pauses, no salience; no random numbers',
    'events_off': 'block-level sensitivity only: no visible-loss or coordinated-pause sampling, but the '
                  'post-HF initial state (pending HF jump, HF salience, 2026 pauses) is kept',
    'init_post_hf': 'True in S1-S3: HF binding response pending with probability p_bind_first, HF salience, '
                    '2026 unilateral pauses, N_LEARN0 = 2; False in S0',
    'Hc_override_hours': 'None by default; a number replaces x_c by log2(Hc/H0_HOURS) for readers '
                         'who believe in a fixed hour count (S0 uses 814 h so that T_c = 2028.5)',
    'phi_const': 'S0 only: phi = 1 in every channel, so the E2 row equals the E1 row',
    'learn_from_fatal': 'False by default: the learning clock counts only clusters that left the path alive '
                        '(the path is conditioned on survival); True also counts fatal clusters, which '
                        'reproduces the time-based halving of paper sec. 5.4 (block 10 checks only)',
}

S0_TC = 2028.5                                   # paper 5.2-5.3 central T_c
S0_HC_HOURS = H0_HOURS * 2.0 ** ((S0_TC - T0) * 12.0 / 4.3)   # = 814 h, x_c = 5.5814

# --- Scenarios; printed in this order -------------------------------------------------
SCENARIOS = [
    {'key': 'S0', 'label': 'v0.1 premises, zero response (regression)', 'override_key': 'S0',
     'stream': 'S0', 'n': None,
     'switches': {'step': True, 'g_clock': 'Tc', 'response_off': True, 'init_post_hf': False,
                  'Hc_override_hours': S0_HC_HOURS, 'phi_const': True}},
    {'key': 'S1', 'label': 'historical human nature (headline)', 'override_key': None,
     'stream': 'S1', 'n': 'main', 'switches': {}},
    {'key': 'S1-cal', 'label': 'S1 with outside-view lethality e1 ~ LogU(1e-4, 1e-2)',
     'override_key': 'S1cal', 'stream': 'S1', 'n': 'var', 'switches': {}},
    {'key': 'S1-side-high', 'label': 'S1 with the higher-hazard side-channel priors used before they were '
                                     'anchored to their cited ranges: lam_mis ~ LogU(0.0005, 0.02), lam_conc ~ '
                                     'LogU(0.001, 0.03), phi_mis ~ Beta(2, 2), L_conc ~ U(0, 1.76) (the former '
                                     'lock-in multiple ratio_conc ~ LogU(1, 30) read at the median d)',
     'override_key': 'S1side', 'stream': 'S1', 'n': 'var2', 'switches': {}},
    {'key': 'S1-containment', 'label': "S1 with learning_mode = 'containment'", 'override_key': None,
     'stream': 'S1', 'n': 'var2', 'switches': {'learning_mode': 'containment'}},
    {'key': 'S2', 'label': 'pessimistic human nature (premise-driven: no learning, racing, C3 in full)',
     'override_key': 'S2', 'stream': 'S2', 'n': 'main', 'switches': {}},
    {'key': 'S3', 'label': 'optimistic coordination', 'override_key': 'S3', 'stream': 'S3',
     'n': 'main', 'switches': {}},
]

# Block-level sensitivity variants on the S1 draws (common random numbers).
# 'key'/'only': apply that scenario's overrides to human-nature ('hn') or AI-nature ('ai') parameters only;
# 'extra': fixed parameter values; 'sw': switch overrides.
BLOCK_VARIANTS = [
    {'id': 'S2hn', 'label': "S2's human-nature overrides only (AI nature as S1; includes s_regime)", 'key': 'S2',
     'only': 'hn'},
    {'id': 'S2ai', 'label': "S2's AI-nature overrides only (e1 ~ U(0.2, 1), phi_acc_max, delta_e2, s_align, "
                            "x_plateau = infinity)", 'key': 'S2', 'only': 'ai'},
    {'id': 'S3hn', 'label': "S3's human-nature overrides only (AI nature as S1)", 'key': 'S3', 'only': 'hn'},
    {'id': 'anchor', 'label': 'safe exit = historical regime anchor only (s_regime; no alignment, restraint or '
                              'halt routes)', 'extra': {'s_align': 0.0, 'kappa_s': 0.0, 's_halt': 0.0}},
    {'id': 'nosafe', 'label': 'no safe exit (s_align = s_regime = s_halt = 0)',
     'extra': {'s_align': 0.0, 's_regime': 0.0, 's_halt': 0.0}},
    {'id': 'noplateau', 'label': 'no capability plateau (x_plateau = infinity, premise C1)',
     'extra': {'x_plateau': INF}},
    {'id': 'nocouple', 'label': 'salience and restraint couplings off (w1 = kappa_s = s_halt = 0; w1 = 0 also '
                                'stops salience raising the learning rate and the alignment effort)',
     'extra': {'w1': 0.0, 'kappa_s': 0.0, 's_halt': 0.0}},
    {'id': 'nopause', 'label': 'coordinated pause off (race_persistence = 1, c_cap = 0)',
     'extra': {'race_persistence': 1.0, 'c_cap': 0.0}},
    {'id': 'noevents', 'label': 'visible-loss responses off (no event or pause sampling; post-HF initial state kept)',
     'sw': {'events_off': True}},
    {'id': 'zero', 'label': 'zero response and no safe exit (the v0.1 premise with S1 AI-nature priors)',
     'sw': {'events_off': True, 'init_post_hf': False},
     'extra': {'eta': 0.0, 's_align': 0.0, 's_regime': 0.0, 's_halt': 0.0}},
    {'id': 'S1sampled', 'label': 'learned clusters sampled as Poisson counts (factor 1 - eta_eff each)',
     'sw': {'learn_count': 'sampled'}},
    {'id': 'S1lograte', 'label': 'eta read as a log-reduction per cluster (factor exp(-eta_eff); exact expectation)',
     'sw': {'learn_count': 'lograte'}},
]
# Variants run on the S3 draws (common random numbers with S3).
S3_VARIANTS = [
    {'id': 'S3anchor', 'label': 'S3, safe exit = regime anchor only (s_align = kappa_s = s_halt = 0)',
     'extra': {'s_align': 0.0, 'kappa_s': 0.0, 's_halt': 0.0}},
    {'id': 'S3nosafe', 'label': 'S3, no safe exit (s_align = s_regime = s_halt = 0)',
     'extra': {'s_align': 0.0, 's_regime': 0.0, 's_halt': 0.0}},
    {'id': 'S3sampled', 'label': 'S3, learned clusters sampled as Poisson counts', 'sw': {'learn_count': 'sampled'}},
    {'id': 'S3lograte', 'label': 'S3, eta read as a log-reduction per cluster', 'sw': {'learn_count': 'lograte'}},
]
# Racing and dismissal parameters for the sign check (prior q01 vs q99; expected sign of dP(E1): +), plus
# lam_conc for the competing-risk check on E2 (lock-in pre-empts later accidental strict extinction) and r0
# as a reference row (the measured 2026 incident rate, AI nature; not a racing parameter).
SIGN_PARAMS = ['r0', 'g', 'r_max', 'w', 'race_persistence', 'lam_conc']
SIGN_RACING = ['g', 'r_max', 'w', 'race_persistence']
# Human-nature effect sweep (block 11): |dP(E1)| below this at 2035, 2050 and 2100 = 'inert in this model'.
INERT_PP = 0.01
HNE_YEARS = (2035, 2050, 2100)

# S0 cells (paper 5.2 / 5.3): p x rate form at T_c = 2028.5, r0 = 4.
# paper_53: the published 5.3 cells (percent; "100" means above 99.5%); paper_52: 5.2 medians;
# paper_52_br: the 5.2 bracket medians at T_c = 2027.3 (d = 3 months, H_c = 167 h) and 2030.5 (d = 7 months,
# H_c = 2,000 h).
S0_CELLS = [
    {'p': 0.01, 'grow': False, 'paper_53': [6, 23, 37, 58, 94], 'paper_52': 2045.8, 'paper_52_br': [2044.7, 2047.8]},
    {'p': 0.01, 'grow': True, 'paper_53': [10, 99, 100, 100, 100], 'paper_52': 2032.2, 'paper_52_br': [2031.0, 2034.2]},
    {'p': 0.05, 'grow': False, 'paper_53': [26, 73, 90, 99, 100], 'paper_52': 2032.0, 'paper_52_br': [2030.8, 2034.0]},
    {'p': 0.05, 'grow': True, 'paper_53': [41, 100, 100, 100, 100], 'paper_52': 2030.3, 'paper_52_br': [2029.1, 2032.3]},
    {'p': 0.20, 'grow': False, 'paper_53': [70, 99, 100, 100, 100], 'paper_52': 2029.4, 'paper_52_br': [2028.2, 2031.4]},
    {'p': 0.20, 'grow': True, 'paper_53': [88, 100, 100, 100, 100], 'paper_52': 2029.2, 'paper_52_br': [2028.0, 2031.2]},
]
S0_BRACKET_CELLS = [(3.0, 167.0), (7.0, 2000.0)]   # (d_months, H_c hours) giving the 5.2 bracket ends
S0_YEARS = [2030, 2035, 2040, 2050, 2100]
S0_GRID_36 = {'d_months': [3.0, 4.3, 7.0], 'Hc_hours': [167.0, 2000.0], 'p': [0.01, 0.05, 0.20]}
REGRESSION_TOL_P = 2e-4      # 0.02 percentage points
REGRESSION_TOL_MED = 0.05    # years

# --- Outside view. Y_src = horizon of the source (None: no horizon, read as 2100) ---
OUTSIDE_VIEW = {
    'E1_pool': [
        {'source': 'AI Impacts 2023 (Grace et al. 2024), Q2 control-failure phrasing',
         'definition': 'E1 via loss of control (extinction or similarly permanent and severe '
                       'disempowerment)', 'vintage': '2023-10', 'hf': 'pre', 'p_src': 0.10,
         'y_src': None, 'tag': '[Fact]', 'src': 'aiimpacts_2023', 'scope': 'loss of control'},
        {'source': 'AI Impacts Dec-2024 wave (published 2026-09-25)',
         'definition': 'E1 (same Q1 phrasing); median, secondary (mean 18%)', 'vintage': '2024-12',
         'hf': 'pre (fielded)', 'p_src': 0.10, 'y_src': None, 'tag': '[Fact, median secondary]',
         'src': 'aiimpacts_2024', 'scope': 'general AI phrasing as recorded (not re-verified)'},
        {'source': 'Ord, The Precipice (2020)',
         'definition': 'existential catastrophe from unaligned AI, 2020-2120', 'vintage': '2020',
         'hf': 'pre', 'p_src': 0.10, 'y_src': 2120.0, 'tag': '[Fact]', 'src': 'ord', 'scope': 'unaligned AI'},
        {'source': 'Carlsmith (2022 update)',
         'definition': 'existential catastrophe from power-seeking AI by 2070 (> 10%)',
         'vintage': '2022-05', 'hf': 'pre', 'p_src': 0.10, 'y_src': 2070.0, 'tag': '[Fact]',
         'src': 'carlsmith_2022', 'scope': 'power-seeking AI (misalignment)'},
        {'source': "Good Judgment superforecasters on Carlsmith's premises",
         'definition': 'same as Carlsmith, by 2070', 'vintage': '2023-10', 'hf': 'pre',
         'p_src': 0.01, 'y_src': 2070.0, 'tag': '[Fact]', 'src': 'carlsmith_gj',
         'scope': 'power-seeking AI (misalignment)'},
    ],
    # channel scope of each pool (printed with the POOL rows in blocks 6 and 7)
    'E1_scope': 'misalignment / loss of control: 4 of 5 anchors (AI Impacts 2023 Q2, Ord, Carlsmith, Good Judgment) '
                'ask only about that channel, which the model calls accidental escape; the AI Impacts Dec-2024 '
                'anchor is recorded with the general phrasing; no anchor covers deliberate misuse or lock-in alone',
    'E2_scope': 'all AI channels; 3 pre-HF anchors (XPT superforecasters, XPT experts, Metaculus Q27035 '
                '[unverified]) and 1 post-HF anchor (Metaculus Pros 2026-10-08, AI kills >= 95%, an upper bound on '
                'E2 and the largest member)',
    'def_notes': {'E1': 'E1 pool: pre-HF, mostly loss-of-control anchors',
                  'E2': 'E2 pool: 3 pre-HF anchors + 1 post-HF upper-bound anchor; the post-HF multipliers partly '
                        'double-count that anchor'},
    'E1_not_pooled': [
        {'source': 'AIRO (frontier-model forecasting panel)', 'value': 0.28,
         'reason': 'disempowerment by 2100; a model panel, not humans; no URL [unverified]', 'src': ''},
        {'source': 'Samotsvety (2022-09)', 'value': 0.25,
         'reason': 'misaligned takeover by 2100, conditional on no pre-APS-AI catastrophe; page fetched '
                   '2026-10-10', 'src': 'samotsvety'},
        {'source': 'x-risk community leaders survey (Feb 2026)', 'value': 0.25,
         'reason': 'any cause, self-selected summit attendees', 'src': 'xrisk_leaders'},
        {'source': "Metaculus Pros 'AI kills >= 95%' (2026-09-23 derived 23%)", 'value': 0.23,
         'reason': 'derived product of three linked questions (2026-09-23) for AI kills >= 95%, a looser event '
                   'than E2 and not an E1 estimate; the E2 pool uses the direct 2026-10-08 elicitation of the '
                   'same event instead', 'src': 'metaculus_0923'},
    ],
    'E2_pool': [
        {'source': 'XPT superforecasters (Karger et al. 2023)', 'definition': 'population < 5,000 by 2100',
         'vintage': '2022', 'hf': 'pre', 'p_src': 0.0038, 'y_src': 2100.0, 'tag': '[Fact]', 'src': 'xpt'},
        {'source': 'XPT AI domain experts', 'definition': 'population < 5,000 by 2100',
         'vintage': '2022', 'hf': 'pre', 'p_src': 0.03, 'y_src': 2100.0, 'tag': '[Fact]', 'src': 'xpt'},
        {'source': 'Metaculus Q27035 (community, midpoint of 1-2%)',
         'definition': 'AI pushes population below 5,000 before 2100', 'vintage': 'undated',
         'hf': 'unknown', 'p_src': 0.015, 'y_src': 2100.0, 'tag': '[unverified]', 'src': 'metaculus_27035'},
        {'source': 'Metaculus Pros direct (2026-10-08): 5.4/10/14% -> geometric mean',
         'definition': 'AI kills >= 95% before 2100 (upper bound on E2)', 'vintage': '2026-10-08',
         'hf': 'post', 'p_src': math.exp((math.log(0.054) + math.log(0.10) + math.log(0.14)) / 3.0),
         'y_src': 2100.0, 'tag': '[Fact]', 'src': 'metaculus_pros'},
    ],
    # cumulative-hazard fraction f(Y) = Lambda(Y)/Lambda(2100); 0 at T_START, 1 at 2100,
    # piecewise linear between these points (XPT, LEAP wave 9, AIRO year points)
    'profiles': {'E1': [(2030.0, 0.03), (2050.0, 0.42)], 'E2': [(2030.0, 0.0013), (2050.0, 0.17)]},
    'profile_sources': 'xpt, leap9 (AIRO year points: no URL [unverified])',
    # Fixed multipliers A on the pooled anchors' century hazard; the model-derived human-nature rows are
    # computed at run time from the block-level variants (block 6).
    'multipliers': [
        {'A': 1.0, 'id': 'raw', 'label': 'outside view raw = central human-nature adjustment (x1.00; anchors as '
                                          'published)',
         'basis': 'the pooled anchors as published; all but the Metaculus Pros anchor of the E2 pool predate HF'},
        {'A': 1.8, 'id': 'x1.8', 'label': 'outside view x1.8, post-HF same-panel update (FRI LEAP 9 -> 12)',
         'basis': 'same-panel FRI LEAP wave 9 -> 12 superforecaster move on AI catastrophe by 2050 '
                  '(0.88% -> 1.6%), rationales cite the HF incident; experts unchanged (x1.0) (sources: leap9, leap12)'},
        {'A': 2.3, 'id': 'x2.3', 'label': 'outside view x2.3, post-HF premise update, generous (timelines x1.19, '
                                          'incentives x1.06, high-impact failures x1.8)',
         'basis': 'Good Judgment premise arithmetic with timelines 0.80 -> 0.95 (x1.19), incentives 0.90 -> 0.95 '
                  '(x1.06), high-impact failures 0.25 -> 0.45 (x1.8), disempowerment premises untouched; the '
                  "review's incident-justified update, 'x2.3 under a generous reading' (sources: carlsmith_gj)"},
        {'A': 0.95 / 0.90, 'id': 'x1.06', 'label': 'outside view x1.06, human-nature component of the x2.3 update '
                                                   '(incentives 0.90 -> 0.95 only)',
         'basis': 'the only human-nature premise in the x2.3 arithmetic: P(incentives to build and deploy APS '
                  'systems) 0.90 -> 0.95, i.e. racing continued after HF; timelines and high-impact failures are '
                  'AI-nature premises (sources: carlsmith_gj)'},
    ],
}

# --- Premise comparison (Carlsmith Substack 2023-10-18) ---------------------------------
PREMISES = {
    'labels': ['timelines', 'incentives', 'alignment difficulty', 'high-impact failures',
               'disempowerment', 'catastrophe | disempowerment'],
    'Good Judgment': {'values': [0.80, 0.90, 0.58, 0.25, 0.05, 0.40], 'overall': '1% by 2070'},
    'Carlsmith': {'values': [0.65, 0.80, 0.40, 0.65, 0.40, 0.95], 'overall': '~5% by 2070'},
    'source': 'carlsmith_gj (joecarlsmith.substack.com/p/superforecasting-the-premises-in, 2023-10-18)',
}

# --- Grid and output years ------------------------------------------------------
YEARS = [2027, 2028, 2029, 2030, 2032, 2035, 2040, 2050, 2075, 2100]
GRID_DOC = ('319 intervals: monthly 2026.75-2040, quarterly 2040-2060, half-yearly 2060-2100; '
            'hazards piecewise constant at interval midpoints')
PERIODS = [(T_START, 2030.0), (2030.0, 2035.0), (2035.0, 2050.0), (2050.0, 2100.0)]

# --- The model in one screen (printed as block 1b and mirrored in params.yaml) -----------
MODEL_SUMMARY = [
    'State per path: capability log2H (doublings of the frontier 50% horizon above H0 = 17 h), binding restraint '
    'R, salience Sal, tier-2 salience Sal2, unilateral restraint m_uni, coordinated pause on/off, learning count '
    'N_learn; the still-at-risk weight S (Rao-Blackwellised).',
    'Capability: d log2H/dt = (1/d_yr)(1 - gamma_d R)(1 - c_cov [paused])(1 - log2H/x_plateau); x_plateau = '
    'infinity is premise C1 (no bound). x_def = (1 - a) log2H (net of defense) drives only the containment clock '
    '(u, dwell time, tier-2 share q2, threshold crossing); the misuse and lock-in gates and phi_acc use raw '
    'capability x = log2H.',
    'Threshold x_c = log2(30 d / tau_c0)/b_c: the capability at which the median containment time (detection -> '
    'no further unauthorised actions) reaches 30 days; T_c = T0 - d_yr x_plateau ln(1 - x_c/((1 - a) x_plateau)), '
    'never if x_c >= (1 - a) x_plateau.',
    'Uncontainability of one sub-incident u = Phi(beta_c (x_def - x_c)), beta_c = b_c ln2/sigma_c (u = 1/2 at x_c).',
    'Root-cause clusters per year r_root = r_start G_r(t)(1 - R) m_avg (1 - c_cov [paused]), r_start = min(r0, '
    'r_max); G_r is a logistic from 1 to r_max/r_start at rate g, so the rate never exceeds r_max; m_avg is the '
    'exact interval average of the unilateral multiplier m_uni. A cluster has m sub-incidents, m_eff = 1 + (m - 1)'
    '(1 - rho) effectively independent.',
    'Per cluster: u_cl = 1 - (1 - u_eff)^m_eff and p_cl = 1 - (1 - u_eff e1)^m_eff, u_eff = u L_learn (1 - beta_p R).',
    'Learning (per incident, leaky): each learned cluster closes the fraction eta_eff of the remaining removable '
    'hazard at the time it is learned; Q(t) = eta_eff(T_START) N_LEARN0 + integral of eta_eff dN_learn, '
    'L_learn = f_min + (1 - f_min) exp(-Q(t - dwell)) (the lagged Q held at its running maximum), eta_eff = '
    'min(ETA_EFF_MAX, eta share/SIGMA_REF); N_learn '
    'is the expected count of detected, non-fatal clusters that are public, uncontainable or not dismissed (w) on '
    'the surviving path. exp(-Q) is the exact expected factor for Poisson counts (mean field); learned failure '
    'modes stay closed.',
    'Safety share = min(SHARE_MAX, safety_share [1 + (share_mult_max - 1)(1 - exp(-w1 Sal))]).',
    'Hazards: accident h_acc = r_root p_cl; misuse h_mis = lam_mis G(x - L_open/d_yr - x_c)(1 - beta_p R); '
    'concentration h_conc = lam_conc G(x - x_c - L_conc/d_yr) max(0, 1 - c_conc R); G(z) = Phi(beta_c z); x '
    'is raw capability. Restraint R cuts the cluster rate by (1 - R); beta_p R additionally cuts uncontainability '
    'and misuse.',
    'Safe exit (absorbing), two routes: verified alignment s_align min(s_effort_max, share/SIGMA_REF) '
    'G(x_def - x_c) (AI nature: science, gated on the path\'s own threshold crossing) plus a durable control regime or '
    'deliberate halt s_regime (1 + kappa_s R) + s_halt [paused] (human nature, calibrated to the historical '
    '0.5-1%/yr).',
    'E1 = first of h_acc + h_mis + h_conc before a safe exit. E2 = population < 5,000 by Y: an accidental E1 world '
    'becomes strict extinction once its raw capability passes the level set by phi_acc, also after the E1 event, so '
    'P(E2 by Y) = phi_acc(x(Y)) P(E1 via accident by Y) + phi_mis P(E1 via misuse by Y) + phi_conc P(E1 via '
    'concentration by Y), a subset of E1.',
    'Visible losses: a Poisson process at rate r_root [q_v (1 - u_cl) + (u_cl - p_cl)] (K ~ Poisson(rate dt) per '
    'interval); each is tier 2 (mass-casualty/major loss) with P = q2 = q2_max G(x_def - x_c + delta_2), else tier 1 '
    '(HF class).',
    'Tier 1: Sal += 1 (after the dwell delay), unilateral pause with P = pi_uni. Tier 2: Sal += m2, Sal2 += 1, '
    'unilateral draw, binding jump R += (1 - (1 - e)^m2)(R_max - R) after L_nat years (the HF jump, size e, is '
    'pending at the start with probability p_bind_first).',
    'Coordinated pause hazard pi0 + pi1 min(Sal2, sal2_sat), pi0 = -ln(race_persistence)/10, pi1 = -ln(1 - c_cap)/'
    'L_int; it lasts D_coord years and leaves R >= e. Tier-1 events do not raise it.',
    'Decay: R halves every tau_h, Sal and Sal2 every h_sal, unilateral restraint every h_uni_weeks.',
    'Output: prior-predictive means over the parameter draws; nothing is conditioned on data except through '
    'the priors and the 2026-10 initial state.',
]
# Design notes for the fix round of 2026-10-10 (mirrored in params.yaml; each answers a verifier issue).
DESIGN_NOTES = [
    'Units: r0 and r_max are root-cause cluster rates (no division by the cluster size m); e1 and q0 are per '
    'effective sub-incident and are compounded over m_eff per cluster.',
    'No hidden caps: the salience multiplier on the safety share saturates smoothly at the sampled human-nature '
    'parameter share_mult_max, the effort multiplier on the alignment route is bounded by the sampled '
    's_effort_max (it replaces a fixed cap of 3); SHARE_MAX and ETA_EFF_MAX are listed constants; the saturation '
    'or binding share of each is printed in block 12.',
    'The coordinated pause is driven by tier-2 (mass-casualty or major-loss) salience Sal2 only, saturating at '
    'sal2_sat, so race_persistence keeps its documented meaning; HF and other tier-1 events raise only the '
    'safety share and unilateral pauses.',
    'An uncontainable but non-fatal cluster is a visible loss; it is tier 2 with the same probability q2 as any '
    'other visible loss, instead of always.',
    'Visible losses are a Poisson count per interval; learning integrates eta_eff over learned clusters '
    '(per-incident, never forgotten); E2 is population < 5,000 by Y with post-E1 progression in capability (the '
    'first-event convention is printed as a lower bound); the safe exit has an AI-nature route s_align (gated on '
    'the threshold crossing) and a human-nature route s_regime (calibrated to the historical anchor); a capability '
    'plateau x_plateau makes premise C1 attackable; the salience response saturates smoothly; m2 sets the size '
    'of tier-2 binding jumps; unilateral restraint uses the exact interval integral of its decay.',
    'Side channels: lam_mis is anchored to the historical range its rationale cites (1e-4 to 3e-3 per yr); '
    'lam_conc has the upper end 1/80 per yr, a judgement rather than a historical bound (zero global lock-ins in '
    '80 yr give 1/80 only as a 63% one-sided bound, 95% ~3/80, and only the ~4-yr 1945-49 monopoly matches its '
    'conditioning on a decisive advantage); phi_mis is anchored to the XPT engineered-pathogen ratios and the '
    'lock-in gap L_conc to the 1-5 yr between the AI Futures Automated Coder and ASI medians (it replaces a '
    'capability multiple LogU(1, 30) above the threshold, 0-1.76 yr at the median d); the former, higher-hazard '
    'priors are kept as scenario S1-side-high (lam_conc upper end 0.03, near that 95% bound), which shows the '
    'sensitivity to them. The E1 outside-view pool covers mostly the accident channel, so '
    'blocks 6 and 7 print the accident channel and the misuse + lock-in channels separately.',
    'Scope of the defense ratio a: only the containment clock (u, dwell time, q2 and the threshold crossing); '
    'misuse, lock-in and strict-extinction progression use raw capability.',
    'The learning clock counts only clusters that left the path alive (survivor conditioning); r_max is a cap '
    'from the start (the rate starts at min(r0, r_max)); the HF binding response happens with probability '
    'p_bind_first instead of with certainty; the dwell-time exponent is capped before exponentiating.',
    'w1 keeps the tag [Assumption]; its prior is U(0.2, 1) under the saturating response (one undecayed '
    'HF-class loss raises the safety share x1.2-1.8 at the share_mult_max median, the review\'s proposed step '
    'was x1.5-2); no post-HF change in the share has been measured.',
]

# Measured q01 -> q99 effects of the human-nature parameters on P(E1 by 2035 / 2050 / 2100), in percentage
# points (common random numbers on the first S1 draws; block 11, mirrored in params.yaml human_nature: effect).
# Recorded from the default run (seed DEFAULT_SEED; N_HNE_DEFAULT draws per side, N_SIGN_DEFAULT for the
# parameters in the sign check). Each full run recomputes them and warns in block 11 if this record is stale.
HN_EFFECT_RECORD = {
    'a': (-6.68, -4.48, -3.33), 'g': (6.86, 7.75, 7.19), 'r_max': (3.99, 7.44, 8.39), 'q0': (-5.71, -5.59, -5.55),
    'eta': (-19.15, -19.59, -17.77), 'f_min': (4.99, 10.41, 13.96), 'w': (1.49, 1.54, 1.15),
    'safety_share': (-8.36, -8.89, -9.07), 'w1': (-1.17, -1.10, -0.92), 'share_mult_max': (-2.83, -3.15, -3.38),
    's_effort_max': (-0.27, -1.19, -2.92), 'm2': (-2.10, -1.53, -1.56), 'L_nat': (6.10, 2.69, 2.08),
    'e': (-7.29, -6.83, -6.55), 'R_max': (-7.16, -8.40, -9.49), 'tau_h': (-1.31, -1.93, -1.92),
    'beta_p': (-3.75, -4.70, -5.12), 'gamma_d': (-0.47, -0.40, -0.32), 'h_sal': (-0.31, -0.69, -0.71),
    'pi_uni': (-0.31, -0.54, -0.37), 'e_uni': (-0.33, -0.68, -0.68), 'h_uni_weeks': (0.16, -0.11, -0.15),
    'race_persistence': (0.51, 0.50, 0.41), 'c_cap': (-0.91, -1.08, -1.34), 'sal2_sat': (-0.32, -0.82, -1.20),
    'L_int': (3.23, 4.39, 5.12), 'c_cov': (-1.30, -1.92, -1.64), 'D_coord': (-1.97, -3.39, -4.02),
    's_regime': (-1.88, -5.44, -11.69), 's_halt': (-0.40, -1.85, -5.41), 'kappa_s': (-0.16, -0.62, -1.63),
    'lam_mis': (0.71, 1.83, 3.57), 'lam_conc': (1.29, 6.30, 13.20), 'c_conc': (-0.15, -0.97, -2.26),
    'p_bind_first': (-0.95, -0.33, -0.19),
}
# Why a parameter is inert in this model (|effect| < INERT_PP at every horizon); written from the measured run.
HN_INERT_REASON = {
    'gamma_d': 'binding restraint slows capability growth only once it has built up (mean R is about 0.1-0.3 in '
               '2028-2030, block 12), while the median path crosses its own threshold around 2029; slower growth '
               'also delays the alignment gate, which offsets part of the gain',
    'h_sal': 'the salience response of the safety share is at least 90% saturated during most of the at-risk time '
             '(block 12), so a longer salience memory adds little; tier-2 losses recur on most paths and renew it',
    'pi_uni': 'unilateral pauses last weeks (h_uni_weeks 2-8) and cut the aggregate cluster rate by only about 5% of '
              'at-risk time on average (block 12, mean unilateral rate cut); with per-incident learning a lower rate '
              'partly delays accidental hits instead of removing them',
    'e_uni': 'same mechanism as pi_uni: unilateral pauses are short and cut the aggregate rate by only about 5% on '
             'average (block 12)',
    'h_uni_weeks': 'same mechanism as pi_uni: even the longest unilateral half-life (8 weeks) is short against the '
                   'decades over which the hazard accumulates',
    'race_persistence': 'it sets only the proactive coordinated-pause hazard pi0; in S1 tier-2 losses arrive early on '
                        'most paths and drive the coordinated-pause hazard through pi1, so only about 4% of first '
                        'coordinated pauses precede every tier-2 loss (block 12); racing acts in this model through g, '
                        'r_max and w',
    'p_bind_first': 'only the HF binding jump depends on it; tier-2 losses produce their own binding jumps later on '
                    'almost every S1 path (share of paths with a first binding measure, block 12), so whether HF '
                    'itself binds matters only before the first tier-2 loss',
}
# === PARAMS END =============================================================


# ----------------------------------------------------------------------------
# Grid
# ----------------------------------------------------------------------------
def _make_grid():
    g = [round(T_START + k / 12.0, 6) for k in range(160)]
    g += [round(2040.0 + j * 0.25, 6) for j in range(1, 81)]
    g += [round(2060.0 + j * 0.5, 6) for j in range(1, 81)]
    return g


GRID = _make_grid()
NB = len(GRID)
DT = [0.0] + [GRID[k] - GRID[k - 1] for k in range(1, NB)]
TM = [0.0] + [GRID[k - 1] + 0.5 * DT[k] for k in range(1, NB)]
DCLS = [0] + [0 if k <= 159 else (1 if k <= 239 else 2) for k in range(1, NB)]
CLS_DT = (1.0 / 12.0, 0.25, 0.5)
YIDX = {}
for _Y in YEARS + [2028.5, 2036.75]:
    _hits = [i for i in range(NB) if abs(GRID[i] - _Y) < 1e-6]
    assert len(_hits) == 1, 'grid does not contain boundary %s' % _Y
    YIDX[_Y] = _hits[0]
NY = len(YEARS)
YPOS = [-1] * NB
for _i, _Y in enumerate(YEARS):
    YPOS[YIDX[_Y]] = _i
YP = dict((Y, i) for i, Y in enumerate(YEARS))
PERIOD = [0] * NB
for _k in range(1, NB):
    PERIOD[_k] = 0 if TM[_k] < 2030.0 else (1 if TM[_k] < 2035.0 else (2 if TM[_k] < 2050.0 else 3))
assert NB == 320
PARAM_NAMES = [p['name'] for p in PARAMS]
PARAM_BY_NAME = dict((p['name'], p) for p in PARAMS)
N_HN = sum(1 for p in PARAMS if p['hn'])
assert len(PARAMS) == 56 and N_HN == 35, (len(PARAMS), N_HN)
SOURCE_KEYS = dict((s[0], s) for s in SOURCES)


def _check_source_keys():
    """Every '(sources: ...)' key in a rationale must resolve in SOURCES."""
    bad = []
    texts = [p['rationale'] for p in PARAMS] + [c['rationale'] for c in CONSTANTS]
    for t in texts:
        i = t.find('(sources:')
        if i < 0:
            continue
        j = t.find(')', i)
        for key in t[i + 9:j].split(','):
            key = key.strip()
            if key and key not in SOURCE_KEYS:
                bad.append(key)
    for blk in ('E1_pool', 'E1_not_pooled', 'E2_pool'):
        for a in OUTSIDE_VIEW[blk]:
            if a.get('src') and a['src'] not in SOURCE_KEYS:
                bad.append(a['src'])
    return bad


assert not _check_source_keys(), 'unresolved source keys: %s' % _check_source_keys()


# ----------------------------------------------------------------------------
# Samplers and small helpers
# ----------------------------------------------------------------------------
def sample(spec, rng):
    d = spec['dist']
    if d == 'point':
        return spec['value']
    if d == 'lognormal':
        return spec['median'] * math.exp(spec['sigma'] * rng.gauss(0.0, 1.0))
    if d == 'loguniform':
        return math.exp(rng.uniform(math.log(spec['lo']), math.log(spec['hi'])))
    if d == 'beta':
        return rng.betavariate(spec['a'], spec['b'])
    if d == 'uniform':
        return rng.uniform(spec['lo'], spec['hi'])
    if d == 'one_plus_lognormal':
        return 1.0 + spec['median'] * math.exp(spec['sigma'] * rng.gauss(0.0, 1.0))
    raise ValueError('unknown prior %r' % d)


def fnum(v):
    if v == 0:
        return '0'
    if abs(v - LN2) < 1e-12:
        return 'ln2'
    a = abs(v)
    if a >= 1e5 or a < 1e-3:
        return '%.3g' % v
    return '%.4g' % v


def prior_str(spec):
    if spec is None:
        return '-'
    d = spec['dist']
    if d == 'point':
        return '= %s' % fnum(spec['value'])
    if d == 'lognormal':
        return 'LogN(med %s, sd %s)' % (fnum(spec['median']), fnum(spec['sigma']))
    if d == 'loguniform':
        return 'LogU(%s, %s)' % (fnum(spec['lo']), fnum(spec['hi']))
    if d == 'beta':
        return 'Beta(%s, %s)' % (fnum(spec['a']), fnum(spec['b']))
    if d == 'uniform':
        return 'U(%s, %s)' % (fnum(spec['lo']), fnum(spec['hi']))
    if d == 'one_plus_lognormal':
        return '1+LogN(med %s, sd %s)' % (fnum(spec['median']), fnum(spec['sigma']))
    return str(spec)


# the S1-side-high label states its priors; it must name every S1side override exactly
_SIDE_LABEL = [sc for sc in SCENARIOS if sc['override_key'] == 'S1side'][0]['label']
assert all('%s ~ %s' % (p['name'], prior_str(p['overrides']['S1side'])) in _SIDE_LABEL
           for p in PARAMS if 'S1side' in p['overrides']), _SIDE_LABEL


def Phi(z):
    v = 0.5 * (1.0 + math.erf(z / SQRT2))
    return min(max(v, 1e-12), 1.0 - 1e-12)


def qtl(xs, q):
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return float('nan')
    pos = q * (n - 1)
    i = int(pos)
    f = pos - i
    if i + 1 < n and f > 0.0:
        if s[i + 1] == INF:
            return INF
        return s[i] * (1.0 - f) + s[i + 1] * f
    return s[i]


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float('nan')


def se_of(xs):
    xs = list(xs)
    return statistics.stdev(xs) / math.sqrt(len(xs)) if len(xs) > 1 else float('nan')


def pc(x, nd=1):
    if x is None or x != x:
        return 'n/a'
    return '%.*f%%' % (nd, 100.0 * x)


def pp(x, nd=1):
    if x is None or x != x:
        return 'n/a'
    return '%+.*f pp' % (nd, 100.0 * x)


def yr(t):
    if t is None:
        return '>2100'
    if t == INF:
        return 'never'
    return '%.1f' % t


def cell(s):
    return str(s).replace('|', '\\|')


OUT = []


def emit(s=''):
    OUT.append(s)


def table(header, rows):
    emit('| ' + ' | '.join(cell(h) for h in header) + ' |')
    emit('|' + '|'.join(' --- ' for _ in header) + '|')
    for r in rows:
        emit('| ' + ' | '.join(cell(c) for c in r) + ' |')
    emit()


def progress(msg):
    sys.stderr.write('[mc.py %6.1fs] %s\n' % (time.time() - T_WALL0, msg))
    sys.stderr.flush()


T_WALL0 = time.time()


# ----------------------------------------------------------------------------
# Parameter draws
# ----------------------------------------------------------------------------
def override_spec(override_key, name):
    if override_key is None:
        return None
    return PARAM_BY_NAME[name]['overrides'].get(override_key)


def draw_params(override_key, stream, j, seed, extra=None, only=None):
    """Draw all parameters in PARAMS order from the S1 priors with rng_p (each draw is consumed even
    when overridden, so streams stay aligned); scenario overrides are drawn from a separate generator
    rng_o so that the non-overridden parameters equal those of the same S1 draw. only = 'hn' or 'ai'
    restricts the overrides to human-nature or AI-nature parameters (block-level sensitivity)."""
    rng_p = random.Random('%d:%s:%d:p' % (seed, stream, j))
    rng_o = random.Random('%d:%s:%d:o' % (seed, stream, j))
    P = {}
    for spec in PARAMS:
        v = sample(spec['prior'], rng_p)
        if override_key is not None:
            ov = spec['overrides'].get(override_key)
            if ov is not None and (only is None or (only == 'hn') == spec['hn']):
                v = sample(ov, rng_o)
        P[spec['name']] = v
    if extra:
        P.update(extra)
    return P


def scenario_switches(sc):
    sw = dict(SWITCHES_DEFAULT)
    sw.update(sc['switches'])
    ov = override_spec(sc['override_key'], 'L_nat')
    sw['L_nat_prior'] = ov if (ov is not None and ov['dist'] != 'point') else PARAM_BY_NAME['L_nat']['prior']
    return sw


def prior_quantiles(seed):
    """q01, q10, q50, q90, q99 of each S1 prior from N_PRIOR_Q draws."""
    rng = random.Random('%d:q' % seed)
    tab = {}
    for spec in PARAMS:
        xs = sorted(sample(spec['prior'], rng) for _ in range(N_PRIOR_Q))
        n = len(xs)
        tab[spec['name']] = tuple(xs[int(round(q * (n - 1)))] for q in (0.01, 0.1, 0.5, 0.9, 0.99))
    return tab


# ----------------------------------------------------------------------------
# Engine: one path (equations M1-M12; Rao-Blackwellised competing risks)
# ----------------------------------------------------------------------------
def new_sums():
    return [[0.0] * NB, [0.0] * NB, [0.0] * NB]


def poisson_inv(lam, U):
    """K ~ Poisson(lam) by inverting one uniform U (cumulative pmf); normal approximation for lam > 500."""
    if lam <= 0.0:
        return 0
    if lam > 500.0:
        z = NORMAL.inv_cdf(min(max(U, 1e-12), 1.0 - 1e-12))
        k = int(round(lam + math.sqrt(lam) * z))
        return k if k > 0 else 0
    c = math.exp(-lam)
    cum = c
    n = 0
    while U > cum and n < 100000:
        n += 1
        c *= lam / n
        cum += c
    return n


def run_path(P, sw, rng_x, sums, diag=False):
    """Integrates one parameter draw on the grid. Hazards are piecewise constant on each interval
    (evaluated at the midpoint), so every per-interval integral is exact. Adds the sub-distribution
    curves P(E1 by t), P(E2 by t), P(safe by t) into sums and returns the per-draw record. With diag,
    also accumulates survival-weighted diagnostics (safe-exit routes, pause time, cap binding)."""
    exp = math.exp
    erf = math.erf
    log2 = math.log2
    rand = rng_x.random
    grid = GRID
    sumE1, sumE2, sumSafe = sums
    DT_, TM_, DCLS_, YPOS_, PERIOD_ = DT, TM, DCLS, YPOS, PERIOD      # local bindings (speed)
    SHARE_MAX_, ETA_EFF_MAX_, TAU_D_CAP_D_, SIGMA_REF_, C_SAT_ = SHARE_MAX, ETA_EFF_MAX, TAU_D_CAP_D, SIGMA_REF, C_SAT_DIAG
    pois = poisson_inv

    d_yr = P['d_months'] / 12.0
    oma = 1.0 - P['a']
    b_c = P['b_c']
    bc2 = b_c * LN2 / P['sigma_c'] / SQRT2       # Phi(beta_c z) = 0.5 (1 + erf(bc2 z))
    if sw['Hc_override_hours'] is not None:
        x_c = log2(sw['Hc_override_hours'] / H0_HOURS)
    else:
        x_c = log2(TAU_STAR_D / P['tau_c0_days']) / b_c
    x_p = P['x_plateau']
    plateau = x_p < 1e6
    inv_dxp = 1.0 / (d_yr * x_p) if plateau else 0.0
    L_c = x_c / oma
    if not plateau:
        Tc_impl = T0 + d_yr * L_c
    elif L_c < x_p:
        Tc_impl = T0 - d_yr * x_p * math.log(1.0 - L_c / x_p)
    else:
        Tc_impl = INF
    step = sw['step']
    containment = (sw['learning_mode'] == 'containment')
    learn_count = sw['learn_count']
    phi_const = sw['phi_const']
    learn_fatal = 1.0 if sw.get('learn_from_fatal') else 0.0
    response_off = sw['response_off']
    events_on = not (response_off or sw['events_off'])

    tau_d0 = P['tau_d0_days']
    b_d = P['b_d']
    tau_cap_exp = log2(TAU_D_CAP_D / tau_d0)     # dwell exponent cap: b_d x_def beyond it means tau_d = cap
    e1 = P['e1']
    phi_acc_max = P['phi_acc_max']
    delta_e2 = P['delta_e2']
    r0 = P['r0']
    m = P['m']
    m_eff = 1.0 + (m - 1.0) * (1.0 - P['rho'])
    pi_det = P['pi_det']
    q2_max = P['q2_max']
    delta_2 = P['delta_2']
    x_open_lag = P['L_open'] / d_yr
    phi_mis = 1.0 if phi_const else P['phi_mis']
    phi_conc = 1.0 if phi_const else P['phi_conc']
    x_conc_lag = P['L_conc'] / d_yr              # lock-in gate offset in doublings above x_c
    s_align = P['s_align']
    s_regime = P['s_regime']
    g = P['g']
    q0 = P['q0']
    eta = P['eta']
    f_min = P['f_min']
    w = P['w']
    safety_share = P['safety_share']
    w1 = P['w1']
    smm1 = P['share_mult_max'] - 1.0
    s_effort_max = P['s_effort_max']
    m2 = P['m2']
    L_nat = P['L_nat']
    e_resp = P['e']
    e_t2 = 1.0 - (1.0 - e_resp) ** m2          # tier-2 binding jump: e compounded over m2 HF-class units
    R_max = P['R_max']
    beta_p = P['beta_p']
    gamma_d = P['gamma_d']
    pi_uni = P['pi_uni']
    e_uni = P['e_uni']
    c_cov = P['c_cov']
    D_coord = P['D_coord']
    s_halt = P['s_halt']
    kappa_s = P['kappa_s']
    lam_mis = P['lam_mis']
    lam_conc = P['lam_conc']
    c_conc = P['c_conc']
    sal2_sat = P['sal2_sat']
    rp = min(max(P['race_persistence'], 1e-12), 1.0)
    pi0 = -math.log(rp) / 10.0
    ccap = min(max(P['c_cap'], 0.0), 1.0 - 1e-12)
    pi1 = -math.log(1.0 - ccap) / P['L_int']
    q_v_cl = 1.0 - (1.0 - q0) ** m_eff
    r_start = r0 if r0 < P['r_max'] else P['r_max']   # the rate starts at min(r0, r_max): r_max is a cap
    ratio = P['r_max'] / r_start
    t_g = Tc_impl if sw['g_clock'] == 'Tc' else T_START
    tau_h = P['tau_h']
    h_sal = P['h_sal']
    h_uni_yr = P['h_uni_weeks'] / 52.0
    decR = [2.0 ** (-c / tau_h) for c in CLS_DT]
    decS = [2.0 ** (-c / h_sal) for c in CLS_DT]
    decUh = [2.0 ** (-0.5 * c / h_uni_yr) for c in CLS_DT]                   # decay over half an interval
    IUh = [(1.0 - a) * h_uni_yr / (LN2 * 0.5 * c) for a, c in zip(decUh, CLS_DT)]   # mean of 2^(-s/h) on a half
    R_floor_after_pause = e_resp if e_resp < R_max else R_max
    share_base = safety_share if safety_share < SHARE_MAX else SHARE_MAX
    sfac_base = share_base / SIGMA_REF
    if sfac_base > s_effort_max:
        sfac_base = s_effort_max
    if learn_count == 'sampled':
        rng_l = random.Random(hash(rng_x.getstate()[1]))   # separate stream: U1/U4 stay aligned
    else:
        rng_l = None

    # ---- initial state ----
    if plateau:
        log2H = x_p * (1.0 - exp(-(T_START - T0) * inv_dxp))
    else:
        log2H = (T_START - T0) / d_yr
    x0 = oma * log2H
    u_x0 = 0.5 * (1.0 + erf(bc2 * (x0 - x_c)))
    Sal2 = 0.0
    if sw['init_post_hf']:
        R = 0.0
        Sal = 2.0 ** (-(T_START - T_HF) / h_sal)
        D = e_uni                           # unilateral deficit 1 - m_uni: the 2026 pauses
        Lp = L_nat
        seed_l = rand()                     # one uniform reserved for the L_nat' redraw
        hf_binds = rand() < P['p_bind_first']   # does the HF incident lead to a binding measure at all?
        if Lp < T_START - T_HF:
            rng_ln = random.Random(seed_l)
            tries = 0
            while Lp < T_START - T_HF and tries < 1000:
                Lp = sample(sw['L_nat_prior'], rng_ln)
                tries += 1
            if Lp < T_START - T_HF:
                Lp = T_START - T_HF
        # (t_apply, kind, jump size); kind 0 = binding jump, 1 = tier-1 salience
        pending = [(T_HF + Lp, 0, e_resp)] if hf_binds else []
        N_learn = N_LEARN0
    else:
        R = 0.0
        Sal = 0.0
        D = 0.0
        pending = []
        N_learn = 0.0
    # learning exponent Q (per-incident, leaky): Q0 = eta_eff(T_START) N_LEARN0
    sh0 = safety_share * (1.0 + smm1 * (1.0 - exp(-w1 * Sal)))
    if sh0 > SHARE_MAX:
        sh0 = SHARE_MAX
    eta0 = eta * sh0 / SIGMA_REF
    if eta0 > ETA_EFF_MAX:
        eta0 = ETA_EFF_MAX
    if learn_count == 'sampled':
        Q = N_learn * (-math.log(1.0 - eta0)) if eta0 < 1.0 else 0.0
    else:
        Q = N_learn * eta0
    Q_hist = [Q]
    Q_seen = 0.0                            # largest learning exponent already in effect (monotone)
    rate_prev = r_start * (1.0 - D) * pi_det * (q_v_cl + (1.0 - q_v_cl) * (1.0 - w))
    qrate_prev = 0.0 if learn_count == 'sampled' else rate_prev * eta0
    next_due = min([it[0] for it in pending]) if pending else INF
    jp = 0
    paused = False
    pause_until = 0.0
    S = 1.0
    acc_E1 = 0.0
    acc_E2f = 0.0                           # first-event convention (diagnostic lower bound)
    P_safe = 0.0
    ch0 = ch1 = ch2 = 0.0
    Lam_U = 0.0
    K = 0.0
    acc_sat = 0.0
    Psafe_Tc = 0.0 if Tc_impl <= T_START else None
    t_cross = None
    n_vis = 0
    n_unc = 0
    n_tier2 = 0
    n_pause = 0
    n_uni = 0
    t_first_binding = None
    t_first_pause = None
    t_first_t2 = None
    N50 = None
    phi_end = 1.0
    yE1 = [0.0] * NY
    yE2 = [0.0] * NY
    yE2f = [0.0] * NY
    ySafe = [0.0] * NY
    yR = [0.0] * NY
    yRr = [0.0] * NY
    yLamU = [0.0] * NY
    yK = [0.0] * NY
    yCh = [None] * NY
    yCh2 = [None] * NY
    r_root = 0.0
    k_stop = NB
    if diag:
        dS = [0.0] * 4
        dSafe = [0.0] * 4
        dSal_al = [0.0] * 4
        dSrg = [0.0] * 4
        dSrk = [0.0] * 4
        dSh = [0.0] * 4
        dPause = [0.0] * 4
        dBm = [0.0] * 4
        dBs = [0.0] * 4
        dBe = [0.0] * 4
        dBf = [0.0] * 4
        dBd = [0.0] * 4
        dSal = [0.0] * 4
        dSal2 = [0.0] * 4
        dUni = [0.0] * 4
        ps_al = ps_rg = ps_rk = ps_h = ps_paused = ps_pre = ps_pre_al = ps_pre_reg = 0.0

    for k in range(1, NB):
        t_prev = grid[k - 1]
        dt = DT_[k]
        tm = TM_[k]
        cls = DCLS_[k]
        # 1. due events at the start of the interval, pause expiry
        if pending and next_due <= t_prev:
            keep = []
            next_due = INF
            for item in pending:
                if item[0] <= t_prev:
                    if item[1] == 0:
                        R = R + item[2] * (R_max - R)
                        if t_first_binding is None:
                            t_first_binding = item[0]
                    else:
                        Sal += 1.0
                        if rand() < pi_uni:
                            D = D + e_uni * (1.0 - D)
                            n_uni += 1
                else:
                    keep.append(item)
                    if item[0] < next_due:
                        next_due = item[0]
            pending = keep
        if paused and t_prev >= pause_until:
            paused = False
            if R < R_floor_after_pause:
                R = R_floor_after_pause
        Pi = (1.0 - c_cov) if paused else 1.0
        # 2. capability (exact on the interval; plateau or unbounded)
        mlt = (1.0 - gamma_d * R) * Pi
        if plateau:
            e_h = exp(-mlt * dt * inv_dxp * 0.5)
            gap = x_p - log2H
            lmid = x_p - gap * e_h
            l_new = x_p - gap * e_h * e_h
        else:
            dl = dt * mlt / d_yr
            lmid = log2H + 0.5 * dl
            l_new = log2H + dl
        x_def = oma * lmid
        if t_cross is None and x_def >= x_c:
            t_cross = t_prev
        # 3. safety share (saturating salience response) and learning factor (dwell-lagged)
        sat_part = 1.0 - exp(-w1 * Sal)
        share = safety_share * (1.0 + smm1 * sat_part)
        b_share = share > SHARE_MAX_
        if b_share:
            share = SHARE_MAX_
        bx_d = b_d * x_def
        b_dwell = bx_d >= tau_cap_exp
        if b_dwell:
            tau_d = TAU_D_CAP_D_
        else:
            tau_d = tau_d0 * 2.0 ** bx_d
        eta_eff = eta * share / SIGMA_REF_
        b_eta = eta_eff > ETA_EFF_MAX_
        if b_eta:
            eta_eff = ETA_EFF_MAX_
        if eta > 0.0:
            t_lag = tm - tau_d / 365.0
            if t_lag >= t_prev:
                Q_lag = Q + (t_lag - t_prev) * qrate_prev
            elif t_lag <= grid[0]:
                Q_lag = Q_hist[0]
            else:
                j = jp if jp < k - 1 else k - 2
                while j < k - 2 and grid[j + 1] <= t_lag:
                    j += 1
                while j > 0 and grid[j] > t_lag:
                    j -= 1
                jp = j
                g0 = grid[j]
                Q_lag = Q_hist[j] + (Q_hist[j + 1] - Q_hist[j]) * (t_lag - g0) / (grid[j + 1] - g0)
            if Q_lag < Q_seen:              # a lengthening dwell time never un-learns a closed failure mode
                Q_lag = Q_seen
            else:
                Q_seen = Q_lag
            L_learn = f_min + (1.0 - f_min) * exp(-Q_lag)
        else:
            L_learn = 1.0
        if containment:
            L_haz = 1.0
            xce = x_c - log2(L_learn if L_learn > 1e-300 else 1e-300) / b_c
        else:
            L_haz = L_learn
            xce = x_c
        # 4. lethality gates
        if step:
            u = 1.0 if x_def >= xce else 0.0
            g_al = 1.0 if x_def >= x_c else 0.0
        else:
            u = 0.5 * (1.0 + erf(bc2 * (x_def - xce)))
            g_al = u if xce == x_c else 0.5 * (1.0 + erf(bc2 * (x_def - x_c)))
        resp_c = 1.0 - beta_p * R
        u_eff = u * L_haz * resp_c
        phi_mid = 1.0 if phi_const else phi_acc_max * 0.5 * (1.0 + erf(bc2 * (lmid - x_c - delta_e2)))
        # 5. root-cause cluster rate (capped logistic growth, restrained), exact unilateral average
        if g > 0.0 and tm > t_g:
            Gr = ratio / (1.0 + (ratio - 1.0) * exp(-g * (tm - t_g)))
        else:
            Gr = 1.0
        r_base = r_start * Gr * (1.0 - R) * Pi
        a_h = decUh[cls]
        I_h = IUh[cls]
        D_mid = D * a_h
        D_avg = 0.5 * I_h * (D + D_mid)
        r_root = r_base * (1.0 - D_avg)
        # 6. cluster quantities and channel hazards
        if u_eff > 0.0:
            u_cl = 1.0 - (1.0 - u_eff) ** m_eff
            p_cl = 1.0 - (1.0 - u_eff * e1) ** m_eff
        else:
            u_cl = 0.0
            p_cl = 0.0
        h_acc = r_root * p_cl
        if lam_mis > 0.0:
            h_mis = lam_mis * 0.5 * (1.0 + erf(bc2 * (lmid - x_open_lag - x_c))) * resp_c
        else:
            h_mis = 0.0
        if lam_conc > 0.0:
            cfac = 1.0 - c_conc * R
            if cfac < 0.0:
                cfac = 0.0
            h_conc = lam_conc * 0.5 * (1.0 + erf(bc2 * (lmid - x_c - x_conc_lag))) * cfac
        else:
            h_conc = 0.0
        sfac = share / SIGMA_REF_
        b_sfac = sfac > s_effort_max
        if b_sfac:
            sfac = s_effort_max
        s_al = s_align * sfac * g_al
        s_rk = s_regime * kappa_s * R
        s_h = s_halt if paused else 0.0
        s = s_al + s_regime + s_rk + s_h
        # 6b. visible losses at the interval midpoint: K ~ Poisson(nu dt) by inverting U1
        if events_on:
            U1 = rand()
            nu_pub = r_root * q_v_cl * (1.0 - u_cl)
            nu_unc = r_root * (u_cl - p_cl)
            nu = nu_pub + nu_unc
            if nu > 0.0:
                lam_ev = nu * dt
                nev = pois(lam_ev, U1) if U1 > exp(-lam_ev) else 0   # the zero case of pois, inlined
            else:
                nev = 0
            if nev:
                q2 = q2_max * 0.5 * (1.0 + erf(bc2 * (x_def - x_c + delta_2)))
                n_vis += nev
                n_unc += nev * nu_unc / nu      # expected count (a diagnostic; no uniform is drawn)
                D_new = D_mid
                for _ev in range(nev):
                    if rand() < q2:
                        Sal += m2
                        Sal2 += 1.0
                        pending.append((tm + L_nat, 0, e_t2))
                        if tm + L_nat < next_due:
                            next_due = tm + L_nat
                        n_tier2 += 1
                        if t_first_t2 is None:
                            t_first_t2 = tm
                        if rand() < pi_uni:
                            D_new = D_new + e_uni * (1.0 - D_new)
                            n_uni += 1
                    else:
                        pending.append((tm + tau_d / 365.0, 1, 0.0))
                        if tm + tau_d / 365.0 < next_due:
                            next_due = tm + tau_d / 365.0
                if D_new != D_mid:          # a unilateral pause from the midpoint on, within this interval
                    D_mid = D_new
                    D_avg = 0.5 * I_h * (D + D_mid)
                    r_root = r_base * (1.0 - D_avg)
                    h_acc = r_root * p_cl
        h_E1 = h_acc + h_mis + h_conc
        h_tot = h_E1 + s
        # 7. exact competing-risks update on the interval
        if h_tot > 0.0:
            S_new = S * exp(-h_tot * dt)
            frac = (S - S_new) / h_tot
            acc_E1 += h_E1 * frac
            acc_E2f += (phi_mid * h_acc + phi_mis * h_mis + phi_conc * h_conc) * frac
            ch0 += h_acc * frac
            ch1 += h_mis * frac
            ch2 += h_conc * frac
            P_safe += s * frac
            if lmid > C_SAT_:
                acc_sat += h_E1 * frac
        else:
            S_new = S
            frac = S * dt
        if diag:
            pk = PERIOD_[k]
            dS[pk] += frac
            dSafe[pk] += s * frac
            dSal_al[pk] += s_al * frac
            dSrg[pk] += s_regime * frac
            dSrk[pk] += s_rk * frac
            dSh[pk] += s_h * frac
            dUni[pk] += D_avg * frac
            if paused:
                dPause[pk] += frac
                ps_paused += s * frac
            if smm1 > 0.0 and sat_part >= 0.9:
                dBm[pk] += frac
            if b_share:
                dBs[pk] += frac
            if b_eta and eta > 0.0:
                dBe[pk] += frac
            if b_sfac and s_align > 0.0:
                dBf[pk] += frac
            if b_dwell:
                dBd[pk] += frac
            dSal[pk] += Sal * frac
            dSal2[pk] += Sal2 * frac
            ps_al += s_al * frac
            ps_rg += s_regime * frac
            ps_rk += s_rk * frac
            ps_h += s_h * frac
            if t_cross is None:
                ps_pre += s * frac
                ps_pre_al += s_al * frac
                ps_pre_reg += (s_regime + s_rk + s_h) * frac
        S = S_new
        Lam_U += r_root * u_cl * dt
        K += r_root * dt
        # 8. coordinated pause (sampled on the surviving path; takes effect from the next interval)
        if events_on:
            U4 = rand()
            if not paused:
                h_pause = pi0 + pi1 * (Sal2 if Sal2 < sal2_sat else sal2_sat)
                if h_pause > 0.0 and U4 < 1.0 - exp(-h_pause * dt):
                    paused = True
                    pause_until = tm + D_coord
                    n_pause += 1
                    if t_first_pause is None:
                        t_first_pause = tm
        # 9. clocks
        # learned clusters: public, uncontainable-but-not-fatal, or not dismissed; fatal clusters excluded on the
        # surviving path unless learn_from_fatal (closed-form checks)
        rate_prev = r_root * pi_det * (q_v_cl * (1.0 - u_cl) + (u_cl - (1.0 - learn_fatal) * p_cl)
                                       + (1.0 - q_v_cl) * (1.0 - u_cl) * (1.0 - w))
        N_learn += dt * rate_prev
        if eta > 0.0:
            if learn_count == 'mean':
                qrate_prev = eta_eff * rate_prev
                Q += dt * qrate_prev
            elif learn_count == 'lograte':
                qrate_prev = (1.0 - exp(-eta_eff)) * rate_prev
                Q += dt * qrate_prev
            else:
                nl = pois(dt * rate_prev, rng_l.random())
                if nl:
                    Q += nl * (-math.log(1.0 - eta_eff))
        Q_hist.append(Q)
        log2H = l_new
        R *= decR[cls]
        Sal *= decS[cls]
        Sal2 *= decS[cls]
        D = D_mid * a_h
        # 10. record; E2 = population < 5,000 by t_k: accidental E1 worlds progress with capability
        if not phi_const:
            phi_end = phi_acc_max * 0.5 * (1.0 + erf(bc2 * (log2H - x_c - delta_e2)))
        c2a = ch0 * phi_end
        c2b = phi_mis * ch1
        c2c = phi_conc * ch2
        acc_E2 = c2a + c2b + c2c
        sumE1[k] += acc_E1
        sumE2[k] += acc_E2
        sumSafe[k] += P_safe
        if Psafe_Tc is None and grid[k] >= Tc_impl:
            Psafe_Tc = P_safe
        yi = YPOS_[k]
        if yi >= 0:
            yE1[yi] = acc_E1
            yE2[yi] = acc_E2
            yE2f[yi] = acc_E2f
            ySafe[yi] = P_safe
            yR[yi] = R
            yRr[yi] = r_root
            yLamU[yi] = Lam_U
            yK[yi] = K
            yCh[yi] = (ch0, ch1, ch2)
            yCh2[yi] = (c2a, c2b, c2c)
            if yi == 7:
                N50 = N_learn
        if S < 1e-9:
            k_stop = k
            break

    if k_stop < NB:                      # early stop: freeze the accumulators; capability keeps growing
        for kk in range(k_stop + 1, NB):  # (unrestrained: the post-E1 world) for the E2 progression
            dtk = DT[kk]
            if plateau:
                log2H = x_p - (x_p - log2H) * exp(-dtk * inv_dxp)
            else:
                log2H += dtk / d_yr
            if not phi_const:
                phi_end = phi_acc_max * 0.5 * (1.0 + erf(bc2 * (log2H - x_c - delta_e2)))
            c2a = ch0 * phi_end
            c2b = phi_mis * ch1
            c2c = phi_conc * ch2
            acc_E2 = c2a + c2b + c2c
            sumE1[kk] += acc_E1
            sumE2[kk] += acc_E2
            sumSafe[kk] += P_safe
            yi = YPOS[kk]
            if yi >= 0:
                yE1[yi] = acc_E1
                yE2[yi] = acc_E2
                yE2f[yi] = acc_E2f
                ySafe[yi] = P_safe
                yR[yi] = R
                yRr[yi] = r_root
                yLamU[yi] = Lam_U
                yK[yi] = K
                yCh[yi] = (ch0, ch1, ch2)
                yCh2[yi] = (c2a, c2b, c2c)
        if N50 is None:
            N50 = N_learn
        if Psafe_Tc is None and Tc_impl <= 2100.0:
            Psafe_Tc = P_safe

    rec = {'E1': yE1, 'E2': yE2, 'E2f': yE2f, 'safe': ySafe, 'R': yR, 'rroot': yRr, 'LamU': yLamU, 'K': yK,
           'ch': yCh, 'ch2': yCh2, 'Tc': Tc_impl, 'x_c': x_c, 'Psafe_Tc': Psafe_Tc, 'u_x0': u_x0,
           'N50': N50, 'n_vis': n_vis, 'n_unc': n_unc, 'n_tier2': n_tier2, 'n_pause': n_pause, 'n_uni': n_uni,
           't_bind': t_first_binding, 't_pause': t_first_pause, 't_t2': t_first_t2, 't_cross': t_cross,
           't_stop': grid[k_stop] if k_stop < NB else None, 'acc_sat': acc_sat, 'stopped': k_stop < NB}
    if diag:
        rec['diag'] = {'S': dS, 'safe': dSafe, 'al': dSal_al, 'rg': dSrg, 'rk': dSrk, 'sh': dSh,
                       'pause': dPause, 'bm': dBm, 'bs': dBs, 'be': dBe, 'bf': dBf, 'bd': dBd, 'sal': dSal,
                       'sal2': dSal2,
                       'uni': dUni, 'ps': (ps_al, ps_rg, ps_rk, ps_h), 'ps_paused': ps_paused, 'ps_pre': ps_pre,
                       'ps_pre_al': ps_pre_al, 'ps_pre_reg': ps_pre_reg}
    return rec


# ----------------------------------------------------------------------------
# Scenario runner and aggregation
# ----------------------------------------------------------------------------
def run_scenario(sc, N, seed, snaps=()):
    """Runs N draws; snaps lists draw counts at which the mean curves are also stored (for the
    common-random-number subsets used by the sensitivity blocks)."""
    sw = scenario_switches(sc)
    sums = new_sums()
    recs = []
    params = []
    snap = {}
    t0 = time.time()
    for j in range(N):
        P = draw_params(sc['override_key'], sc['stream'], j, seed)
        rng_x = random.Random('%d:%s:%d:x' % (seed, sc['stream'], j))
        recs.append(run_path(P, sw, rng_x, sums, diag=True))
        params.append(P)
        if (j + 1) in snaps:
            snap[j + 1] = curves_of(sums, j + 1)
    el = time.time() - t0
    out = {'key': sc['key'], 'label': sc['label'], 'N': N, 'sw': sw, 'recs': recs, 'params': params,
           'time': el, 'snap': snap}
    out.update(curves_of(sums, N))
    return out


def curves_of(sums, N):
    return {'E1': [v / N for v in sums[0]], 'E2': [v / N for v in sums[1]], 'safe': [v / N for v in sums[2]]}


def run_draws(N, seed, param_fn, sw, stream='S1'):
    sums = new_sums()
    recs = []
    for j in range(N):
        recs.append(run_path(param_fn(j), sw, random.Random('%d:%s:%d:x' % (seed, stream, j)), sums))
    out = {'N': N, 'recs': recs}
    out.update(curves_of(sums, N))
    return out


def t_quantile(curve, q):
    if curve[-1] < q:
        return None
    for k in range(1, NB):
        if curve[k] >= q:
            c0 = curve[k - 1]
            c1 = curve[k]
            if c1 <= c0:
                return GRID[k]
            return GRID[k - 1] + (q - c0) / (c1 - c0) * (GRID[k] - GRID[k - 1])
    return None


def at(curve, Y):
    return curve[YIDX[Y]]


def paired(ra, rb, key, Y):
    """Mean and standard error of rb - ra over common draws at year Y."""
    yi = YP[Y]
    d = [b[key][yi] - a[key][yi] for a, b in zip(ra, rb)]
    return mean(d), se_of(d)


# ----------------------------------------------------------------------------
# v0.1 closed forms (paper 4.1; model/model.py), re-implemented inline
# ----------------------------------------------------------------------------
def v01_Tc(Hc, d_months):
    return T0 + (d_months / 12.0) * math.log2(Hc / H0_HOURS)


def v01_P(Y, Tc, p, grow, r0=4.0):
    t = max(0.0, Y - Tc)
    lam = r0 * p * t if not grow else r0 * p * (2.0 ** t - 1.0) / LN2
    return 1.0 - math.exp(-lam)


def v01_median(p, grow, Tc=S0_TC, r0=4.0):
    if not grow:
        return Tc + LN2 / (r0 * p)
    return Tc + math.log2(1.0 + LN2 ** 2 / (r0 * p))


def v01_counter(p0, r0=4.0):
    return 1.0 - math.exp(-r0 * p0 / LN2)


def run_s0_cells(seed):
    """The six S0 cells at T_c = 2028.5, plus the two paper 5.2 bracket ends per cell (T_c = 2027.3 and
    2030.5, as d = 3 months / H_c = 167 h and d = 7 months / H_c = 2,000 h give them); the engine keeps
    d = 4.3 months and moves the threshold through Hc_override_hours, which leaves the closed form unchanged."""
    sc = SCENARIOS[0]
    sw = scenario_switches(sc)
    out = []
    for ci, c in enumerate(S0_CELLS):
        P = draw_params('S0', 'S0', ci, seed, extra={'e1': c['p'], 'g': (LN2 if c['grow'] else 0.0)})
        sums = new_sums()
        rec = run_path(P, sw, random.Random('%d:S0:%d:x' % (seed, ci)), sums)
        br = []
        for d_m, Hc in S0_BRACKET_CELLS:
            Tc_b = v01_Tc(Hc, d_m)
            sw_b = dict(sw)
            sw_b['Hc_override_hours'] = H0_HOURS * 2.0 ** ((Tc_b - T0) * 12.0 / 4.3)
            sums_b = new_sums()
            run_path(P, sw_b, random.Random('%d:S0:%d:x' % (seed, ci)), sums_b)
            br.append({'Tc': Tc_b, 'E1': sums_b[0]})
        out.append({'cell': c, 'E1': sums[0], 'E2': sums[1], 'safe': sums[2], 'rec': rec, 'P': P, 'br': br})
    return out


def s0_prior_averaged():
    cells = []
    for d in S0_GRID_36['d_months']:
        for Hc in S0_GRID_36['Hc_hours']:
            Tc = v01_Tc(Hc, d)
            for p in S0_GRID_36['p']:
                for grow in (False, True):
                    cells.append((Tc, p, grow))
    n = len(cells)

    def F(Y):
        return sum(v01_P(Y, Tc, p, grow) for (Tc, p, grow) in cells) / n
    py = dict((Y, F(Y)) for Y in YEARS)
    ts = [2027.0 + 0.01 * i for i in range(7301)]
    Fs = [F(t) for t in ts]
    qs = []
    for q in (0.1, 0.5, 0.9):
        val = None
        for i in range(1, len(ts)):
            if Fs[i] >= q:
                val = ts[i - 1] + (q - Fs[i - 1]) / (Fs[i] - Fs[i - 1]) * 0.01 if Fs[i] > Fs[i - 1] else ts[i]
                break
        qs.append(val)
    return {'P': py, 'q': qs, 'n': n}


# ----------------------------------------------------------------------------
# Outside view
# ----------------------------------------------------------------------------
def ov_convert(p_src, y_src):
    if y_src is None or abs(y_src - 2100.0) < 1e-9:
        return p_src
    frac = (2100.0 - T_START) / (y_src - T_START)
    return 1.0 - (1.0 - p_src) ** frac


def gmean(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def profile_f(points, Y):
    pts = [(T_START, 0.0)] + list(points) + [(2100.0, 1.0)]
    if Y <= pts[0][0]:
        return 0.0
    for (a, fa), (b, fb) in zip(pts[:-1], pts[1:]):
        if a <= Y <= b:
            return fa + (fb - fa) * (Y - a) / (b - a)
    return 1.0


def mass_share_2050(curve):
    lam100 = -math.log(1.0 - at(curve, 2100))
    if lam100 <= 0:
        return float('nan')
    return -math.log(1.0 - at(curve, 2050)) / lam100


def cumhaz(curve, Y=2100):
    v = at(curve, Y)
    return -math.log(1.0 - v) if v < 1.0 else float('inf')


def hn_multipliers(S1, blocks):
    """Model-derived human-nature multipliers on the century hazard, from the block-level variants (same
    S1 draws): A_pess = Lambda(S2 human nature only)/Lambda(S1), A_opt = Lambda(S3 human nature only)/
    Lambda(S1), and the size of the modelled human response A_resp = Lambda(S1)/Lambda(zero response).
    Lambda = -ln(1 - P(E by 2100)) of the mean curve (safe exits included)."""
    s1b = S1['snap'][blocks[0]['N']]
    b = dict((x['id'], x) for x in blocks)
    out = {}
    for d in ('E1', 'E2'):
        l1 = cumhaz(s1b[d])
        out[d] = {'S1': l1, 'S2hn': cumhaz(b['S2hn'][d]), 'S3hn': cumhaz(b['S3hn'][d]), 'zero': cumhaz(b['zero'][d]),
                  'pess': cumhaz(b['S2hn'][d]) / l1 if l1 > 0 else float('nan'),
                  'opt': cumhaz(b['S3hn'][d]) / l1 if l1 > 0 else float('nan'),
                  'resp': l1 / cumhaz(b['zero'][d]) if cumhaz(b['zero'][d]) > 0 else float('nan')}
    return out


def outside_view(S1, hnm):
    pools = {}
    for d, key in (('E1', 'E1_pool'), ('E2', 'E2_pool')):
        conv = [ov_convert(a['p_src'], a['y_src']) for a in OUTSIDE_VIEW[key]]
        pools[d] = {'conv': conv, 'pool': gmean(conv)}
    rows = {}
    for d in ('E1', 'E2'):
        lam = -math.log(1.0 - pools[d]['pool'])
        prof = OUTSIDE_VIEW['profiles'][d]
        for mlt in OUTSIDE_VIEW['multipliers']:
            rows[(d, mlt['id'])] = [1.0 - math.exp(-mlt['A'] * lam * profile_f(prof, Y)) for Y in YEARS]
        for key in ('pess', 'opt'):
            A = hnm[d][key]
            rows[(d, 'hn_' + key)] = [1.0 - math.exp(-A * lam * profile_f(prof, Y)) for Y in YEARS]
        curve = S1[d]
        lamS = -math.log(1.0 - at(curve, 2100))
        fS = [(-math.log(1.0 - at(curve, Y)) / lamS) if lamS > 0 else 0.0 for Y in YEARS]
        rows[(d, 'S1')] = [1.0 - math.exp(-lam * f) for f in fS]
        pools[d]['fS1_2050'] = fS[YP[2050]]
    return pools, rows


# ----------------------------------------------------------------------------
# Closed-form checks run through the production engine
# ----------------------------------------------------------------------------
def engine_check_runs(seed):
    sw_s = dict(SWITCHES_DEFAULT)
    sw_s.update({'step': True, 'response_off': True, 'init_post_hf': False, 'phi_const': True,
                 'Hc_override_hours': H0_HOURS * 2.0 ** ((T_START - T0) * 12.0 / 4.3)})
    sw_s['L_nat_prior'] = PARAM_BY_NAME['L_nat']['prior']
    sw = dict(sw_s)
    sw['learn_from_fatal'] = True      # every cluster teaches: the time-based halving of paper sec. 5.4
    common = {'d_months': 4.3, 'a': 0.0, 'g': 0.0, 'r_max': 1e12, 'm': 1.0, 'rho': 0.0, 'f_min': 0.0,
              'pi_det': 1.0, 'w': 0.0, 'q0': 1.0, 'safety_share': SIGMA_REF, 'w1': 0.0, 'share_mult_max': 1.0, 's_effort_max': 1.0,
              'tau_d0_days': 1e-6, 'b_d': 0.0, 's_align': 0.0, 's_regime': 0.0, 's_halt': 0.0, 'lam_mis': 0.0,
              'x_plateau': INF,
              'lam_conc': 0.0, 'e': 0.0, 'R_max': 0.0, 'beta_p': 0.0, 'gamma_d': 0.0, 'kappa_s': 0.0,
              'pi_uni': 0.0, 'e_uni': 0.0, 'race_persistence': 1.0, 'c_cap': 0.0, 'q2_max': 0.0,
              'sal2_sat': 1.0}
    res = {'identity': [], 'counter': [], 'surv': []}
    T_h = 2100.0 - T_START
    p_id, eta_id = 0.05, 0.10
    for r0 in (1.0, 4.0, 10.0):
        P = draw_params(None, 'CHK', 0, seed, extra=dict(common, r0=r0, e1=p_id, eta=eta_id))
        sums = new_sums()
        run_path(P, sw, random.Random('chk'), sums)
        closed_T = 1.0 - math.exp(-(p_id / eta_id) * (1.0 - math.exp(-eta_id * r0 * T_h)))
        res['identity'].append((r0, sums[0][-1], closed_T, 1.0 - math.exp(-p_id / eta_id)))
        # survivor-conditioned clock (default): dQ/dt = eta r (1 - p e^-Q), so
        # Lambda(T) = (1/eta)[eta r T - ln(p + (1 - p) e^(eta r T))] and P(ever) = 1 - (1 - p)^(1/eta)
        sums = new_sums()
        run_path(P, sw_s, random.Random('chk'), sums)
        x = eta_id * r0 * T_h
        lam_T = (x - (x + math.log((1.0 - p_id) + p_id * math.exp(-x)))) / eta_id
        res['surv'].append((r0, sums[0][-1], 1.0 - math.exp(-lam_T), 1.0 - (1.0 - p_id) ** (1.0 / eta_id)))
    for p0 in (0.01, 0.05, 0.20):
        P = draw_params(None, 'CHK', 0, seed, extra=dict(common, r0=4.0, e1=p0, eta=LN2 / 4.0))
        sums = new_sums()
        run_path(P, sw, random.Random('chk'), sums)
        res['counter'].append((p0, sums[0][-1], v01_counter(p0)))
    return res


# ----------------------------------------------------------------------------
# Sensitivity: OAT, block-level variants and the sign check (common random numbers)
# ----------------------------------------------------------------------------
def _fixed_param_runs(S1res, name, values, N, seed, sw):
    base = S1res['params'][:N]
    out = []
    for v in values:
        def pf(j, v=v):
            P = dict(base[j])
            P[name] = v
            return P
        out.append(run_draws(N, seed, pf, sw))
    return out


def run_oat(S1res, N_OAT, seed, qtab):
    sw = scenario_switches(SCENARIOS[1])
    rows = []
    for spec in PARAMS:
        name = spec['name']
        q10, q90 = qtab[name][1], qtab[name][3]
        lo, hi = _fixed_param_runs(S1res, name, (q10, q90), N_OAT, seed, sw)
        sw50, se50 = paired(lo['recs'], hi['recs'], 'E1', 2050)
        d35, _ = paired(lo['recs'], hi['recs'], 'E1', 2035)
        dsafe, _ = paired(lo['recs'], hi['recs'], 'safe', 2100)
        rows.append({'name': name, 'hn': spec['hn'], 'tag': spec['tag'], 'q10': q10, 'q90': q90,
                     'p50_lo': at(lo['E1'], 2050), 'p50_hi': at(hi['E1'], 2050), 'swing': sw50, 'se': se50,
                     'd35': d35, 'dsafe': dsafe})
    base = S1res['recs'][:N_OAT]
    return rows, mean(r['E1'][YP[2050]] for r in base)


def run_blocks(S1res, N, seed):
    sw0 = scenario_switches(SCENARIOS[1])
    base_params = S1res['params'][:N]
    out = []
    for v in BLOCK_VARIANTS:
        sw = dict(sw0)
        sw.update(v.get('sw', {}))
        if 'key' in v:
            def pf(j, v=v):
                return draw_params(v['key'], 'S1', j, seed, extra=v.get('extra'), only=v.get('only'))
        else:
            def pf(j, v=v):
                P = dict(base_params[j])
                P.update(v.get('extra', {}))
                return P
        res = run_draws(N, seed, pf, sw)
        res['id'] = v['id']
        res['label'] = v['label']
        out.append(res)
    return out


def run_s3_variants(S3res, N, seed):
    sw0 = scenario_switches([sc for sc in SCENARIOS if sc['key'] == 'S3'][0])
    base_params = S3res['params'][:N]
    out = []
    for v in S3_VARIANTS:
        sw = dict(sw0)
        sw.update(v.get('sw', {}))

        def pf(j, v=v):
            P = dict(base_params[j])
            P.update(v.get('extra', {}))
            return P
        res = run_draws(N, seed, pf, sw, stream='S3')
        res['id'] = v['id']
        res['label'] = v['label']
        out.append(res)
    return out


def run_sign(S1res, N, seed, qtab):
    sw = scenario_switches(SCENARIOS[1])
    rows = []
    for name in SIGN_PARAMS:
        lo_v, hi_v = qtab[name][0], qtab[name][4]
        lo, hi = _fixed_param_runs(S1res, name, (lo_v, hi_v), N, seed, sw)
        rows.append({'name': name, 'lo': lo_v, 'hi': hi_v,
                     'p50': (at(lo['E1'], 2050), at(hi['E1'], 2050)),
                     'p100': (at(lo['E1'], 2100), at(hi['E1'], 2100)),
                     'd35': paired(lo['recs'], hi['recs'], 'E1', 2035),
                     'd50': paired(lo['recs'], hi['recs'], 'E1', 2050),
                     'd100': paired(lo['recs'], hi['recs'], 'E1', 2100),
                     'dE2': paired(lo['recs'], hi['recs'], 'E2', 2100),
                     'dE2f': paired(lo['recs'], hi['recs'], 'E2f', 2100)})
    return rows


def run_hn_effects(S1res, N, seed, qtab, sign_rows, n_sign):
    """Measured effect of each human-nature parameter: P(E1 by Y) at its prior q99 minus at its q01, the
    other parameters keeping their S1 draws (common random numbers on the first N S1 draws), for Y in
    HNE_YEARS. Parameters already in the sign check reuse those runs (n_sign draws per side)."""
    sw = scenario_switches(SCENARIOS[1])
    sign_by = dict((r['name'], r) for r in sign_rows)
    out = {}
    for spec in PARAMS:
        if not spec['hn']:
            continue
        name = spec['name']
        if name in sign_by:
            r = sign_by[name]
            d = {2035: r['d35'], 2050: r['d50'], 2100: r['d100']}
            n = n_sign
        else:
            lo, hi = _fixed_param_runs(S1res, name, (qtab[name][0], qtab[name][4]), N, seed, sw)
            d = dict((Y, paired(lo['recs'], hi['recs'], 'E1', Y)) for Y in HNE_YEARS)
            n = N
        out[name] = {'lo': qtab[name][0], 'hi': qtab[name][4], 'N': n, 'd': d,
                     'inert': all(abs(d[Y][0]) < INERT_PP for Y in HNE_YEARS)}
    return out


def hn_record_check(hne, default_run):
    """Compares the measured effects (rounded to 0.01 pp) with HN_EFFECT_RECORD and the inert set with
    HN_INERT_REASON. Returns (status text, list of stale names)."""
    if not default_run:
        return 'not compared (non-default seed or draw counts)', []
    stale = []
    for name, e in hne.items():
        cur = tuple(round(100.0 * e['d'][Y][0], 2) for Y in HNE_YEARS)
        rec = HN_EFFECT_RECORD.get(name)
        if rec is None or any(abs(a - b) > 0.0051 for a, b in zip(rec, cur)):
            stale.append(name)
        if e['inert'] != (name in HN_INERT_REASON):
            stale.append(name + ' (inert reason)')
    extra = [n for n in HN_INERT_REASON if n not in hne]
    stale += [n + ' (unknown name)' for n in extra]
    return ('matches this run' if not stale else 'STALE for ' + ', '.join(stale)), stale


def tercile_contrast(S1res):
    """e1-stratified tercile contrast of P(E1 by 2050): within each e1 bin, the top-minus-bottom-tercile mean
    over the parameter's draws; bins weighted by their draw counts. Stratifying on e1 removes the chance
    correlation of each parameter with the dominant lever. Returns {name: (contrast, SE)}."""
    recs = S1res['recs']
    params = S1res['params']
    N = float(len(recs))
    y = [r['E1'][YP[2050]] for r in recs]
    bins = [b['idx'] for b in e1_bins(S1res) if len(b['idx']) >= 6]
    out = {}
    for spec in PARAMS:
        name = spec['name']
        c = 0.0
        v = 0.0
        for idx in bins:
            n = len(idx)
            n3 = n // 3
            order = sorted(idx, key=lambda i: params[i][name])
            lo = [y[i] for i in order[:n3]]
            hi = [y[i] for i in order[n - n3:]]
            wb = n / N
            c += wb * (mean(hi) - mean(lo))
            v += wb * wb * (statistics.variance(hi) / n3 + statistics.variance(lo) / n3)
        out[name] = (c, math.sqrt(v))
    return out


# ----------------------------------------------------------------------------
# params.yaml mirror
# ----------------------------------------------------------------------------
def yaml_check(path):
    try:
        with open(path) as fh:
            text = fh.read()
    except (IOError, OSError) as exc:
        return ['params.yaml not readable (%s); mirror check skipped' % exc]
    missing = [n for n in PARAM_NAMES + [c['name'] for c in CONSTANTS] if n not in text]
    return ['params.yaml does not mention %s' % n for n in missing]


def _yv(v):
    if v is None:
        return 'null'
    if isinstance(v, bool):
        return 'true' if v else 'false'
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if v != v or v in (float('inf'), float('-inf')):
            return '.nan' if v != v else ('.inf' if v > 0 else '-.inf')
        r = repr(v)
        if 'e' in r and '.' not in r.split('e')[0]:
            r = r.replace('e', '.0e')
        return r
    if isinstance(v, dict):
        return '{}'
    if isinstance(v, (list, tuple)):
        return '[]'
    return '"' + str(v).replace('\\', '\\\\').replace('"', '\\"') + '"'


def _scalar(v):
    return not isinstance(v, (dict, list, tuple))


def _flow(v):
    """Flow style for short scalar-only lists and for prior specs; None if block style is better."""
    if isinstance(v, (list, tuple)) and v and all(_scalar(x) for x in v) and len(v) <= 12 \
            and all(not isinstance(x, str) or len(x) < 40 for x in v):
        return '[' + ', '.join(_yv(x) for x in v) + ']'
    if isinstance(v, dict) and 'dist' in v and all(_scalar(x) for x in v.values()):
        return '{' + ', '.join('%s: %s' % (k, _yv(x)) for k, x in v.items()) + '}'
    return None


def _yaml_lines(obj, ind):
    pad = '  ' * ind
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            fl = _flow(v)
            if fl is not None:
                out.append('%s%s: %s' % (pad, k, fl))
            elif isinstance(v, (dict, list, tuple)) and len(v) > 0:
                out.append('%s%s:' % (pad, k))
                out.extend(_yaml_lines(v, ind + 1))
            else:
                out.append('%s%s: %s' % (pad, k, _yv(v)))
    else:
        for v in obj:
            fl = _flow(v)
            if fl is not None:
                out.append('%s- %s' % (pad, fl))
            elif isinstance(v, (dict, list, tuple)) and len(v) > 0:
                sub = _yaml_lines(v, ind + 1)
                out.append('%s- %s' % (pad, sub[0].lstrip()))
                out.extend(sub[1:])
            else:
                out.append('%s- %s' % (pad, _yv(v)))
    return out


def emit_yaml():
    lines = []
    lines.append('')
    lines.append('# ===========================================================================')
    lines.append('# v0.2 Monte Carlo model (model/mc.py), added 2026-10-10. Documentation only:')
    lines.append('# mc.py does not parse this file; at start-up it warns if a parameter or constant')
    lines.append('# name below is missing. Everything above this block is the v0.1 parameter set,')
    lines.append('# unchanged. Regenerate this block with: python3 model/mc.py --emit-yaml')
    lines.append('# Tags: [Fact] directly measured constant, [Inference] distribution read from named')
    lines.append('# analogues, [Assumption] no supporting data. Source keys in rationales resolve in')
    lines.append('# sources: at the end of this file; review F-ids resolve in review_findings:.')
    lines.append('# Priors: lognormal(median, sigma) = median*exp(sigma*N(0,1)); loguniform(lo, hi);')
    lines.append('# beta(a, b); uniform(lo, hi); one_plus_lognormal = 1 + lognormal; point = fixed.')
    lines.append('# ===========================================================================')
    mc = {
        'version': MC_VERSION, 'as_of': AS_OF,
        'run': {'seed': DEFAULT_SEED, 'n_main': N_MAIN_DEFAULT, 'n_var': N_VAR_DEFAULT, 'n_var2': N_VAR2_DEFAULT,
                'n_oat_per_side': N_OAT_DEFAULT, 'n_block': N_BLOCK_DEFAULT, 'n_sign_per_side': N_SIGN_DEFAULT,
                'n_hn_effect_per_side': N_HNE_DEFAULT, 'n_prior_quantile_draws': N_PRIOR_Q,
                'grid': GRID_DOC, 'learning_mode_default': SWITCHES_DEFAULT['learning_mode'],
                'output_years': YEARS},
        'model_summary': MODEL_SUMMARY,
        'design_notes': DESIGN_NOTES,
        'constants': [{'name': c['name'], 'value': c['value'], 'tag': c['tag'],
                       'rationale': c['rationale']} for c in CONSTANTS],
        'switches': [{'name': k, 'default': v, 'meaning': SWITCH_DOCS[k]}
                     for k, v in SWITCHES_DEFAULT.items()],
        'outside_view': {
            'anchors_E1': [dict(a) for a in OUTSIDE_VIEW['E1_pool']],
            'E1_pool_scope': OUTSIDE_VIEW['E1_scope'],
            'not_pooled': [dict(a) for a in OUTSIDE_VIEW['E1_not_pooled']],
            'anchors_E2': [dict(a) for a in OUTSIDE_VIEW['E2_pool']],
            'E2_pool_scope': OUTSIDE_VIEW['E2_scope'],
            'channel_matched_rows': 'blocks 6 and 7 print the model\'s accident-channel P(E1 by Y) (the channel the '
                                    'E1 pool covers) and its misuse + lock-in P(E1 by Y) separately for '
                                    + ', '.join(CHANNEL_MATCH),
            'conversion': 'P_2100 = 1 - (1 - P_src)^((2100 - T_START)/(Y_src - T_START)); no horizon = 2100; the '
                          'whole source probability is assigned to the window after T_START',
            'pooling': 'geometric mean of the converted anchors per definition',
            'profiles': {'E1': [{'year': y, 'f': f} for y, f in OUTSIDE_VIEW['profiles']['E1']],
                         'E2': [{'year': y, 'f': f} for y, f in OUTSIDE_VIEW['profiles']['E2']]},
            'profile_sources': OUTSIDE_VIEW['profile_sources'],
            'multipliers': [{'A': round(x['A'], 4), 'label': x['label'], 'basis': x['basis']}
                            for x in OUTSIDE_VIEW['multipliers']],
            'human_nature_multipliers': 'computed at run time (block 6): A_pess = Lambda(S2 human-nature '
                                        'overrides only)/Lambda(S1) and A_opt = Lambda(S3 human-nature overrides '
                                        'only)/Lambda(S1), Lambda = -ln(1 - P(E by 2100)) on the same S1 draws; '
                                        'the anchors are read as embodying historical human nature (S1)',
        },
        'regression': {'cells': [{'p': c['p'], 'rate': 'doubling' if c['grow'] else 'fixed',
                                  'paper_5_3_percent': c['paper_53'], 'paper_5_2_median': c['paper_52']}
                                 for c in S0_CELLS],
                       'years': S0_YEARS, 'Tc': S0_TC, 'r0': 4.0, 'Hc_override_hours': S0_HC_HOURS,
                       'tolerance_pp': 100.0 * REGRESSION_TOL_P,
                       'tolerance_median_yr': REGRESSION_TOL_MED,
                       'prior_averaged_grid': S0_GRID_36},
        'sensitivity': {'block_variants': [{'id': v['id'], 'label': v['label']} for v in BLOCK_VARIANTS],
                        's3_variants': [{'id': v['id'], 'label': v['label']} for v in S3_VARIANTS],
                        'sign_check_parameters': SIGN_PARAMS,
                        'sign_check_racing': SIGN_RACING,
                        'sign_check_values': 'prior q01 and q99',
                        'hn_effect_sweep': 'every human-nature parameter at its prior q01 and q99, CRN on the first '
                                           'S1 draws; inert in this model = |dP(E1)| < %g pp at %s'
                                           % (100 * INERT_PP, ', '.join(str(y) for y in HNE_YEARS)),
                        'main_result_table': 'S1 P(E1 by Y) and P(E2 by Y) for every output year, conditional on '
                                             'the e1 bins [1e-4, 1e-3), [1e-3, 1e-2), [1e-2, 1e-1), [1e-1, 0.5] '
                                             '(each row an average over e1 spread log-uniformly across the decade, '
                                             'not a point value), with prior mass, draw count and the accident / '
                                             'misuse + lock-in split at 2050 and 2100; and S1 P(E1) and P(E2) by '
                                             '%s at the point values e1 = %s (local means over the S1 draws within '
                                             '+/-%g decade, with draw count and SE; block 4b)'
                                             % (', '.join(str(Y) for Y in E1_POINT_YEARS),
                                                ', '.join(fnum(x) for x in E1_POINTS), E1_POINT_HALFWIDTH),
                        'conditional_tables': 'S1 conditional on the s_regime, s_align and x_plateau terciles '
                                              '(block 4c)'},
        'safe_exit_calibration': {'anchor_per_year': list(SAFE_ANCHOR),
                                  'checked_quantity': 'realised survival-weighted S1 regime-route hazard '
                                                      's_regime (1 + kappa_s R) + s_halt [paused], '
                                                      '2026.75-2050 (block 12)'},
        'premises': {'good_judgment': {'labels': PREMISES['labels'],
                                       'values': PREMISES['Good Judgment']['values'],
                                       'overall': PREMISES['Good Judgment']['overall']},
                     'carlsmith': {'labels': PREMISES['labels'],
                                   'values': PREMISES['Carlsmith']['values'],
                                   'overall': PREMISES['Carlsmith']['overall']},
                     'source': PREMISES['source']},
    }
    lines.append('mc:')
    lines.extend(_yaml_lines(mc, 1))

    def pentry(spec):
        ov = {}
        for key in ('S0', 'S1cal', 'S1side', 'S2', 'S3'):
            if key in spec['overrides']:
                ov[key] = dict(spec['overrides'][key])
        if spec['name'] in ('e1', 'g'):
            ov['S0'] = 'set per S0 cell: e1 = p in {0.01, 0.05, 0.20}; g in {0, ln2}'
        ent = {'name': spec['name'], 'meaning': spec['meaning'], 'prior': dict(spec['prior']),
               'tag': spec['tag'], 'human_nature': spec['hn'], 'rationale': spec['rationale'],
               'overrides': ov}
        if spec['hn']:
            rec = HN_EFFECT_RECORD.get(spec['name'])
            if rec is None:
                ent['effect'] = 'not recorded'
            else:
                inert = spec['name'] in HN_INERT_REASON
                ent['effect'] = {
                    'dE1_pp_2035_2050_2100': list(rec),
                    'class': 'inert in this model' if inert else 'active',
                    'reason': HN_INERT_REASON.get(spec['name'], ''),
                    'how_measured': 'P(E1 by Y) at the prior q99 minus at the q01, common random numbers on the '
                                    'first %d S1 draws per side (seed %d); inert = |effect| < %g pp at every year'
                                    % (N_SIGN_DEFAULT if spec['name'] in SIGN_PARAMS else N_HNE_DEFAULT,
                                       DEFAULT_SEED, 100 * INERT_PP)}
                if not ent['effect']['reason']:
                    del ent['effect']['reason']
        return ent
    lines.append('')
    lines.append('# Human-nature parameters (%d): behaviour of labs, states, firms and publics.' % N_HN)
    lines.append('human_nature:')
    lines.extend(_yaml_lines([pentry(s) for s in PARAMS if s['hn']], 1))
    lines.append('')
    lines.append('# AI-nature, capability and channel parameters (%d).' % (len(PARAMS) - N_HN))
    lines.append('priors:')
    lines.extend(_yaml_lines([pentry(s) for s in PARAMS if not s['hn']], 1))
    lines.append('')
    lines.append('# Scenarios: a scenario is the S1 priors plus the overrides listed per parameter above')
    lines.append('# (keys S0, S1cal, S1side, S2, S3) and the switches below.')
    scen = {}
    for sc in SCENARIOS:
        sw = dict(sc['switches'])
        scen[sc['key']] = {'label': sc['label'], 'override_key': sc['override_key'],
                           'random_stream': sc['stream'],
                           'draws': {'main': N_MAIN_DEFAULT, 'var': N_VAR_DEFAULT, 'var2': N_VAR2_DEFAULT,
                                     None: 6}[sc['n']],
                           'switches': sw}
    lines.append('scenarios:')
    lines.extend(_yaml_lines(scen, 1))
    lines.append('')
    lines.append('# Sources cited by key in the rationales above (public URLs).')
    lines.append('sources:')
    lines.extend(_yaml_lines([{'key': k, 'title': t, 'url': u} for k, t, u in SOURCES], 1))
    lines.append('')
    lines.append('# Review-finding ids (F01-F40) used in mc.py, model.py and the rationales: the 2026-10-10')
    lines.append('# review of paper v0.1. "refuted" means the finding did not survive adversarial verification.')
    lines.append('review_findings:')
    lines.extend(_yaml_lines([{'id': i, 'status': s, 'title': t} for i, s, t in REVIEW_FINDINGS], 1))
    return '\n'.join(lines) + '\n'


def print_sources_md():
    out = ['## Sources cited by key in model/mc.py', '',
           '| key | source | URL |', '| --- | --- | --- |']
    for k, t, u in SOURCES:
        out.append('| %s | %s | %s |' % (k, cell(t), u))
    out += ['', '## Review-finding index (2026-10-10 review of paper v0.1)', '',
            '| id | status | title |', '| --- | --- | --- |']
    for i, s, t in REVIEW_FINDINGS:
        out.append('| %s | %s | %s |' % (i, s, cell(t)))
    return '\n'.join(out) + '\n'


# ----------------------------------------------------------------------------
# Printers
# ----------------------------------------------------------------------------
def curve_row(label, d, curve):
    q = [t_quantile(curve, x) for x in (0.1, 0.5, 0.9)]
    return [label, d] + [pc(at(curve, Y)) for Y in YEARS] + [yr(x) for x in q] + [pc(1.0 - at(curve, 2100))]


HDR_CURVE = ['scenario', 'def'] + [str(Y) for Y in YEARS] + ['q10', 'q50', 'q90', 'P(no hit by 2100)']


def print_model_summary():
    emit('### 1b. The model in one screen')
    emit()
    for line in MODEL_SUMMARY:
        emit('- ' + line)
    emit()
    emit('Symbols: x raw capability (doublings above H0); x_def capability net of defense (containment clock only); '
         'x_c threshold; u uncontainability; '
         'R binding restraint (share of frontier activity removed); Sal salience in HF-class units; Sal2 '
         'tier-2 salience (count of undecayed mass-casualty/major losses); m_uni unilateral restraint '
         'multiplier; pi0, pi1 coordinated-pause hazard terms; m_eff effective independent sub-incidents per '
         'cluster; G and Phi the standard normal CDF gate. Full equations (M1-M13): docstring of model/mc.py. '
         'Design notes for this version: ' + ' '.join(DESIGN_NOTES))
    emit()


def print_regression(cells):
    emit('### 2. S0 regression against v0.1 (paper sec. 5.2 and 5.3; gate)')
    emit()
    emit('S0 runs the production engine with the v0.1 premises: step lethality at T_c = 2028.5 '
         '(Hc_override_hours = %.0f h), r0 = 4, m = 1, rho = 0, no response, no safe state, no misuse or '
         'concentration channel; rate fixed (g = 0) or doubling yearly from T_c (g = ln2, r_max = 1e12). '
         'Gate (all parts must hold for PASS): |P_engine - P_closed| <= %.2f pp at every year and |median '
         'difference| <= %.2f yr, medians read off the deterministic engine CDF; every engine cell rounds to the '
         'published sec. 5.3 cell and every engine median to the published sec. 5.2 median; the 12 sec. 5.2 bracket '
         'medians (T_c = %.3f and %.3f) meet the same median tolerance and round to the published brackets.'
         % (S0_HC_HOURS, 100 * REGRESSION_TOL_P, REGRESSION_TOL_MED, v01_Tc(*S0_BRACKET_CELLS[0][::-1]),
            v01_Tc(*S0_BRACKET_CELLS[1][::-1])))
    emit()
    rows = []
    worst_p = 0.0
    worst_m = 0.0
    all_pass = True
    all_round = True
    for c in cells:
        cl = c['cell']
        lab_p = '%g%%' % (100 * cl['p'])
        lab_r = 'doubling' if cl['grow'] else 'fixed'
        eng = [c['E1'][YIDX[Y]] for Y in S0_YEARS]
        clo = [v01_P(Y, S0_TC, cl['p'], cl['grow']) for Y in S0_YEARS]
        med_e = t_quantile(c['E1'], 0.5)
        med_c = v01_median(cl['p'], cl['grow'])
        dp = max(abs(a - b) for a, b in zip(eng, clo))
        dm = abs(med_e - med_c) if med_e is not None else float('inf')
        ok = dp <= REGRESSION_TOL_P and dm <= REGRESSION_TOL_MED
        all_pass = all_pass and ok
        worst_p = max(worst_p, dp)
        worst_m = max(worst_m, dm)
        rnd = all(abs(round(a, 2) - pp_ / 100.0) < 1e-9 for a, pp_ in zip(eng, cl['paper_53']))
        rnd = rnd and abs(round(med_e, 1) - cl['paper_52']) < 1e-9
        all_round = all_round and rnd
        rows.append([lab_p, lab_r, 'engine'] + ['%.3f%%' % (100 * v) for v in eng] +
                    ['%.2f' % med_e, 'PASS' if ok else 'FAIL'])
        rows.append([lab_p, lab_r, 'v0.1 closed form'] + ['%.3f%%' % (100 * v) for v in clo] +
                    ['%.2f' % med_c, 'max |dP| %.4f pp, |d med| %.3f yr' % (100 * dp, dm)])
        rows.append([lab_p, lab_r, 'paper 5.3 / 5.2 cell'] + ['%d%%' % v for v in cl['paper_53']] +
                    ['%.1f' % cl['paper_52'], 'engine rounds to paper: %s' % ('yes' if rnd else 'NO')])
    table(['p', 'r growth', 'metric'] + [str(Y) for Y in S0_YEARS] + ['median', 'PASS'], rows)
    brows = []
    br_pass = True
    worst_mb = 0.0
    for c in cells:
        cl = c['cell']
        cells_txt = []
        ok_c = True
        for b, paper_v in zip(c['br'], cl['paper_52_br']):
            med_e = t_quantile(b['E1'], 0.5)
            med_c = v01_median(cl['p'], cl['grow'], Tc=b['Tc'])
            dm = abs(med_e - med_c) if med_e is not None else float('inf')
            rnd = med_e is not None and abs(round(med_e, 1) - paper_v) < 1e-9
            ok_c = ok_c and dm <= REGRESSION_TOL_MED and rnd
            worst_mb = max(worst_mb, dm)
            cells_txt.append('%.3f / %.3f / %.1f' % (med_e, med_c, paper_v))
        br_pass = br_pass and ok_c
        brows.append(['%g%%' % (100 * cl['p']), 'doubling' if cl['grow'] else 'fixed'] + cells_txt +
                     ['PASS' if ok_c else 'FAIL'])
    emit('Paper sec. 5.2 bracket medians (engine / closed form / published), T_c = %.3f and %.3f:'
         % (cells[0]['br'][0]['Tc'], cells[0]['br'][1]['Tc']))
    emit()
    table(['p', 'r growth', 'T_c = %.3f' % cells[0]['br'][0]['Tc'], 'T_c = %.3f' % cells[0]['br'][1]['Tc'], 'PASS'],
          brows)
    gate = all_pass and all_round and br_pass
    line = ('S0 REGRESSION: %s (worst |dP| = %.4f pp, worst |d median| = %.3f yr at T_c = 2028.5 and %.3f yr at '
            'the brackets; every engine cell and median rounds to the published paper value: %s; bracket medians: '
            '%s)' % ('PASS' if gate else 'FAIL', 100 * worst_p, worst_m, worst_mb, 'yes' if all_round else 'NO',
                     'pass' if br_pass else 'FAIL'))
    emit(line)
    emit()
    return gate, worst_p, max(worst_m, worst_mb), line


def print_s0_cells(cells, avg):
    emit('### 3. S0 cells and the 36-cell prior-averaged row')
    emit()
    emit('Six deterministic engine paths (E2 = E1 because v0.1 does not distinguish them), then the '
         'uniform average of the paper\'s 36 closed-form cells (d in {3, 4.3, 7} months x H_c in '
         '{167, 2000} h x p in {1, 5, 20}% x {fixed, doubling}), each with its own T_c and clock.')
    emit()
    rows = []
    for c in cells:
        cl = c['cell']
        lab = 'S0 p = %g%%, %s' % (100 * cl['p'], 'doubling' if cl['grow'] else 'fixed')
        rows.append(curve_row(lab, 'E1', c['E1']))
        rows.append(curve_row(lab, 'E2 (= E1, v0.1)', c['E2']))
    for d in ('E1', 'E2 (= E1, v0.1)'):
        rows.append(['S0 prior-averaged (36 cells)', d] + [pc(avg['P'][Y]) for Y in YEARS] +
                    [yr(x) for x in avg['q']] + [pc(1.0 - avg['P'][2100])])
    table(HDR_CURVE, rows)
    emit('S0 overrides (from the PARAMS block): ' + '; '.join(
        '%s %s' % (s['name'], prior_str(s['overrides']['S0'])) for s in PARAMS if 'S0' in s['overrides'])
        + '; e1 = p and g in {0, ln2} per cell; switches: step, g_clock = Tc, response_off, '
          'init_post_hf = False, phi_const, Hc_override_hours = %.0f.' % S0_HC_HOURS)
    emit()


def band(res, Y):
    xs = [r['E1'][YP[Y]] for r in res['recs']]
    return qtl(xs, 0.1), qtl(xs, 0.5), qtl(xs, 0.9)


def print_main(results):
    emit('### 4. Main results: prior-predictive means by scenario')
    emit()
    emit('P(hit by Y) is the prior-predictive mean of P(E by Y) over the scenario\'s priors; q10/q50/q90 '
         'are quantiles of T_hit read off that mean curve (">2100" if never reached); "P(no hit by 2100)" '
         'includes safe exits, printed separately below each block. E2 strict is population < 5,000 by Y: an '
         'accidental E1 world counts once its capability passes its strict-extinction level, also after the E1 '
         'event (block 5 prints the older first-event convention as a lower bound).')
    emit()
    for res in results:
        emit('#### %s: %s (N = %d)' % (res['key'], res['label'], res['N']))
        emit()
        rows = [curve_row(res['key'], 'E1 broad', res['E1']),
                curve_row(res['key'], 'E2 strict (pop. < 5,000)', res['E2'])]
        rows.append([res['key'], 'P(safe exit by Y)'] + [pc(at(res['safe'], Y)) for Y in YEARS] +
                    ['-', '-', '-', '-'])
        rows.append([res['key'], 'P(still at risk at Y)'] +
                    [pc(1.0 - at(res['E1'], Y) - at(res['safe'], Y)) for Y in YEARS] + ['-', '-', '-', '-'])
        table(HDR_CURVE, rows)
        b35 = band(res, 2035)
        b50 = band(res, 2050)
        emit('parameter-and-event band (10/50/90%% across draws): P(E1 by 2035) = %s / %s / %s; '
             'P(E1 by 2050) = %s / %s / %s' % (pc(b35[0]), pc(b35[1]), pc(b35[2]),
                                                pc(b50[0]), pc(b50[1]), pc(b50[2])))
        emit()


E1_DECADES = [(1e-4, 1e-3), (1e-3, 1e-2), (1e-2, 1e-1), (1e-1, 0.5 + 1e-12)]
# Point values of e1 for the block-4b point-value table: local means over the S1 draws within
# E1_POINT_HALFWIDTH decade of each point (no new runs).
E1_POINTS = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3]
E1_POINT_HALFWIDTH = 0.15
E1_POINT_YEARS = (2035, 2050, 2100)


def cond_rows(res, label, groups):
    """groups: list of (group label, list of draw indices). Means of per-draw values over each group."""
    recs = res['recs']
    N = float(len(recs))
    rows = []
    for glab, idx in groups:
        if not idx:
            rows.append([label, glab, '0.0%'] + ['n/a'] * 9)
            continue
        sub = [recs[i] for i in idx]
        e1 = [mean(r['E1'][YP[Y]] for r in sub) for Y in (2035, 2050, 2100)]
        e2 = [mean(r['E2'][YP[Y]] for r in sub) for Y in (2050, 2100)]
        sf = mean(r['safe'][YP[2100]] for r in sub)
        tot = [sum(r['ch'][YP[2100]][c] for r in sub) for c in range(3)]
        st = sum(tot)
        sh = [t / st if st > 0 else float('nan') for t in tot]
        rows.append([label, glab, pc(len(idx) / N)] + [pc(x) for x in e1] + [pc(x) for x in e2] + [pc(sf)] +
                    [pc(x) for x in sh])
    return rows


def contrib_share(res, idx, Y):
    """Share of the scenario's mean P(E1 by Y) carried by the draws in idx."""
    tot = sum(r['E1'][YP[Y]] for r in res['recs'])
    part = sum(res['recs'][i]['E1'][YP[Y]] for i in idx)
    return part / tot if tot > 0 else float('nan')


def tercile_groups(res, name):
    params = res['params']
    N = len(params)
    order = sorted(range(N), key=lambda i: params[i][name])
    n3 = N // 3
    cuts = (order[:n3], order[n3:N - n3], order[N - n3:])
    labs = []
    for c, lab in zip(cuts, ('bottom', 'middle', 'top')):
        lo = params[c[0]][name]
        hi = params[c[-1]][name]
        labs.append(('%s tercile (%s-%s)' % (lab, fnum(lo), fnum(hi)), list(c)))
    return labs


def e1_bins(res):
    """The S1 draws grouped by the e1 decade bins (E1_DECADES), with the prior mass of each bin under the S1
    prior of e1 (log-uniform) and the draw indices."""
    spec = PARAM_BY_NAME['e1']['prior']
    lnw = math.log(spec['hi'] / spec['lo'])
    out = []
    for lo, hi in E1_DECADES:
        hi_c = min(hi, spec['hi'])
        idx = [i for i, P in enumerate(res['params']) if lo <= P['e1'] < hi]
        lab = 'e1 in [%s, %s%s' % (fnum(lo), fnum(hi_c), ']' if hi > spec['hi'] else ')')
        out.append({'label': lab, 'lo': lo, 'hi': hi_c, 'mass': math.log(hi_c / lo) / lnw, 'idx': idx})
    return out


def e1_points(res):
    """The S1 draws whose log10 e1 lies within E1_POINT_HALFWIDTH decade of each point in E1_POINTS (clipped
    to the prior's range), with the local mean, its Monte Carlo SE and the intercept of a local linear fit in
    log10 e1 (read at the point) of P(E1 by Y) and P(E2 by Y) for Y in E1_POINT_YEARS."""
    spec = PARAM_BY_NAME['e1']['prior']
    llo, lhi = math.log10(spec['lo']), math.log10(spec['hi'])
    le = [math.log10(P['e1']) for P in res['params']]
    out = []
    for x in E1_POINTS:
        c = math.log10(x)
        lo, hi = max(c - E1_POINT_HALFWIDTH, llo), min(c + E1_POINT_HALFWIDTH, lhi)
        idx = [i for i, v in enumerate(le) if lo <= v < hi or (hi == lhi and v == hi)]
        pt = {'e1': x, 'lo': 10 ** lo, 'hi': 10 ** hi, 'mid': 10 ** (0.5 * (lo + hi)),
              'one_sided': abs((c - lo) - (hi - c)) > 1e-9, 'idx': idx,
              'mean': {'E1': {}, 'E2': {}}, 'se': {'E1': {}, 'E2': {}}, 'loclin': {'E1': {}, 'E2': {}}}
        xs = [le[i] - c for i in idx]
        mx = mean(xs)
        sxx = sum((u - mx) ** 2 for u in xs)
        for d in ('E1', 'E2'):
            for Y in E1_POINT_YEARS:
                ys = [res['recs'][i][d][YP[Y]] for i in idx]
                my = mean(ys)
                pt['mean'][d][Y] = my
                pt['se'][d][Y] = se_of(ys)
                if len(idx) > 2 and sxx > 0:
                    b = sum((u - mx) * (y - my) for u, y in zip(xs, ys)) / sxx
                    pt['loclin'][d][Y] = my - b * mx
                else:
                    pt['loclin'][d][Y] = float('nan')
        out.append(pt)
    return out


def channel_split(res, idx, Y):
    """Mean P(E1 via accident by Y) and P(E1 via misuse or lock-in by Y) over the draws in idx, with the
    Monte Carlo SE of each."""
    yi = YP[Y]
    acc = [res['recs'][i]['ch'][yi][0] for i in idx]
    side = [res['recs'][i]['ch'][yi][1] + res['recs'][i]['ch'][yi][2] for i in idx]
    return mean(acc), se_of(acc), mean(side), se_of(side)


def print_conditional(S1):
    emit('### 4b. Main result table: S1 by e1 decade bin and at point values of e1 (e1 is the dominant '
         '[Assumption]; a reader with a point belief about e1 reads the point-value table)')
    emit()
    bins = e1_bins(S1)
    allidx = list(range(len(S1['recs'])))
    emit('Each row is the mean over the S1 draws whose per-sub-incident lethality e1 falls in the bin, that is the '
         'average over e1 spread log-uniformly across that decade, not the value at any point in it; the prior '
         'mass is that of the S1 prior e1 ~ %s. P(E1) rises steeply within each decade, so a bin row overstates '
         'the value at the bin\'s lower end and understates it at its upper end. A reader with a point belief about '
         'e1 reads the point-value table below, not the row for its bin and not the prior mean (the last rows). '
         'E1 broad = extinction or permanent loss of collective self-determination; E2 strict = population < '
         '5,000 by Y.' % prior_str(PARAM_BY_NAME['e1']['prior']))
    emit()
    rows = []
    for b in bins + [{'label': 'all S1 draws (prior mean)', 'mass': 1.0, 'idx': allidx}]:
        sub = [S1['recs'][i] for i in b['idx']]
        for d, dl in (('E1', 'E1 broad'), ('E2', 'E2 strict')):
            vals = [pc(mean(r[d][YP[Y]] for r in sub)) if sub else 'n/a' for Y in YEARS]
            rows.append([b['label'], pc(b['mass']), str(len(sub)), dl] + vals)
    table(['e1 bin', 'prior mass', 'draws', 'def'] + ['P(by %d)' % Y for Y in YEARS], rows)
    pts = e1_points(S1)
    emit('Point values of e1 (local means over the same S1 draws, no new runs): each row is the mean over the S1 '
         'draws whose log10 e1 lies within +/-%g decade of the point (a window spanning a factor %.1f in e1), with '
         'the Monte Carlo SE in pp in brackets. At the prior\'s lower end the window is one-sided, so that row reads e1 '
         'near the window\'s log-midpoint rather than at the point.'
         % (E1_POINT_HALFWIDTH, 10 ** (2 * E1_POINT_HALFWIDTH)))
    emit()
    rows = []
    for pt in pts:
        idx = pt['idx']
        lab = fnum(pt['e1'])
        if pt['one_sided']:
            lab += ' (one-sided window; log-midpoint %s)' % fnum(pt['mid'])
        cells = []
        for d in ('E1', 'E2'):
            for Y in E1_POINT_YEARS:
                v = [S1['recs'][i][d][YP[Y]] for i in idx]
                se = 100 * se_of(v) if len(v) > 1 else float('nan')
                cells.append('%s (%s)' % (pc(mean(v)), '<0.1' if se < 0.05 else '%.1f' % se) if se == se else 'n/a')
        rows.append([lab, '[%s, %s)' % (fnum(pt['lo']), fnum(pt['hi'])), str(len(idx))] + cells)
    table(['e1 (point)', 'window', 'draws'] + ['P(E1 by %d)' % Y for Y in E1_POINT_YEARS] +
          ['P(E2 by %d)' % Y for Y in E1_POINT_YEARS], rows)
    ll_dev = max(abs(pt['loclin'][d][Y] - pt['mean'][d][Y]) for pt in pts
                 if not pt['one_sided'] and len(pt['idx']) > 2 for d in ('E1', 'E2') for Y in E1_POINT_YEARS)
    ll_rat = max(abs(pt['loclin'][d][Y] - pt['mean'][d][Y]) / pt['se'][d][Y] for pt in pts
                 if not pt['one_sided'] and len(pt['idx']) > 2 for d in ('E1', 'E2') for Y in E1_POINT_YEARS
                 if pt['se'][d][Y] > 0)
    p_lo = [pt for pt in pts if abs(pt['e1'] - 1e-2) < 1e-12][0]
    b_lo = [b for b in bins if abs(b['lo'] - 1e-2) < 1e-12][0]
    emit('Check: a local linear fit in log10 e1 on the same windows, read at the point, differs from the local mean '
         'by at most %s at the two-sided points and by at most %.1f times the local mean\'s own Monte Carlo SE, so '
         'the slope and curvature of P(E1) and P(E2) inside a window %s. The bin rows are not point '
         'values: at e1 = 1e-2, the lower end of the [1e-2, 1e-1) bin, P(E1 by 2050) is %s and P(E1 by 2100) %s, '
         'against %s and %s in that bin\'s row.'
         % (pp(ll_dev).lstrip('+'), ll_rat,
            'do not bias the local means beyond their sampling error' if ll_rat < 1.0 else
            'bias some local means by more than their sampling error (read such a row as a window average)',
            pc(p_lo['mean']['E1'][2050]), pc(p_lo['mean']['E1'][2100]),
            pc(mean(S1['recs'][i]['E1'][YP[2050]] for i in b_lo['idx'])),
            pc(mean(S1['recs'][i]['E1'][YP[2100]] for i in b_lo['idx']))))
    emit()
    emit('The same bins split by channel (cumulative incidence of the first E1 event by channel; accident = '
         'accidental escape, side = deliberate misuse + power concentration or lock-in; Monte Carlo SE in '
         'brackets):')
    emit()
    rows = []
    for b in bins + [{'label': 'all S1 draws (prior mean)', 'mass': 1.0, 'idx': allidx}]:
        idx = b['idx']
        if not idx:
            rows.append([b['label'], pc(b['mass']), '0'] + ['n/a'] * 7)
            continue
        cells = []
        for Y in (2050, 2100):
            a, sa, sd, ss = channel_split(S1, idx, Y)
            e1v = [S1['recs'][i]['E1'][YP[Y]] for i in idx]
            cells += ['%s (%.1f)' % (pc(mean(e1v)), 100 * se_of(e1v)), '%s (%.1f)' % (pc(a), 100 * sa),
                      '%s (%.1f)' % (pc(sd), 100 * ss)]
        sf = mean(S1['recs'][i]['safe'][YP[2100]] for i in idx)
        rows.append([b['label'], pc(b['mass']), str(len(idx))] + cells + [pc(sf)])
    table(['e1 bin', 'prior mass', 'draws', 'P(E1 by 2050)', 'accident 2050', 'side 2050', 'P(E1 by 2100)',
           'accident 2100', 'side 2100', 'P(safe by 2100)'], rows)
    hi_idx = [i for i, P in enumerate(S1['params']) if P['e1'] >= 1e-2]
    lo_idx = [i for i, P in enumerate(S1['params']) if P['e1'] < 1e-2]
    low = bins[0]
    prior_mass = sum(b['mass'] for b in bins if b['lo'] >= 1e-2)
    side_low = channel_split(S1, low['idx'], 2100)[2] if low['idx'] else float('nan')
    e1_low = mean(S1['recs'][i]['E1'][YP[2100]] for i in low['idx']) if low['idx'] else float('nan')
    emit('Draws with e1 >= 1e-2 (prior mass %s, %s of the draws) carry %s of S1\'s P(E1 by 2050) and %s of P(E1 by '
         '2100); draws with e1 < 1e-2 give P(E1 by 2050) = %s. In the lowest bin (e1 < 1e-3) P(E1 by 2100) is %s, of '
         'which %s (%s of it) come from the misuse and lock-in channels: that floor is set by lam_mis, lam_conc and '
         'L_conc, not by e1.'
         % (pc(prior_mass), pc(len(hi_idx) / float(len(S1['params']))), pc(contrib_share(S1, hi_idx, 2050)),
            pc(contrib_share(S1, hi_idx, 2100)), pc(mean(S1['recs'][i]['E1'][YP[2050]] for i in lo_idx)),
            pc(e1_low), pp(side_low).lstrip('+'), pc(side_low / e1_low if e1_low > 0 else float('nan'), 0)))
    emit()
    emit('### 4c. S1 conditional on the safe-exit and plateau priors (terciles)')
    emit()
    rows = cond_rows(S1, 's_regime tercile', tercile_groups(S1, 's_regime'))
    rows += cond_rows(S1, 's_align tercile', tercile_groups(S1, 's_align'))
    rows += cond_rows(S1, 'x_plateau tercile', tercile_groups(S1, 'x_plateau'))
    table(['conditioning', 'group', 'share of draws', 'P(E1 by 2035)', 'P(E1 by 2050)', 'P(E1 by 2100)',
           'P(E2 by 2050)', 'P(E2 by 2100)', 'P(safe by 2100)', 'accident share 2100', 'misuse share 2100',
           'concentration share 2100'], rows)
    emit('Each row is the mean over the S1 draws in that tercile of the named parameter.')
    emit()
    return {'prior_mass_hi': prior_mass, 'share50_hi': contrib_share(S1, hi_idx, 2050),
            'share100_hi': contrib_share(S1, hi_idx, 2100),
            'p50_lo': mean(S1['recs'][i]['E1'][YP[2050]] for i in lo_idx),
            'p50_hi': mean(S1['recs'][i]['E1'][YP[2050]] for i in hi_idx),
            'e1_low_2100': e1_low, 'side_low_2100': side_low,
            'points': [{'e1': pt['e1'], 'window': [pt['lo'], pt['hi']], 'one_sided': pt['one_sided'],
                        'draws': len(pt['idx']),
                        'E1': dict((str(Y), pt['mean']['E1'][Y]) for Y in E1_POINT_YEARS),
                        'E2': dict((str(Y), pt['mean']['E2'][Y]) for Y in E1_POINT_YEARS),
                        'E1_se': dict((str(Y), pt['se']['E1'][Y]) for Y in E1_POINT_YEARS),
                        'E2_se': dict((str(Y), pt['se']['E2'][Y]) for Y in E1_POINT_YEARS)} for pt in pts],
            'bins': [{'label': b['label'], 'prior_mass': b['mass'], 'draws': len(b['idx']),
                      'E1': dict((str(Y), mean(S1['recs'][i]['E1'][YP[Y]] for i in b['idx'])) for Y in YEARS),
                      'E2': dict((str(Y), mean(S1['recs'][i]['E2'][YP[Y]] for i in b['idx'])) for Y in YEARS),
                      'accident_2050': channel_split(S1, b['idx'], 2050)[0],
                      'side_2050': channel_split(S1, b['idx'], 2050)[2],
                      'accident_2100': channel_split(S1, b['idx'], 2100)[0],
                      'side_2100': channel_split(S1, b['idx'], 2100)[2]} for b in bins if b['idx']]}


def channel_shares(res, Y, which='ch'):
    yi = YP[Y]
    tot = [0.0, 0.0, 0.0]
    for r in res['recs']:
        c = r[which][yi]
        for i in range(3):
            tot[i] += c[i]
    s = sum(tot)
    return [t / s if s > 0 else float('nan') for t in tot]


def print_channels(results):
    emit('### 5. Channel shares of first E1 hits')
    emit()
    rows = []
    for res in results:
        for Y in (2050, 2100):
            sh = channel_shares(res, Y)
            rows.append([res['key'], Y, pc(sh[0]), pc(sh[1]), pc(sh[2]), pc(at(res['E1'], Y)),
                         pc(at(res['safe'], Y))])
    table(['scenario', 'Y', 'accidental escape', 'deliberate misuse', 'concentration or lock-in',
           'P(E1 by Y)', 'P(safe by Y)'], rows)
    rows = []
    for res in results:
        sh = channel_shares(res, 2100, 'ch2')
        e2f = mean(r['E2f'][YP[2100]] for r in res['recs'])
        e1 = at(res['E1'], 2100)
        rows.append([res['key'], pc(sh[0]), pc(sh[1]), pc(sh[2]), pc(at(res['E2'], 2100)),
                     pc(at(res['E2'], 2100) / e1) if e1 > 0 else 'n/a', pc(e2f),
                     pc(e2f / e1) if e1 > 0 else 'n/a'])
    emit('E2 (strict, population < 5,000 by 2100) by channel, 2100. The last two columns use the older convention '
         'in which E2 is decided at the first E1 event with the capability of that moment; it cannot count an E1 '
         'world that later becomes strict extinction, so it is a lower bound on the XPT quantity:')
    emit()
    table(['scenario', 'accidental escape', 'deliberate misuse', 'concentration or lock-in',
           'P(E2 by 2100)', 'P(E2 | E1) by 2100', 'P(E2 at the first E1 event by 2100) (lower bound)',
           'its ratio to P(E1)'], rows)


def channel_curve(res, part):
    """Mean cumulative incidence of the first E1 event by channel at each output year: part = 'acc'
    (accidental escape) or 'side' (deliberate misuse + power concentration or lock-in)."""
    out = []
    for Y in YEARS:
        yi = YP[Y]
        if part == 'acc':
            out.append(mean(r['ch'][yi][0] for r in res['recs']))
        else:
            out.append(mean(r['ch'][yi][1] + r['ch'][yi][2] for r in res['recs']))
    return out


CHANNEL_MATCH = ('S1', 'S1-cal', 'S3')


def print_outside_view(pools, rows, S1, blocks, hnm, results):
    emit('### 6. Outside view: calibrated external anchors and the human-nature-implied adjustment')
    emit()
    emit('Anchors are human estimates matched by definition (E1 broad, E2 strict) as far as the sources allow; '
         'the tags mark the unverified anchor and the "scope" column the channels each anchor covers. Horizons '
         'other than 2100 are converted to the window 2026.75-2100 at constant hazard: '
         'P_2100 = 1 - (1 - P_src)^((2100 - 2026.75)/(Y_src - 2026.75)), i.e. the whole source probability is '
         'assigned to the window after 2026.75; no stated horizon is read as 2100. Pools are geometric means. '
         'Source keys resolve in block 11. Channel scope of the E1 pool: %s. Scope of the E2 pool: %s.'
         % (OUTSIDE_VIEW['E1_scope'], OUTSIDE_VIEW['E2_scope']))
    emit()
    trows = []
    for d, key in (('E1', 'E1_pool'), ('E2', 'E2_pool')):
        for a, cv in zip(OUTSIDE_VIEW[key], pools[d]['conv']):
            trows.append([d, a['source'], a['definition'], a.get('scope', 'all AI channels'), a['vintage'], a['hf'],
                          pc(a['p_src'], 2),
                          'horizon %s' % ('none (read as 2100)' if a['y_src'] is None else '%g' % a['y_src']),
                          pc(cv, 2), 'pooled', a['tag'], a['src']])
        trows.append([d, 'POOL (geometric mean; %s)' % OUTSIDE_VIEW['def_notes'][d], '', '', '', '', '', '',
                      pc(pools[d]['pool'], 2), '', '', ''])
    for a in OUTSIDE_VIEW['E1_not_pooled']:
        trows.append(['E1 high end', a['source'], a['reason'], '', '', '', pc(a['value'], 1), '', '',
                      'not pooled', '', a['src'] or '-'])
    table(['def', 'source', 'definition', 'scope', 'vintage', 'pre/post HF', 'value', 'conversion',
           'P by 2100', 'use', 'tag', 'source key'], trows)
    R = dict((r['key'], r) for r in results)
    orows = []
    for d in ('E1', 'E2'):
        for mlt in OUTSIDE_VIEW['multipliers']:
            orows.append(['%s [%s]' % (mlt['label'], OUTSIDE_VIEW['def_notes'][d]), d] +
                         [pc(v) for v in rows[(d, mlt['id'])]])
        if d == 'E1':
            for k in CHANNEL_MATCH:
                orows.append(['model %s, accident channel only (the channel the E1 pool covers)' % k, 'E1 accident'] +
                             [pc(v) for v in channel_curve(R[k], 'acc')])
                orows.append(['model %s, misuse + lock-in channels (outside the E1 pool\'s scope)' % k, 'E1 side'] +
                             [pc(v) for v in channel_curve(R[k], 'side')])
                orows.append(['model %s, all channels' % k, 'E1'] + [pc(at(R[k]['E1'], Y)) for Y in YEARS])
        orows.append(['outside view x A_pess = %.2f, human-nature-implied, pessimistic (model-derived)' % hnm[d]['pess'],
                      d] + [pc(v) for v in rows[(d, 'hn_pess')]])
        orows.append(['outside view x A_opt = %.2f, human-nature-implied, optimistic (model-derived)' % hnm[d]['opt'],
                      d] + [pc(v) for v in rows[(d, 'hn_opt')]])
        orows.append(['level from external anchors, timing from this model (S1 profile)', d] +
                     [pc(v) for v in rows[(d, 'S1')]])
    table(['row', 'def'] + [str(Y) for Y in YEARS], orows)
    for mlt in OUTSIDE_VIEW['multipliers'][1:]:
        emit('- %s: %s' % (mlt['label'], mlt['basis']))
    emit('- Time profiles: %s.' % OUTSIDE_VIEW['profile_sources'])
    emit('- Human-nature-implied rows (derivation): the anchors are read as embodying historical human nature, '
         'i.e. S1\'s human-nature priors, so the central human-nature adjustment is x1.00. The model then scales '
         'their century hazard by the ratio of its own cumulative hazards Lambda = -ln(1 - P(E by 2100)) on the same '
         '%d S1 draws: A_pess = Lambda(S2 human-nature overrides only) / Lambda(S1) = %.3f / %.3f = %.2f (E1), '
         '%.2f (E2); A_opt = Lambda(S3 human-nature overrides only) / Lambda(S1) = %.3f / %.3f = %.2f (E1), %.2f '
         '(E2). For scale, the whole modelled human response (S1 against the zero-response variant) is A_resp = '
         'Lambda(S1) / Lambda(zero response) = %.3f / %.3f = %.2f (E1). Of the fixed post-HF multipliers, only '
         'x1.06 (incentives) is a human-nature update; x1.19 (timelines) and x1.8 (high-impact failures) are '
         'AI-nature updates.'
         % (blocks[0]['N'], hnm['E1']['S2hn'], hnm['E1']['S1'], hnm['E1']['pess'], hnm['E2']['pess'],
            hnm['E1']['S3hn'], hnm['E1']['S1'], hnm['E1']['opt'], hnm['E2']['opt'], hnm['E1']['S1'],
            hnm['E1']['zero'], hnm['E1']['resp']))
    emit()
    nos = [b for b in blocks if b['id'] == 'nosafe'][0]
    f_nos = mass_share_2050(nos['E1'])
    f_s1 = pools['E1']['fS1_2050']
    f_s1b = mass_share_2050(S1['snap'][nos['N']]['E1'])     # S1 on the same first draws as the no-safe-exit run
    f_fc = OUTSIDE_VIEW['profiles']['E1'][1][1]
    f_uni = (2050.0 - T_START) / (2100.0 - T_START)
    absorb = f_s1b - f_nos      # safe-exit absorption: S1 against S1 with no safe exit, same draws
    timing = f_nos - f_fc       # the model's own hazard timing without a safe exit, against the forecasters
    gap = f_s1b - f_fc
    if gap <= 0.02:
        cause = ('On these draws S1 is not more front-loaded than the forecasters\' profile (gap %+.2f), so there is '
                 'no excess to attribute' % gap)
    else:
        if absorb > 0 and timing > 0 and abs(absorb - timing) >= 0.02:
            which = ('safe-exit absorption is the larger part' if absorb > timing else
                     'the hazard\'s own timing is the larger part and safe-exit absorption explains %s of the gap'
                     % ('less than half' if absorb < 0.5 * gap else 'about half'))
        elif absorb > 0 and timing > 0:
            which = 'the two parts are about equal'
        else:
            which = 'the two parts have opposite signs, so neither explains the gap alone'
        cause = ('Of S1\'s excess front-loading over the forecasters\' profile (%.2f against %.2f, %+.2f, on the '
                 'first %d S1 draws), safe-exit absorption (paths that reach the safe state stop accumulating hazard; '
                 'S1 against S1 with no safe exit) accounts for %+.2f and the hazard\'s own timing (S1 with no safe '
                 'exit against the forecasters\' profile) for %+.2f: %s. The timing part includes the spread of '
                 'hazards across draws (draws with high e1 are hit early, which front-loads the mean curve); for '
                 'scale, a hazard constant over %.2f-2100 gives %.2f'
                 % (f_s1b, f_fc, gap, nos['N'], absorb, timing, which, T_START, f_uni))
    emit('Share of century mass by 2050 (Lambda(2050)/Lambda(2100)): forecasters\' profile %.2f (E1) / %.2f (E2); '
         'S1 profile %.2f (E1) / %.2f (E2) (all %d draws; %.2f on the first %d); S1 with no safe exit %.2f (E1, '
         'N = %d, same first draws). %s.'
         % (f_fc, OUTSIDE_VIEW['profiles']['E2'][1][1], f_s1, pools['E2']['fS1_2050'], S1['N'], f_s1b, nos['N'],
            f_nos, nos['N'], cause))
    emit()


def comp_row(lab, res):
    return [lab, pc(at(res['E1'], 2035)), pc(at(res['E1'], 2050)), pc(at(res['E1'], 2100)),
            pc(at(res['E2'], 2100)), yr(t_quantile(res['E1'], 0.5)), pc(at(res['safe'], 2100))]


def print_comparison(cells, avg, results, ov_rows, blocks, s3v, hnm):
    emit('### 7. Comparison table')
    emit()
    rows = []
    c0 = cells[0]
    rows.append(['S0 (p = 1%, fixed; v0.1)', pc(c0['E1'][YIDX[2035]]), pc(c0['E1'][YIDX[2050]]),
                 pc(c0['E1'][YIDX[2100]]), pc(c0['E2'][YIDX[2100]]) + ' (= E1)',
                 yr(t_quantile(c0['E1'], 0.5)), pc(0.0)])
    rows.append(['S0 prior-averaged (36 cells)', pc(avg['P'][2035]), pc(avg['P'][2050]), pc(avg['P'][2100]),
                 pc(avg['P'][2100]) + ' (= E1)', yr(avg['q'][1]), pc(0.0)])
    S1 = [r for r in results if r['key'] == 'S1'][0]
    s1b = S1['snap'][blocks[0]['N']]
    for res in results:
        lab = res['key'] + (' (premise-driven)' if res['key'] == 'S2' else '')
        rows.append(comp_row(lab, res))
        if res['key'] == 'S1':
            rows.append(comp_row('S1, first %d draws (base of the S1 variant rows)' % blocks[0]['N'], s1b))
            for bid in ('anchor', 'nosafe', 'noplateau', 'nocouple', 'nopause', 'noevents'):
                b = block_by_id(blocks, bid)
                rows.append(comp_row('S1, %s (N = %d)' % (b['label'], b['N']), b))
        if res['key'] == 'S3':
            s3b = res['snap'][s3v[0]['N']]
            rows.append(comp_row('S3, first %d draws (base of the S3 variant rows)' % s3v[0]['N'], s3b))
            for v in s3v:
                if v['id'] in ('S3anchor', 'S3nosafe'):
                    rows.append(comp_row('%s (N = %d)' % (v['label'], v['N']), v))
    R = dict((r['key'], r) for r in results)
    for k in CHANNEL_MATCH:
        acc = channel_curve(R[k], 'acc')
        side = channel_curve(R[k], 'side')
        rows.append(['%s, accident channel only (matches the E1 pool\'s scope)' % k, pc(acc[YP[2035]]),
                     pc(acc[YP[2050]]), pc(acc[YP[2100]]), '-', '-', '-'])
        rows.append(['%s, misuse + lock-in channels only (outside the E1 pool\'s scope)' % k, pc(side[YP[2035]]),
                     pc(side[YP[2050]]), pc(side[YP[2100]]), '-', '-', '-'])
    for mid in ('raw', 'x2.3', 'x1.06'):
        mlt = [x for x in OUTSIDE_VIEW['multipliers'] if x['id'] == mid][0]
        e1 = ov_rows[('E1', mid)]
        e2 = ov_rows[('E2', mid)]
        rows.append(['%s [%s; %s]' % (mlt['label'], OUTSIDE_VIEW['def_notes']['E1'], OUTSIDE_VIEW['def_notes']['E2']),
                     pc(e1[YP[2035]]), pc(e1[YP[2050]]), pc(e1[YP[2100]]), pc(e2[YP[2100]]), '>2100', '-'])
    for key, lab in (('hn_pess', 'outside view x A_pess = %.2f, human-nature-implied, pessimistic (block 6)'
                      % hnm['E1']['pess']),
                     ('hn_opt', 'outside view x A_opt = %.2f, human-nature-implied, optimistic (block 6)'
                      % hnm['E1']['opt'])):
        e1 = ov_rows[('E1', key)]
        e2 = ov_rows[('E2', key)]
        rows.append([lab, pc(e1[YP[2035]]), pc(e1[YP[2050]]), pc(e1[YP[2100]]), pc(e2[YP[2100]]),
                     '>2100' if e1[YP[2100]] < 0.5 else '-', '-'])
    table(['row', 'P(E1 by 2035)', 'P(E1 by 2050)', 'P(E1 by 2100)', 'P(E2 by 2100)',
           'median T_hit (E1)', 'P(safe by 2100)'], rows)
    emit('The "S1, ..." rows rerun the first %d S1 draws with one block changed (common random numbers); compare '
         'them with the "S1, first %d draws" row. "Safe exit = historical regime anchor only" keeps s_regime and '
         'drops the alignment, restraint and halt routes. The S3 rows do the same on the first %d S3 draws. The '
         'outside-view x2.3 multiplier factors into timelines x1.19, incentives x1.06 and high-impact failures x1.8; '
         'only the incentives factor is a human-nature update. The "accident channel only" rows are the like-for-like '
         'comparison with the outside-view E1 rows, whose anchors mostly cover misalignment / loss of control; the '
         '"misuse + lock-in" rows are outside that scope and rest on lam_mis, lam_conc and L_conc ([Assumption]).'
         % (blocks[0]['N'], blocks[0]['N'], s3v[0]['N']))
    emit()


def premise_factors(res):
    recs = res['recs']
    N = len(recs)
    out = {}
    for Y in (2050, 2100):
        yi = YP[Y]
        sel = [r for r in recs if r['Tc'] <= Y]
        p_tc = len(sel) / float(N)
        p_cont = 1.0 - mean(r['Psafe_Tc'] for r in sel) if sel else float('nan')
        p_unc = mean(1.0 - math.exp(-r['LamU'][yi]) for r in recs)
        f_acc = mean(r['ch'][yi][0] for r in recs)
        e_cum = f_acc / p_unc if p_unc > 0 else float('nan')
        e1y = at(res['E1'], Y)
        out[Y] = [p_tc, p_cont, None, p_unc, e_cum, 1.0, at(res['E2'], Y) / e1y if e1y > 0 else float('nan'), e1y]
    return out


def print_premises(S1):
    emit('### 8. Premise reconciliation (S1) against Good Judgment and Carlsmith')
    emit()
    f = premise_factors(S1)
    labels = ['timelines: P(T_c <= Y)', 'incentives: P(continue | T_c <= Y) = 1 - mean P(safe by T_c)',
              'alignment difficulty (no separate model factor; folded into u and e1)',
              'high-impact failure: P(>= 1 uncontainable cluster by Y), ignores absorption',
              'disempowerment given failure: implied cumulative e = F_acc(Y) / P(>= 1 uncontainable)',
              'catastrophe given disempowerment: 1 by definition (E1 counts permanent disempowerment)',
              'strict extinction given E1: P(E2 | E1) by Y (no Carlsmith or Good Judgment analogue)',
              'overall: P(E1 by Y)']
    ext = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
    rows = []
    for i, lab in enumerate(labels):
        a = f[2050][i]
        b = f[2100][i]
        if i in ext:
            gj = '%.2f' % PREMISES['Good Judgment']['values'][ext[i]]
            ca = '%.2f' % PREMISES['Carlsmith']['values'][ext[i]]
        elif i == 6:
            gj = ca = '-'
        else:
            gj = PREMISES['Good Judgment']['overall']
            ca = PREMISES['Carlsmith']['overall']
        rows.append([lab, '-' if a is None else '%.3f' % a, '-' if b is None else '%.3f' % b, gj, ca])
    table(['premise', 'model factor (2050)', 'model factor (2100)', 'Good Judgment', 'Carlsmith'], rows)
    prod = {}
    for who in ('Good Judgment', 'Carlsmith'):
        v = 1.0
        for x in PREMISES[who]['values']:
            v *= x
        prod[who] = v
    mprod = {}
    for Y in (2050, 2100):
        v = 1.0
        for i in (0, 1, 3, 4, 5):
            v *= f[Y][i]
        mprod[Y] = v
    emit('Source of the premise values: %s. Notes: (i) the model factors are not a multiplicative decomposition '
         'of the overall row: their product is %.3f (2050) and %.3f (2100) against the overall %.3f and %.3f, '
         'because the overall row adds the misuse and concentration channels and counts absorption; (ii) the '
         "model's high-impact failure is an uncontainable cluster (containment > 30 days), a far weaker event "
         "than Carlsmith's high-impact failure of a power-seeking APS system, so its factor is not comparable "
         'one-to-one; (iii) the external overall rows are the forecasters\' direct estimates, not the products of '
         'their premises: the Good Judgment premises multiply to %.2f%% against their stated 1%%, Carlsmith\'s to '
         '%.1f%% against ~5%%. The 2026 record moved the timelines and incident-frequency factors (threshold dates, '
         'r0, cluster counts); it did not move uncontainability or disempowerment, which in this model rest on the '
         'containment-time prior and on e1, an [Assumption].'
         % (PREMISES['source'], mprod[2050], mprod[2100], f[2050][7], f[2100][7], 100 * prod['Good Judgment'],
            100 * prod['Carlsmith']))
    emit()


def block_by_id(blocks, bid):
    return [b for b in blocks if b['id'] == bid][0]


def print_sensitivity(oat_rows, base50, terc, N_OAT, S1, blocks, sign_rows):
    emit('### 9. S1 sensitivity')
    emit()
    emit('(a) One-at-a-time with common random numbers: each row fixes one parameter at its prior q10 or q90 '
         'and resamples nothing else (the other parameters keep their S1 draws). "swing" and the dP columns are '
         'the value at q90 minus the value at q10; "paired SE" is the standard error of the swing over the %d '
         'paired draws.' % N_OAT)
    emit()
    rows = sorted(oat_rows, key=lambda r: -abs(r['swing']))
    trows = []
    for r in rows:
        tc, tse = terc[r['name']]
        trows.append([r['name'], 'HN' if r['hn'] else '', fnum(r['q10']), fnum(r['q90']), pc(r['p50_lo']),
                      pc(r['p50_hi']), pp(r['swing']), '%.1f pp' % (100 * r['se']), pp(r['d35']),
                      pp(r['dsafe']), '%s (%.1f)%s' % (pp(tc), 100 * tse, ' noise' if abs(tc) < 2 * tse else '')])
    table(['parameter', 'HN', 'q10 value', 'q90 value', 'P(E1 by 2050) at q10', 'at q90', 'swing',
           'paired SE', 'dP(E1 by 2035)', 'dP(safe by 2100)', 'e1-stratified tercile contrast (SE)'], trows)
    e1row = [r for r in rows if r['name'] == 'e1'][0]
    hn = [r for r in rows if r['hn']]
    ai = [r for r in rows if not r['hn']]
    top_hn = hn[0]
    top_ai = ai[0]
    inert = [r for r in rows if abs(r['swing']) < 0.01]
    byname = dict((r['name'], r) for r in rows)

    def sw_of(names):
        return ', '.join('%s %s' % (n, pp(byname[n]['swing'])) for n in names if n in byname)
    emit('Inert or nearly inert at the prior q10/q90 (|swing on P(E1 by 2050)| < 1 pp; paired SE in pp): ' +
         (', '.join('%s%s (%s, SE %.1f)' % (r['name'], ' [HN]' if r['hn'] else '', pp(r['swing']), 100 * r['se'])
                    for r in inert) or 'none') +
         '. This list is a measured result, not a design claim. Unilateral restraint (now integrated exactly over '
         'each interval): %s. Rate-only levers such as these and r0 stay small for two reasons: unilateral pauses '
         'cut the aggregate rate by only a few percent on average (block 12, "mean unilateral rate cut"), and '
         'learning is per incident, so fewer clusters also means fewer learned clusters and a rate cut partly '
         'delays accidental hits instead of removing them (with constant lethality the probability of ever being '
         'hit does not depend on the rate, because the learned count scales with it; block 10(a)). Salience '
         'response (saturating, no hard '
         'cap; the share of at-risk time it spends at least 90%% saturated is printed in block 12): %s. Tier-2 '
         'salience and jump size: %s. phi_* and delta_e2 move only E2, which this column does not show.'
         % (sw_of(['pi_uni', 'e_uni', 'h_uni_weeks']), sw_of(['w1', 'share_mult_max', 'h_sal']),
            sw_of(['m2', 'e', 'R_max'])))
    emit()
    base = S1['recs'][:blocks[0]['N']]
    s1b = S1['snap'][blocks[0]['N']]
    bd = {}
    for b in blocks:
        bd[b['id']] = {Y: paired(base, b['recs'], 'E1', Y) for Y in (2035, 2050, 2100)}
    # (b) dominant-block statement, from single swings and from the block-level table
    d_s2hn = bd['S2hn'][2050][0]
    d_s2ai = bd['S2ai'][2050][0]
    d_s3hn = bd['S3hn'][2050][0]
    d_ev = bd['noevents'][2050][0]
    resp = [r for r in rows if r['name'] in ('e', 'm2', 'L_nat', 'tau_h', 'c_cap', 'R_max', 'race_persistence',
                                             'sal2_sat', 'L_int', 'D_coord', 'c_cov')]
    mx = max(abs(r['swing']) for r in resp)
    emit('(b) e1 is the largest single lever (swing %s; largest AI-nature swing %s %s; largest human-nature '
         'swing %s %s). Taken as a block, on the same %d S1 draws, S2\'s human-nature overrides alone move P(E1 by '
         '2035 / 2050 / 2100) by %s / %s / %s and S3\'s by %s / %s / %s, against %s / %s / %s for S2\'s AI-nature '
         'overrides alone (e1 ~ U(0.2, 1), C3 in full, s_align low, no capability plateau). Classification of the '
         'safe exit: s_align (verified alignment, a scientific milestone) is AI nature, s_regime (durable '
         'control regime or halt) is human nature and sits in the S2 and S3 human-nature blocks. Response-size '
         'parameters (%s): largest single |swing| %.1f pp, while switching every sampled visible-loss response off '
         'moves P(E1 by 2050) by %s: the response block is %s.'
         % (pp(e1row['swing']), top_ai['name'], pp(top_ai['swing']), top_hn['name'], pp(top_hn['swing']),
            blocks[0]['N'], pp(bd['S2hn'][2035][0]), pp(d_s2hn), pp(bd['S2hn'][2100][0]), pp(bd['S3hn'][2035][0]),
            pp(d_s3hn), pp(bd['S3hn'][2100][0]), pp(bd['S2ai'][2035][0]), pp(d_s2ai), pp(bd['S2ai'][2100][0]),
            ', '.join(r['name'] for r in resp), 100 * mx, pp(d_ev),
            'saturated, not inert (each part is small because the others still respond)' if mx < 0.03 and abs(d_ev) > 0.03
            else ('small at the 2050 horizon' if abs(d_ev) <= 0.03 else 'material')))
    emit()
    emit('(c) Swings are marginal effects at the prior q10/q90, not a decomposition of the headline; the base '
         'P(E1 by 2050) over the first %d draws is %s (full S1: %s). The tercile contrast is the top-minus-'
         'bottom-tercile mean of P(E1 by 2050) within each e1 bin of block 4b, averaged over the bins by draw count '
         '(all %d S1 draws; stratifying on e1 removes chance correlations with the dominant lever); SE in brackets, '
         '"noise" marks |contrast| < 2 SE. It averages over the other parameters\' joint prior instead of fixing '
         'them, so it can differ from the OAT swing where effects interact.'
         % (N_OAT, pc(base50), pc(at(S1['E1'], 2050)), S1['N']))
    emit()
    emit('(d) Block-level sensitivity: the first %d S1 draws rerun with one block replaced (common random numbers; '
         'differences are paired, SE in brackets).' % blocks[0]['N'])
    emit()
    brows = [['S1 (same draws)', pc(at(s1b['E1'], 2035)), pc(at(s1b['E1'], 2050)), pc(at(s1b['E1'], 2100)),
              pc(at(s1b['E2'], 2100)), pc(at(s1b['safe'], 2100)), '-', '-']]
    for b in blocks:
        d50 = bd[b['id']][2050]
        d100 = bd[b['id']][2100]
        brows.append([b['label'], pc(at(b['E1'], 2035)), pc(at(b['E1'], 2050)), pc(at(b['E1'], 2100)),
                      pc(at(b['E2'], 2100)), pc(at(b['safe'], 2100)),
                      '%s (%.1f)' % (pp(d50[0]), 100 * d50[1]), '%s (%.1f)' % (pp(d100[0]), 100 * d100[1])])
    table(['variant', 'P(E1 by 2035)', 'P(E1 by 2050)', 'P(E1 by 2100)', 'P(E2 by 2100)', 'P(safe by 2100)',
           'dP(E1 by 2050)', 'dP(E1 by 2100)'], brows)
    emit('(e) Sign check of the racing and dismissal parameters (S1, first %d draws, each parameter at its '
         'prior q01 and q99 with the others fixed; a + means more racing or dismissal raises risk, as the '
         'human-nature premise expects). lam_conc (lock-in hazard) is added to check E2 for competing-risk '
         'effects; r0 (the measured 2026 incident rate, AI nature) is a reference row, not a racing parameter. The '
         'last E2 column uses the older first-event convention, for comparison.'
         % (len(sign_rows) and N_SIGN_USED[0]))
    emit()
    srows = []
    for r in sign_rows:
        d50, s50 = r['d50']
        d100, s100 = r['d100']
        dE2, sE2 = r['dE2']
        dE2f, sE2f = r['dE2f']
        verdict = []
        for dd, ss in ((d50, s50), (d100, s100), (dE2, sE2)):
            if abs(dd) < 2 * ss:
                verdict.append('0')
            else:
                verdict.append('+' if dd > 0 else '-')
        srows.append([r['name'] + ('' if r['name'] in SIGN_RACING or r['name'] == 'lam_conc' else
                                   ' (reference: 2026 rate, AI nature, not a racing parameter)'),
                      '%s -> %s' % (fnum(r['lo']), fnum(r['hi'])),
                      '%s -> %s' % (pc(r['p50'][0]), pc(r['p50'][1])), '%s (%.2f)' % (pp(d50, 2), 100 * s50),
                      '%s -> %s' % (pc(r['p100'][0]), pc(r['p100'][1])), '%s (%.2f)' % (pp(d100, 2), 100 * s100),
                      '%s (%.2f)' % (pp(dE2, 2), 100 * sE2), '%s (%.2f)' % (pp(dE2f, 2), 100 * sE2f),
                      '%s / %s / %s' % tuple(verdict)])
    table(['parameter', 'q01 -> q99', 'P(E1 by 2050)', 'dP 2050 (SE)', 'P(E1 by 2100)', 'dP 2100 (SE)',
           'dP(E2 by 2100) (SE)', 'dP(E2 at first E1 event by 2100) (SE)',
           'sign E1 2050 / E1 2100 / E2 2100 (0 = within 2 SE)'], srows)
    emit(sign_sentence(sign_rows))
    emit()
    return rows, bd


N_SIGN_USED = [0]


def sign_sentence(sign_rows):
    neg = [r['name'] for r in sign_rows if r['name'] in SIGN_RACING and
           (r['d100'][0] < -2 * r['d100'][1] or r['d50'][0] < -2 * r['d50'][1])]
    neg2 = [r['name'] for r in sign_rows if r['dE2'][0] < -2 * r['dE2'][1]]
    neg2f = [r['name'] for r in sign_rows if r['dE2f'][0] < -2 * r['dE2f'][1]]
    if neg:
        txt = ('More %s lowers P(E1 by 2050) or P(E1 by 2100) beyond 2 SE in this model: incidents also drive '
               'learning (eta), '
               'salience and binding responses, so a higher incident count buys earlier restraint and safe exits '
               '("tombstone" dynamics). This is a property of the response block, not evidence that racing is '
               'safe.' % ' or '.join(neg))
    else:
        txt = 'No racing or dismissal parameter lowers P(E1 by 2050) or P(E1 by 2100) beyond 2 SE.'
    if neg2:
        txt += ' More %s lowers P(E2 by 2100) beyond 2 SE under the population-by-Y definition.' % ' or '.join(neg2)
        if 'lam_conc' in neg2:
            txt += (' For lam_conc this is competing risk: a lock-in (phi_conc ~2%) pre-empts a later accidental E1 '
                    'whose strict share is larger.')
        if [x for x in neg2 if x != 'lam_conc']:
            txt += (' For %s it means that earlier responses and safe exits outweigh the extra hits.'
                    % ' or '.join(x for x in neg2 if x != 'lam_conc'))
    else:
        txt += ' No parameter in this check lowers P(E2 by 2100) beyond 2 SE under the population-by-Y definition.'
    if neg2f:
        txt += (' Under the older first-event convention, more %s lowers P(E2) beyond 2 SE: that convention fixes '
                'the strict share at the capability of the first E1 event, so earlier hits at lower capability '
                'counted as permanently non-strict.' % ' or '.join(neg2f))
    return txt


def print_checks(chk, qtab, S1, S3, blocks, s3v):
    emit('### 10. Closed-form checks')
    emit()
    emit('(a) Per-incident learning identity (internal check; review F25, which proposed it as an established '
         'result, was refuted): constant r, u = 1 from T_START, f_min = 0, no response, eta_eff = 0.1, p_cl = '
         '0.05. P(ever) = 1 - exp(-p_cl/eta_eff) = %s for any r; by 2100 the closed form is '
         '1 - exp(-(p/eta)(1 - exp(-eta r (2100 - 2026.75)))).' % pc(chk['identity'][0][3], 2))
    emit()
    table(['r0 (per yr)', 'engine P(E1 by 2100)', 'closed form by 2100', 'P(ever)', 'diff (pp)'],
          [['%g' % r0, pc(e, 2), pc(c, 2), pc(pe, 2), '%+.3f' % (100 * (e - c))]
           for r0, e, c, pe in chk['identity']])
    emit('The table above counts every cluster on the learning clock (switch learn_from_fatal = True, the '
         'time-based reading of paper sec. 5.4). The default clock counts only clusters that left the path alive, '
         'because the path is conditioned on survival; then dQ/dt = eta r (1 - p e^-Q), the closed form by 2100 is '
         '1 - exp(-Lambda), Lambda = (1/eta)[eta r T - ln(p + (1 - p) e^(eta r T))], and P(ever) = 1 - (1 - p)^(1/eta) '
         '= %s, again for any r:' % pc(chk['surv'][0][3], 2))
    emit()
    table(['r0 (per yr)', 'engine P(E1 by 2100), default clock', 'closed form by 2100', 'P(ever)', 'diff (pp)'],
          [['%g' % r0, pc(e, 2), pc(c, 2), pc(pe, 2), '%+.3f' % (100 * (e - c))]
           for r0, e, c, pe in chk['surv']])
    emit('(b) Paper 5.4 counter-scenario reproduced by the engine: r = 4 fixed, eta_eff = ln2/4 (so the '
         'learning factor halves every year), f_min = 0, pi_det = 1, w = 0, q0 = 1, hazard mode, step at '
         'T_c (clock shifted to T_c = T_START; the closed form is shift-invariant), learn_from_fatal = True.')
    emit()
    table(['p0', 'engine P(ever)', '1 - exp(-r0 p0/ln2)', 'paper 5.4', 'diff (pp)'],
          [['%g%%' % (100 * p0), pc(e, 2), pc(c, 2), {0.01: '5.6%', 0.05: '25.1%', 0.20: '68.5%'}[p0],
            '%+.3f' % (100 * (e - c))] for p0, e, c in chk['counter']])
    med = dict((k, v[2]) for k, v in qtab.items())
    m_eff = 1.0 + (med['m'] - 1.0) * (1.0 - med['rho'])
    p_cl = 1.0 - (1.0 - med['e1']) ** m_eff
    s = med['s_align'] * min(med['s_effort_max'], med['safety_share'] / SIGMA_REF) + med['s_regime']
    hm = med['lam_mis']
    hc = med['lam_conc']
    crow = []
    for lab, r, lf in (('r = r0 median, no learning', med['r0'], 1.0),
                       ('r = r_max median, no learning', med['r_max'], 1.0),
                       ('r = r_max median, learning at floor f_min median', med['r_max'], med['f_min'])):
        p_eff = 1.0 - (1.0 - lf * med['e1']) ** m_eff
        h_acc = r * p_eff
        h = h_acc + hm + hc
        crow.append([lab, '%.4f' % h_acc, '%.4f' % hm, '%.4f' % hc, '%.4f' % s, '%.3f' % (h / (h + s))])
    emit('(c) F04 asymptote h/(h + s) at the S1 prior medians, at full capability (u = 1, gates open), R = 0, '
         'baseline safety share, no salience (root-cause rate r, p_cl at the medians = %.4f, m_eff = %.2f; '
         's = s_align min(s_effort_max, share/SIGMA_REF) + s_regime at the medians, alignment gate open). The '
         'realised safe-exit hazard, with restraint, halts and salience, is printed by route in block 12.'
         % (p_cl, m_eff))
    emit()
    table(['case', 'h_acc', 'h_mis', 'h_conc', 's', 'P(E1 eventually) = h/(h+s)'], crow)
    pcl = []
    for P in S1['params']:
        me = 1.0 + (P['m'] - 1.0) * (1.0 - P['rho'])
        pcl.append(1.0 - (1.0 - P['e1']) ** me)
    emit('(d) Implied per-cluster lethality at full capability, p_cl(u = 1) = 1 - (1 - e1)^m_eff over the '
         'S1 draws: q10 %.2g, median %.2g, q90 %.2g; compare the review\'s grounded per-incident median 6e-4 '
         '(F19), the outside-view-implied 1e-3 and the paper v0.1 floor 1e-2. Share of S1 draws with '
         'p_cl >= 1%%: %s.'
         % (qtl(pcl, 0.1), qtl(pcl, 0.5), qtl(pcl, 0.9), pc(sum(1 for x in pcl if x >= 0.01) / float(len(pcl)))))
    emit()
    k50 = [r['K'][YP[2050]] for r in S1['recs']]
    k75 = [r['K'][YP[2075]] for r in S1['recs']]
    sub50 = [r['K'][YP[2050]] * P['m'] for r, P in zip(S1['recs'], S1['params'])]
    emit('(e) Expected root-cause clusters from 2026.75 (restrained, integral of r_root dt) in S1: K(2050) mean '
         '%.1f (q10 %.1f, median %.1f, q90 %.1f), i.e. %.0f sub-incidents on average (K x m); K(2075) mean %.1f '
         '(median %.1f); r0 prior median %.1f and r_max prior median %.1f root-cause clusters per year. Count x '
         'lethality is what the data constrain: a 10x larger count with a 10x smaller e1 gives a similar hazard.'
         % (mean(k50), qtl(k50, 0.1), qtl(k50, 0.5), qtl(k50, 0.9), mean(sub50), mean(k75), qtl(k75, 0.5),
            med['r0'], med['r_max']))
    emit()
    s1b = S1['snap'][blocks[0]['N']]
    s3b = S3['snap'][s3v[0]['N']]
    bS1 = dict((b['id'], b) for b in blocks)
    bS3 = dict((b['id'], b) for b in s3v)
    lrows = []
    for lab, base, smp, lgr in (('S1', s1b, bS1['S1sampled'], bS1['S1lograte']),
                                ('S3', s3b, bS3['S3sampled'], bS3['S3lograte'])):
        for name, res in (('mean field (headline): exp(-Q), factor 1 - eta_eff per cluster', base),
                          ('sampled Poisson learned counts, factor 1 - eta_eff per cluster', smp),
                          ("'lograte' reading: factor exp(-eta_eff) per cluster, exact expectation", lgr)):
            lrows.append(['%s (N = %d)' % (lab, res['N'] if 'N' in res else blocks[0]['N']), name,
                          pc(at(res['E1'], 2035), 2), pc(at(res['E1'], 2050), 2), pc(at(res['E1'], 2100), 2),
                          pc(at(res['E2'], 2100), 2)])
    d_s1 = [paired(S1['recs'][:blocks[0]['N']], bS1[k]['recs'], 'E1', 2050) for k in ('S1sampled', 'S1lograte')]
    d_s3 = [paired(S3['recs'][:s3v[0]['N']], bS3[k]['recs'], 'E1', 2050) for k in ('S3sampled', 'S3lograte')]
    emit('(f) Learning-count check (note N4). With eta defined as the fraction of the remaining removable hazard '
         'closed per learned cluster (factor 1 - eta_eff), the Poisson probability generating functional gives '
         'E[product of (1 - eta_i)] = exp(-Q) exactly, so the mean-field learning factor equals the expected factor; '
         'the remaining gap is Jensen on survival (E[exp(-Lambda)] >= exp(-E[Lambda])), which makes the mean field '
         'overstate P(hit) slightly. Reading eta instead as a log-reduction per cluster (factor exp(-eta_eff)), '
         'the expected factor exp(-(1 - e^-eta) N) is larger and P(hit) rises. Same draws, paired differences in '
         'P(E1 by 2050) against the mean field: S1 sampled %s (SE %.2f), S1 lograte %s (SE %.2f); S3 sampled %s '
         '(SE %.2f), S3 lograte %s (SE %.2f).'
         % (pp(d_s1[0][0], 2), 100 * d_s1[0][1], pp(d_s1[1][0], 2), 100 * d_s1[1][1],
            pp(d_s3[0][0], 2), 100 * d_s3[0][1], pp(d_s3[1][0], 2), 100 * d_s3[1][1]))
    emit()
    table(['scenario', 'learning count', 'P(E1 by 2035)', 'P(E1 by 2050)', 'P(E1 by 2100)', 'P(E2 by 2100)'],
          lrows)
    return {'S1': d_s1, 'S3': d_s3}


def effect_text(e, name):
    d = e['d']
    txt = '%s / %s / %s (SE %.2f / %.2f / %.2f; N = %d per side)' % (
        pp(d[2035][0], 2), pp(d[2050][0], 2), pp(d[2100][0], 2), 100 * d[2035][1], 100 * d[2050][1], 100 * d[2100][1],
        e['N'])
    if e['inert']:
        txt += '; inert in this model: ' + HN_INERT_REASON.get(name, '(reason not recorded)')
    return txt


def print_param_lists(qtab, hne, rec_status):
    emit('### 11. Parameters, constants and sources')
    emit()
    emit('Prior notation: LogN(med m, sd s) = m exp(s N(0,1)), so s is the sd of ln x; LogU(lo, hi) = exp(U(ln lo, '
         'ln hi)); 1+LogN = 1 + LogN; Beta(a, b); U(lo, hi); "= v" is a point value.')
    emit()
    emit('Human-nature parameters (%d): the behaviour of labs, states, firms and publics. S1 uses the '
         'prior; "-" means the scenario keeps the S1 prior. Source keys resolve in the source table below. '
         'Classification criterion: the measured 2026 state of the technology and of incident detection (r0, '
         'pi_det) is AI nature; how labs, states, firms and publics behave from now on is human nature. The '
         '"measured effect" column is P(E1 by 2035 / 2050 / 2100) with the parameter at its prior q99 minus at its '
         'q01, the other parameters keeping their S1 draws (common random numbers on the first N S1 draws; the '
         'sign-check parameters reuse those runs); a parameter whose |effect| is below %.0f pp at all three '
         'horizons is marked "inert in this model" with the reason. The same values are recorded in PARAMS '
         '(HN_EFFECT_RECORD) and params.yaml (human_nature: effect); record check: %s.'
         % (N_HN, 100 * INERT_PP, rec_status))
    emit()
    rows = []
    for s in PARAMS:
        if not s['hn']:
            continue
        rows.append([s['name'], s['meaning'], prior_str(s['prior']), s['tag'], s['rationale'],
                     prior_str(s['overrides'].get('S2')), prior_str(s['overrides'].get('S3')),
                     prior_str(s['overrides'].get('S1side')),
                     '%s -> %s' % (fnum(hne[s['name']]['lo']), fnum(hne[s['name']]['hi'])),
                     effect_text(hne[s['name']], s['name'])])
    table(['name', 'meaning', 'prior', 'tag', 'rationale (sources)', 'S2 override', 'S3 override',
           'S1-side-high override', 'q01 -> q99', 'measured effect dP(E1 by 2035 / 2050 / 2100), q01 -> q99'],
          rows)
    inert = [n for n, e in hne.items() if e['inert']]
    emit('Inert in this model (|effect| < %.0f pp at 2035, 2050 and 2100): %s.' % (100 * INERT_PP, ', '.join(inert)
                                                                                  or 'none'))
    emit()
    emit('AI-nature, capability and channel parameters (%d):' % (len(PARAMS) - N_HN))
    emit()
    rows = []
    for s in PARAMS:
        if s['hn']:
            continue
        rows.append([s['name'], s['meaning'], prior_str(s['prior']), s['tag'], s['rationale'],
                     prior_str(s['overrides'].get('S2')), prior_str(s['overrides'].get('S3')),
                     prior_str(s['overrides'].get('S1cal')), prior_str(s['overrides'].get('S1side'))])
    table(['name', 'meaning', 'prior', 'tag', 'rationale (sources)', 'S2 override', 'S3 override',
           'S1-cal override', 'S1-side-high override'], rows)
    emit('Fixed constants:')
    emit()
    table(['constant', 'value', 'tag', 'rationale'],
          [[c['name'], '%g' % c['value'], c['tag'], c['rationale']] for c in CONSTANTS])
    emit('Prior quantiles (q01 / q10 / q50 / q90 / q99 from %d draws; the OAT uses q10/q90, the sign check '
         'q01/q99): ' % N_PRIOR_Q +
         '; '.join('%s %s' % (n, '/'.join(fnum(x) for x in qtab[n])) for n in PARAM_NAMES))
    emit()
    emit('Switches: ' + '; '.join('%s = %r (%s)' % (k, v, SWITCH_DOCS[k]) for k, v in SWITCHES_DEFAULT.items()))
    emit()
    emit('Sources cited by key (also: python3 model/mc.py --sources, and params.yaml sources:):')
    emit()
    table(['key', 'source', 'URL'], [[k, t, u] for k, t, u in SOURCES])
    emit('Review-finding index (the 2026-10-10 review of paper v0.1; ids used in mc.py, model.py and the '
         'rationales):')
    emit()
    table(['id', 'status', 'title'], [[i, s, t] for i, s, t in REVIEW_FINDINGS])


def pooled(recs, field, num, den='S'):
    """Survival-weighted pooled mean per period: sum_j num_j[p] / sum_j den_j[p]."""
    out = []
    for p in range(4):
        n = sum(r['diag'][num][p] for r in recs)
        d = sum(r['diag'][den][p] for r in recs)
        out.append(n / d if d > 0 else float('nan'))
    return out


def pooled_all(recs, num, den='S'):
    n = sum(sum(r['diag'][num]) for r in recs)
    d = sum(sum(r['diag'][den]) for r in recs)
    return n / d if d > 0 else float('nan')


def pooled_span(recs, nums, periods, den='S'):
    n = sum(r['diag'][x][p] for r in recs for x in nums for p in periods)
    d = sum(r['diag'][den][p] for r in recs for p in periods)
    return n / d if d > 0 else float('nan')


def diag_values(res):
    recs = res['recs']
    N = float(len(recs))
    P = res['params']
    d = {}
    tcs = [r['Tc'] for r in recs]
    d['Tc'] = (qtl(tcs, 0.1), qtl(tcs, 0.5), qtl(tcs, 0.9), sum(1 for t in tcs if t < T_START) / N)
    d['Tc_by'] = [sum(1 for t in tcs if t <= Y) / N for Y in (2030, 2050, 2070, 2100)] + \
                 [sum(1 for t in tcs if t == INF) / N]
    tx = [r['t_cross'] for r in recs if r['t_cross'] is not None]
    d['tcross'] = (qtl(tx, 0.5) if tx else None, len(tx) / N)
    tc0 = [p['tau_c0_days'] for p in P]
    td0 = [p['tau_d0_days'] for p in P]
    sm = [a + b for a, b in zip(tc0, td0)]
    d['tau'] = ((qtl(tc0, 0.1), qtl(tc0, 0.5), qtl(tc0, 0.9)), (qtl(td0, 0.1), qtl(td0, 0.5), qtl(td0, 0.9)),
                (qtl(sm, 0.1), qtl(sm, 0.5), qtl(sm, 0.9)))
    ux = [r['u_x0'] for r in recs]
    # 2026 record: N_2026 contained root-cause clusters, each of m_eff effective sub-incidents
    w = [(1.0 - u) ** (N_2026 * (1.0 + (p['m'] - 1.0) * (1.0 - p['rho']))) for u, p in zip(ux, P)]
    sw = sum(w)
    sw2 = sum(x * x for x in w)
    ess = sw * sw / (N * sw2) if sw2 > 0 else float('nan')
    wp = [sum(wi * r['E1'][YP[Y]] for wi, r in zip(w, recs)) / sw if sw > 0 else float('nan')
          for Y in (2035, 2050, 2100)]
    wnear = [sum(wi * r['E1'][YP[Y]] for wi, r in zip(w, recs)) / sw if sw > 0 else float('nan')
             for Y in (2027, 2028, 2029, 2030)]
    unear = [mean(r['E1'][YP[Y]] for r in recs) for Y in (2027, 2028, 2029, 2030)]
    d['u_x0'] = (qtl(ux, 0.1), qtl(ux, 0.5), qtl(ux, 0.9), ess, wp, wnear, unear)
    ys = [2028, 2030, 2035, 2050, 2075, 2100]
    d['R'] = [mean(r['R'][YP[Y]] for r in recs) for Y in ys]
    d['rr'] = [mean(r['rroot'][YP[Y]] for r in recs) for Y in ys]
    d['pause'] = [sum(1 for r in recs if r['t_pause'] is not None and r['t_pause'] <= Y) / N
                  for Y in (2028, 2030, 2035, 2050, 2100)]
    first_p = [r for r in recs if r['t_pause'] is not None]
    d['pause_before_t2'] = (sum(1 for r in first_p if r['t_t2'] is None or r['t_pause'] < r['t_t2'])
                            / float(len(first_p)) if first_p else float('nan'))
    d['pause10_pre_t2'] = sum(1 for r in recs if r['t_pause'] is not None and r['t_pause'] <= 2036.75
                              and (r['t_t2'] is None or r['t_pause'] < r['t_t2'])) / N
    d['rp_mean'] = mean(p['race_persistence'] for p in P)
    d['Rgt'] = [sum(1 for r in recs if r['R'][YP[Y]] > 0.05) / N for Y in (2030, 2035, 2050)]
    tb = [r['t_bind'] for r in recs if r['t_bind'] is not None]
    d['bind'] = (mean(tb) if tb else None, len(tb) / N)
    d['nvis'] = mean(r['n_vis'] for r in recs)
    d['nunc'] = mean(r['n_unc'] for r in recs)
    d['ntier2'] = mean(r['n_tier2'] for r in recs)
    d['npause'] = mean(r['n_pause'] for r in recs)
    d['nuni'] = mean(r['n_uni'] for r in recs)
    tot = sum(r['E1'][YP[2100]] for r in recs)
    d['sat'] = sum(r['acc_sat'] for r in recs) / tot if tot > 0 else float('nan')
    d['stopped'] = sum(1 for r in recs if r['stopped']) / N
    d['N50'] = mean(r['N50'] for r in recs)
    # safe-exit hazard and its routes (survival-weighted), pause time, saturation and caps, salience
    d['s'] = pooled(recs, 'diag', 'safe')
    d['s_al'] = pooled(recs, 'diag', 'al')
    d['s_rg'] = pooled(recs, 'diag', 'rg')
    d['s_rk'] = pooled(recs, 'diag', 'rk')
    d['s_h'] = pooled(recs, 'diag', 'sh')
    d['reg_cal'] = pooled_span(recs, ('rg', 'rk', 'sh'), (0, 1, 2))
    d['tot_cal'] = pooled_span(recs, ('safe',), (0, 1, 2))
    d['uni'] = pooled(recs, 'diag', 'uni')
    d['pause_t'] = pooled(recs, 'diag', 'pause')
    d['sal'] = pooled(recs, 'diag', 'sal')
    d['sal2'] = pooled(recs, 'diag', 'sal2')
    d['bind_caps'] = (pooled_all(recs, 'bm'), pooled_all(recs, 'bf'), pooled_all(recs, 'bs'), pooled_all(recs, 'be'),
                      pooled_all(recs, 'bd'))
    d['s_align_mean'] = mean(p['s_align'] for p in P)
    d['s_regime_mean'] = mean(p['s_regime'] for p in P)
    ps = [sum(r['diag']['ps'][i] for r in recs) / N for i in range(4)]
    d['ps_routes'] = ps
    tot_safe = sum(ps)
    d['ps_paused'] = (sum(r['diag']['ps_paused'] for r in recs) / N) / tot_safe if tot_safe > 0 else float('nan')
    d['ps_pre'] = (sum(r['diag']['ps_pre'] for r in recs) / N) / tot_safe if tot_safe > 0 else float('nan')
    d['ps_pre_al'] = (sum(r['diag']['ps_pre_al'] for r in recs) / N) / tot_safe if tot_safe > 0 else float('nan')
    d['ps_pre_reg'] = (sum(r['diag']['ps_pre_reg'] for r in recs) / N) / tot_safe if tot_safe > 0 else float('nan')
    return d


def print_diagnostics(results, S1, S1c):
    emit('### 12. Diagnostics')
    emit()
    dv = [diag_values(r) for r in results]
    keys = [r['key'] for r in results]

    def per(vals, fmt='%.3f'):
        return ' / '.join('n/a' if x != x else fmt % x for x in vals)
    rows = [
        ['implied T_c q10 / q50 / q90 (no feedback; plateau-aware)'] + ['%s / %s / %s' % (yr(d['Tc'][0]), yr(d['Tc'][1]), yr(d['Tc'][2])) for d in dv],
        ['P(T_c <= 2030 / 2050 / 2070 / 2100); share never (capability plateaus below the threshold)'] + [' / '.join(pc(x) for x in d['Tc_by']) for d in dv],
        ['share of draws with T_c before 2026.75'] + [pc(d['Tc'][3]) for d in dv],
        ['realised threshold crossing (median; share crossing by 2100)'] + ['%s; %s' % (yr(d['tcross'][0]), pc(d['tcross'][1])) for d in dv],
        ['containment median at H0 tau_c0 (days) q10 / q50 / q90'] + ['%.2g / %.2g / %.2g' % d['tau'][0] for d in dv],
        ['dwell median at H0 tau_d0 (days) q10 / q50 / q90'] + ['%.2g / %.2g / %.2g' % d['tau'][1] for d in dv],
        ['dwell + containment (days) q10 / q50 / q90'] + ['%.3g / %.3g / %.3g' % d['tau'][2] for d in dv],
        ['u(x0) today q10 / q50 / q90'] + ['%.2g / %.2g / %.2g' % d['u_x0'][:3] for d in dv],
        ['2026-record likelihood ESS/N (w = (1 - u(x0))^(4 m_eff): 4 contained root-cause clusters of m_eff effective sub-incidents)'] + ['%.3f' % d['u_x0'][3] for d in dv],
        ['record-weighted P(E1 by 2027 / 2028 / 2029 / 2030) (unweighted, as in block 4, in brackets)'] + ['%s (%s)' % (' / '.join(pc(x, 2) for x in d['u_x0'][5]), ' / '.join(pc(x, 2) for x in d['u_x0'][6])) for d in dv],
        ['record-weighted P(E1 by 2035 / 2050 / 2100)'] + [' / '.join(pc(x) for x in d['u_x0'][4]) for d in dv],
        ['mean R at 2028 / 2030 / 2035 / 2050 / 2075 / 2100'] + [per(d['R']) for d in dv],
        ['mean restrained r_root (root-cause clusters per yr) at the same years'] + [per(d['rr'], '%.2f') for d in dv],
        ['mean unilateral rate cut 1 - m_avg (survival-weighted) %s' % ' / '.join(('2026.75-30', '2030-35', '2035-50', '2050-2100'))] + [per(d['uni'], '%.3f') for d in dv],
        ['realised safe-exit hazard s (per yr, survival-weighted), same periods'] + [per(d['s'], '%.4f') for d in dv],
        ['  of which verified alignment s_align min(s_effort_max, share/SIGMA_REF) G(x_def - x_c) (AI nature)'] + [per(d['s_al'], '%.4f') for d in dv],
        ['  of which regime base rate s_regime (human nature)'] + [per(d['s_rg'], '%.4f') for d in dv],
        ['  of which restraint boost s_regime kappa_s R'] + [per(d['s_rk'], '%.4f') for d in dv],
        ['  of which s_halt while a coordinated pause is in force'] + [per(d['s_h'], '%.4f') for d in dv],
        ['CALIBRATION: realised regime route s_regime (1 + kappa_s R) + s_halt [paused], 2026.75-2050 (anchor %.3f-%.3f per yr)' % SAFE_ANCHOR] + ['%.4f (%s)' % (d['reg_cal'], 'within' if SAFE_ANCHOR[0] <= d['reg_cal'] <= SAFE_ANCHOR[1] else ('below' if d['reg_cal'] < SAFE_ANCHOR[0] else 'above')) for d in dv],
        ['realised total safe-exit hazard incl. alignment, 2026.75-2050'] + ['%.4f' % d['tot_cal'] for d in dv],
        ['mean s_align / s_regime over the draws'] + ['%.4f / %.4f' % (d['s_align_mean'], d['s_regime_mean']) for d in dv],
        ['P(safe by 2100) by route: alignment / regime base / restraint boost / halt'] + [' / '.join(pc(x) for x in d['ps_routes']) for d in dv],
        ['share of P(safe by 2100) entered while a coordinated pause was in force'] + [pc(d['ps_paused']) for d in dv],
        ['share of P(safe by 2100) entered before the path crossed its own threshold: total (alignment / regime route)'] + ['%s (%s / %s)' % (pc(d['ps_pre']), pc(d['ps_pre_al']), pc(d['ps_pre_reg'])) for d in dv],
        ['race_persistence prior mean = model-implied P(no coordinated pause in 10 yr | no tier-2 event)'] + ['%.3f' % d['rp_mean'] for d in dv],
        ['simulated P(a coordinated pause by 2036.75 that precedes every tier-2 event); by construction <= 1 - the row above'] + ['%.3f' % d['pause10_pre_t2'] for d in dv],
        ['P(first coordinated pause by 2028 / 2030 / 2035 / 2050 / 2100)'] + [' / '.join(pc(x) for x in d['pause']) for d in dv],
        ['share of first coordinated pauses that precede every tier-2 event'] + [pc(d['pause_before_t2']) for d in dv],
        ['share of at-risk time under a coordinated pause (survival-weighted, same periods)'] + [per(d['pause_t'], '%.3f') for d in dv],
        ['mean salience Sal (HF-class units, survival-weighted, same periods)'] + [per(d['sal'], '%.2f') for d in dv],
        ['mean tier-2 salience Sal2 (same periods)'] + [per(d['sal2'], '%.2f') for d in dv],
        ['share of at-risk time 2026.75-2100: salience response >= 90% saturated / s_effort_max binding / SHARE_MAX binding / ETA_EFF_MAX binding / dwell cap TAU_D_CAP_D binding'] + [' / '.join(pc(x) for x in d['bind_caps']) for d in dv],
        ['share with binding restraint R > 0.05 at 2030 / 2035 / 2050'] + [' / '.join(pc(x) for x in d['Rgt']) for d in dv],
        ['mean year of first binding measure (share of paths with one; the HF jump happens with probability p_bind_first)'] + ['%s (%s)' % (yr(d['bind'][0]), pc(d['bind'][1])) for d in dv],
        ['mean visible losses (Poisson count) / of which uncontainable / tier-2 / unilateral pauses / coordinated pauses, by 2100'] + ['%.1f / %.1f / %.2f / %.1f / %.2f' % (d['nvis'], d['nunc'], d['ntier2'], d['nuni'], d['npause']) for d in dv],
        ['mean expected learning count N_learn at 2050'] + ['%.1f' % d['N50'] for d in dv],
        ['share of E1 mass by 2100 at log2(H/H0) > 14 (METR saturation)'] + [pc(d['sat']) for d in dv],
        ['share of paths stopped early (P(still at risk) < 1e-9; state frozen, capability continues for E2)'] + [pc(d['stopped']) for d in dv],
    ]
    table(['diagnostic'] + keys, rows)
    emit('Survival-weighted means pool sum_j integral x S dt / sum_j integral S dt over the period, so they '
         'describe the at-risk world. 2026 record for the anchor: HF side dwell 2.5-4.5 d + containment hours; '
         'OpenAI side dwell 9-11 d (74 d from the first warning) + containment 1-16 d. The record-likelihood '
         'weights are shown for transparency and are not applied to the headline, because the containment-time '
         'prior already encodes the same 2026 observations; they matter most in the near-term columns, where the '
         'unweighted cells lean high. Visible losses are a Poisson count per interval '
         '(K ~ Poisson(nu dt), several per interval when the rate is high); event counts are unweighted means over '
         'simulated paths (events are sampled on the surviving path; a path stops once its still-at-risk weight '
         'is below 1e-9). The calibration row checks the human-nature regime route against the historical '
         'durable-equilibrium rate, which already includes post-loss responses; the alignment route has no '
         'historical analogue and is reported separately.')
    emit()
    n = S1c['N']
    lines = []
    for Y in (2035, 2050, 2100):
        dm, se = paired(S1['recs'][:n], S1c['recs'], 'E1', Y)
        lines.append('%d %s (paired SE %.1f pp)' % (Y, pp(dm), 100 * se))
    emit('Structural uncertainty, S1-containment minus S1 (hazard mode) on the same %d draws: P(E1 by Y) '
         % n + '; '.join(lines) + '.')
    emit()
    return dict(zip(keys, dv))


def print_footer(S1, oat_sorted, bd, blocks, sign_rows, dv, S3, s3v, cond, lchk, R, ov_rows):
    emit('### 13. How to read this output')
    emit()
    cal = R['S1-cal']
    cal_acc = channel_curve(cal, 'acc')
    cal_side = channel_curve(cal, 'side')
    ov1 = ov_rows[('E1', 'raw')]
    s1_acc = channel_curve(S1, 'acc')
    s1_side = channel_curve(S1, 'side')
    e1row = [r for r in oat_sorted if r['name'] == 'e1'][0]
    nos = block_by_id(blocks, 'nosafe')
    anc = block_by_id(blocks, 'anchor')
    npl = block_by_id(blocks, 'noplateau')
    s1b = S1['snap'][blocks[0]['N']]
    d1 = dv['S1']
    b3 = dict((v['id'], v) for v in s3v)
    s3b = S3['snap'][s3v[0]['N']]
    notes = [
        'N1 E1 is extinction or permanent loss of collective self-determination; E2 is strict extinction, '
        'population < 5,000 by Y (the XPT definition). An accidental E1 world becomes E2 once its capability, which '
        'keeps growing after the E1 event, passes its strict-extinction level (phi_acc at the capability reached '
        'by Y); misuse and lock-in E1 events carry the constant shares phi_mis and phi_conc. E2 is a subset of E1 '
        'by construction. The older convention (E2 decided at the first E1 event) is a lower bound and is printed '
        'in block 5. S0\'s E2 row equals E1 because v0.1 does not distinguish them.',
        'N2 All means are prior-predictive; "P(no hit by 2100)" includes safe exits, which are printed separately.',
        'N3 H_0 cancels out of the threshold, but b_c carries the same horizon-to-evasion mapping uncertainty '
        'under another name and alone moves T_c by years; Hc_override_hours is the switch for a fixed hour count. '
        'Premise C1 is now a parameter: x_plateau (capability saturation level) gives P(T_c <= 2050) = %s and a '
        'share %s of S1 draws whose capability never reaches the threshold (block 12); x_plateau = infinity '
        '(premise C1) on the same draws gives P(E1 by 2100) = %s instead of %s.'
        % (pc(d1['Tc_by'][1]), pc(d1['Tc_by'][4]), pc(at(npl['E1'], 2100)), pc(at(s1b['E1'], 2100))),
        'N4 Learning uses the expected count of learned clusters (mean field) and is per incident: each learned '
        'cluster closes the fraction eta_eff of the remaining removable hazard at the rate in force when it is '
        'learned, so learned failure modes stay closed. With that definition the mean-field factor exp(-Q) is the '
        'exact expected learning factor under Poisson counts; it is an approximation, not a bound, because Jensen '
        'on survival makes the mean field overstate P(hit) slightly (sampled counts: S1 %s, S3 %s on P(E1 by '
        '2050)), while reading eta as a log-reduction per cluster raises it (S1 %s, S3 %s; block 10(f)). The '
        'printed band excludes incident-count variance.'
        % (pp(lchk['S1'][0][0], 2), pp(lchk['S3'][0][0], 2), pp(lchk['S1'][1][0], 2), pp(lchk['S3'][1][0], 2)),
        'N5 e1 (an [Assumption] spanning 3.7 orders of magnitude) is the largest single lever: OAT swing %s on '
        'P(E1 by 2050); draws with e1 >= 1e-2 (prior mass %s) carry %s of S1\'s P(E1 by 2050), and block 4b prints '
        'the result per e1 decade bin (an average over the decade) and at eight point values of e1 (local means), '
        'so a reader can substitute their own e1 from the point-value table. The S1 level is set mainly by e1 and the '
        'side-channel priors lam_mis, lam_conc and L_conc: of P(E1 by 2100) = %s, the accident channel carries %s '
        'and misuse + lock-in %s (with the former, higher-hazard side-channel priors, S1-side-high, P(E1 by 2100) '
        '= %s). The safe-exit block '
        'reduces it: on the block-9(d) draws P(E1 by 2100) is %s with all routes, %s with the safe exit limited '
        'to the historical regime anchor (s_regime only) and %s with no safe exit. The realised regime route '
        '2026.75-2050 is %.4f per year against the historical %.3f-%.3f (block 12). The response block also '
        'matters: switching every sampled visible-loss response off adds %s to P(E1 by 2050), switching the '
        'coordinated pause off adds %s. Human nature taken as a block moves P(E1 by 2050) by %s (S2 human nature '
        'only, now including s_regime) and %s (S3 human nature only), against %s for S2\'s AI-nature overrides '
        '(including s_align and no capability plateau). S1-cal calibrates only e1 (LogU(1e-4, 1e-2), the range '
        'implied by inverting the outside-view anchors in the v0.1 single-channel model) and does not reproduce the '
        'outside-view level: its P(E1 by 2100) is %s against the E1 pool\'s %s. Its accident channel alone, the '
        'channel the E1 pool\'s anchors mostly cover, gives %s by 2050 and %s by 2100 against the pool\'s %s and %s '
        '(block 6); its misuse and lock-in channels add %s and %s, which rest on lam_mis, lam_conc and L_conc '
        '([Assumption]) '
        'and which the pool does not cover. The S2 row is premise-driven by construction.'
        % (pp(e1row['swing']), pc(cond['prior_mass_hi']), pc(cond['share50_hi']),
           pc(at(S1['E1'], 2100)), pc(s1_acc[YP[2100]]), pc(s1_side[YP[2100]]), pc(at(R['S1-side-high']['E1'], 2100)),
           pc(at(s1b['E1'], 2100)),
           pc(at(anc['E1'], 2100)), pc(at(nos['E1'], 2100)), d1['reg_cal'], SAFE_ANCHOR[0], SAFE_ANCHOR[1],
           pp(bd['noevents'][2050][0]), pp(bd['nopause'][2050][0]), pp(bd['S2hn'][2050][0]),
           pp(bd['S3hn'][2050][0]), pp(bd['S2ai'][2050][0]),
           pc(at(cal['E1'], 2100)), pc(ov1[YP[2100]]), pc(cal_acc[YP[2050]]), pc(cal_acc[YP[2100]]),
           pc(ov1[YP[2050]]), pc(ov1[YP[2100]]), pc(cal_side[YP[2050]]), pc(cal_side[YP[2100]])),
        'N6 No quantile is extrapolated beyond 2100. The safe state is fully absorbing: a control equilibrium that '
        'later fails is not representable. The verified-alignment route opens only with the threshold gate; the '
        'regime route (durable control regime or deliberate halt) has no capability gate, because a regime can '
        'precede the catastrophe it prevents (the Montreal Protocol followed the 1985 ozone-hole observation, not a '
        'casualty event), and %s of '
        'S1\'s P(safe by 2100) is entered before the path crosses its own threshold (alignment %s, regime route '
        '%s). Gradual '
        'disempowerment is folded into lam_conc without a drift term; channel shares for misuse and concentration '
        'are [Assumption]-driven; cross-lab common-cause coupling enters only through rho and the aggregate '
        'unilateral multiplier m_uni. The HF incident leads to a binding national measure with probability '
        'p_bind_first (%s; %s of S1 paths have a first binding measure by 2100, block 12); later tier-2 losses '
        'always produce a binding jump. The defense ratio a acts only on the containment clock; misuse, lock-in '
        'and strict-extinction progression use raw capability.'
        % (pc(d1['ps_pre']), pc(d1['ps_pre_al']), pc(d1['ps_pre_reg']),
           prior_str(PARAM_BY_NAME['p_bind_first']['prior']), pc(d1['bind'][1])),
        'N7 Tags: [Fact] is kept for directly measured constants; every sampled prior is [Inference] (a '
        'distribution read from named analogues) or [Assumption] (no data). Every source key in a rationale '
        'resolves to a public URL in block 11 (and params.yaml sources:); review F-ids resolve in the review '
        'index. Items marked [unverified] or [not fetched] were not confirmed at a primary source and must be '
        'sourced before the paper cites them as facts.',
        'N8 Units: r0 and r_max are root-cause cluster rates (block 10(e) compares them with the simulated count '
        'K); e1 and q0 are per effective sub-incident and compound over m_eff per cluster (block 10(d)).',
        'N9 Sign check (block 9(e)): ' + sign_sentence(sign_rows),
        'N10 S3 (optimistic coordination) keeps S1\'s AI-nature priors, including s_align; its optimism is in '
        'human nature: on its first %d draws P(E1 by 2100) is %s with all routes, %s with the safe exit limited to '
        'its regime anchor (s_regime only) and %s with no safe exit; S3 reaches P(safe by 2030) = %s with a '
        'realised safe-exit hazard of %.3f per year in 2026.75-2030 (regime route %.4f per year 2026.75-2050).'
        % (s3v[0]['N'], pc(at(s3b['E1'], 2100)), pc(at(b3['S3anchor']['E1'], 2100)),
           pc(at(b3['S3nosafe']['E1'], 2100)), pc(at(S3['safe'], 2030)), dv['S3']['s'][0], dv['S3']['reg_cal']),
        'N11 Saturation and caps (all listed in block 11 and params.yaml): in S1 the salience response is at least '
        '90%% saturated during %s of at-risk time (a smooth saturation at share_mult_max, not a hard cap) and the '
        'effort bound s_effort_max binds during %s; the constants SHARE_MAX and ETA_EFF_MAX bind during %s and %s '
        '(S3: %s and %s); the dwell-time cap TAU_D_CAP_D binds during %s (S3: %s).'
        % (pc(d1['bind_caps'][0]), pc(d1['bind_caps'][1]), pc(d1['bind_caps'][2]), pc(d1['bind_caps'][3]),
           pc(dv['S3']['bind_caps'][2]), pc(dv['S3']['bind_caps'][3]), pc(d1['bind_caps'][4]),
           pc(dv['S3']['bind_caps'][4])),
    ]
    for nline in notes:
        emit('- ' + nline)
    emit()
    N = S1['N']
    ses = []
    for Y in (2035, 2050, 2100):
        xs = [r['E1'][YP[Y]] for r in S1['recs']]
        ses.append('P(E1 by %d) %.2f pp' % (Y, 100 * se_of(xs)))
    xs = [r['E2'][YP[2100]] for r in S1['recs']]
    ses.append('P(E2 by 2100) %.2f pp' % (100 * se_of(xs)))
    emit('Monte Carlo standard error of the S1 headline cells (sd across draws / sqrt(N), N = %d): %s.'
         % (N, '; '.join(ses)))
    emit()
    ass = [r for r in oat_sorted if r['tag'] == '[Assumption]']
    carry = [r for r in ass if abs(r['swing']) >= 0.01] or ass[:5]
    emit('[Assumption] parameters that carry the result (|OAT swing on P(E1 by 2050)| >= 1 pp, largest '
         'first): ' + ', '.join('%s (%s)' % (r['name'], pp(r['swing'])) for r in carry) + '.')
    emit()


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def parse_args(argv):
    seed = DEFAULT_SEED
    n_main, n_var, n_var2, n_oat = N_MAIN_DEFAULT, N_VAR_DEFAULT, N_VAR2_DEFAULT, N_OAT_DEFAULT
    n_block, n_sign, n_hne = N_BLOCK_DEFAULT, N_SIGN_DEFAULT, N_HNE_DEFAULT
    quick = False
    mode = 'run'
    i = 1
    while i < len(argv):
        a = argv[i]
        if a == '--seed':
            seed = int(argv[i + 1])
            i += 1
        elif a == '--quick':
            quick = True
        elif a == '--n':
            n_main = int(argv[i + 1])
            n_var = int(round(0.6 * n_main))
            n_var2 = int(round(0.3 * n_main))
            n_oat = max(2, int(round(0.012 * n_main)))
            n_block = max(2, int(round(0.12 * n_main)))
            n_sign = max(2, int(round(0.06 * n_main)))
            n_hne = max(2, int(round(0.016 * n_main)))
            i += 1
        elif a == '--emit-yaml':
            mode = 'yaml'
        elif a == '--sources':
            mode = 'sources'
        elif a in ('-h', '--help'):
            print(__doc__)
            sys.exit(0)
        else:
            raise SystemExit('unknown argument %r (use --quick, --seed S, --n N, --emit-yaml, --sources)' % a)
        i += 1
    if quick:
        n_main, n_var, n_var2, n_oat = max(1, n_main // 4), max(1, n_var // 4), max(1, n_var2 // 4), max(2, n_oat // 4)
        n_block, n_sign, n_hne = max(2, n_block // 4), max(2, n_sign // 4), max(2, n_hne // 4)
    n_oat = min(n_oat, n_main)
    n_var = min(n_var, n_main)
    n_var2 = min(n_var2, n_main)
    n_block = min(n_block, n_main)
    n_sign = min(n_sign, n_main)
    n_hne = min(n_hne, n_main)
    return seed, n_main, n_var, n_var2, n_oat, n_block, n_sign, n_hne, quick, mode


def pch(x):
    """Headline percent: whole percent from 1% up (review F17: the inputs do not support more precision)."""
    if x != x:
        return 'n/a'
    return '%.0f%%' % (100.0 * x) if x >= 0.01 else '%.1f%%' % (100.0 * x)


def pph(x):
    """Headline absolute difference or share in percentage points, at the precision of pch."""
    if x != x:
        return 'n/a'
    return '%.0f pp' % (100.0 * x) if abs(x) >= 0.01 else '%.1f pp' % (100.0 * x)


def main(argv):
    seed, n_main, n_var, n_var2, n_oat, n_block, n_sign, n_hne, quick, mode = parse_args(argv)
    if mode == 'yaml':
        sys.stdout.write(emit_yaml())
        return 0
    if mode == 'sources':
        sys.stdout.write(print_sources_md())
        return 0
    N_SIGN_USED[0] = n_sign
    default_run = (seed == DEFAULT_SEED and not quick and (n_main, n_var, n_var2, n_oat, n_block, n_sign, n_hne) ==
                   (N_MAIN_DEFAULT, N_VAR_DEFAULT, N_VAR2_DEFAULT, N_OAT_DEFAULT, N_BLOCK_DEFAULT, N_SIGN_DEFAULT,
                    N_HNE_DEFAULT))
    here = os.path.dirname(os.path.abspath(__file__))
    warnings = yaml_check(os.path.join(here, 'params.yaml'))
    n_by = {'main': n_main, 'var': n_var, 'var2': n_var2}
    n_hn_sweep = sum(1 for s in PARAMS if s['hn'] and s['name'] not in SIGN_PARAMS)
    n_paths = (6 * (1 + len(S0_BRACKET_CELLS)) + sum(n_by[sc['n']] for sc in SCENARIOS[1:])
               + 2 * len(PARAMS) * n_oat + (len(BLOCK_VARIANTS) + len(S3_VARIANTS)) * n_block
               + 2 * len(SIGN_PARAMS) * n_sign + 2 * n_hn_sweep * n_hne + 9)

    progress('S0 regression cells and paper 5.2 brackets')
    cells = run_s0_cells(seed)
    avg = s0_prior_averaged()
    progress('prior quantiles')
    qtab = prior_quantiles(seed)
    results = []
    proj = None
    paths_done = 6 * (1 + len(S0_BRACKET_CELLS))
    for sc in SCENARIOS[1:]:
        N = n_by[sc['n']]
        progress('%s (N = %d)' % (sc['key'], N))
        snaps = (n_block, n_oat, n_sign, n_hne) if sc['key'] == 'S1' else ((n_block,) if sc['key'] == 'S3' else ())
        results.append(run_scenario(sc, N, seed, snaps=snaps))
        paths_done += N
        if sc['key'] == 'S1':
            ms = 1000.0 * results[-1]['time'] / N
            proj = (time.time() - T_WALL0) + (n_paths - paths_done) * ms / 1000.0
            progress('projected total run time %.0f s (%d engine paths at %.2f ms/path measured on S1); a laptop '
                     '2.5x slower would take about %.0f s; --quick runs every N / 4' % (proj, n_paths, ms, 2.5 * proj))
    R = dict((r['key'], r) for r in results)
    S1 = R['S1']
    progress('S1 sensitivity: OAT (%d parameters x 2 sides x %d draws)' % (len(PARAMS), n_oat))
    t_oat = time.time()
    oat_rows, base50 = run_oat(S1, n_oat, seed, qtab)
    t_oat = time.time() - t_oat
    progress('S1 sensitivity: block-level variants (%d x %d draws)' % (len(BLOCK_VARIANTS), n_block))
    blocks = run_blocks(S1, n_block, seed)
    progress('S3 variants (%d x %d draws)' % (len(S3_VARIANTS), n_block))
    s3v = run_s3_variants(R['S3'], n_block, seed)
    progress('S1 sensitivity: sign check (%d parameters x 2 x %d draws)' % (len(SIGN_PARAMS), n_sign))
    sign_rows = run_sign(S1, n_sign, seed, qtab)
    progress('human-nature effect sweep (%d parameters x 2 x %d draws; %d more from the sign check)'
             % (n_hn_sweep, n_hne, N_HN - n_hn_sweep))
    hne = run_hn_effects(S1, n_hne, seed, qtab, sign_rows, n_sign)
    rec_status, stale = hn_record_check(hne, default_run)
    progress('closed-form checks')
    chk = engine_check_runs(seed)
    terc = tercile_contrast(S1)
    hnm = hn_multipliers(S1, blocks)
    pools, ov_rows = outside_view(S1, hnm)

    # ---- assemble output; the header is written last so it can carry the wall time ----
    head = []
    OUT_BODY = []
    globals()['OUT'] = OUT_BODY
    print_model_summary()
    ok, worst_p, worst_m, reg_line = print_regression(cells)
    print_s0_cells(cells, avg)
    print_main(results)
    cond = print_conditional(S1)
    print_channels(results)
    print_outside_view(pools, ov_rows, S1, blocks, hnm, results)
    print_comparison(cells, avg, results, ov_rows, blocks, s3v, hnm)
    print_premises(S1)
    oat_sorted, bd = print_sensitivity(oat_rows, base50, terc, n_oat, S1, blocks, sign_rows)
    lchk = print_checks(chk, qtab, S1, R['S3'], blocks, s3v)
    print_param_lists(qtab, hne, rec_status)
    dv = print_diagnostics(results, S1, R['S1-containment'])
    print_footer(S1, oat_sorted, bd, blocks, sign_rows, dv, R['S3'], s3v, cond, lchk, R, ov_rows)
    wall = time.time() - T_WALL0

    globals()['OUT'] = head
    emit('## 1. mc.py v%s: AI-caused extinction timing, competing hazards with explicit human-nature parameters'
         % MC_VERSION)
    emit()
    emit('Data for examination, not instructions. Run %s; seed %d; Python %s; N_MAIN = %d (S1, S2, S3), '
         'N_VAR = %d (S1-cal), N_VAR2 = %d (S1-side-high, S1-containment; both with common random numbers with S1), '
         'N_OAT = %d per side x 2 x %d parameters, N_BLOCK = %d per variant x %d (S1) + %d (S3), N_SIGN = %d per '
         'side x 2 x %d, N_HNE = %d per side x 2 x %d human-nature parameters%s; %d engine paths in total; grid: %s; '
         'learning_mode default = %r, learn_count default = %r; projected run time %.0f s, wall time %.1f s (S1 '
         '%.2f ms/path; OAT %.1f s; a laptop 2.5x slower would take about %.0f s).'
         % (AS_OF, seed, sys.version.split()[0], n_main, n_var, n_var2, n_oat, len(PARAMS), n_block,
            len(BLOCK_VARIANTS), len(S3_VARIANTS), n_sign, len(SIGN_PARAMS), n_hne, n_hn_sweep,
            ' (--quick)' if quick else '', n_paths, GRID_DOC, SWITCHES_DEFAULT['learning_mode'],
            SWITCHES_DEFAULT['learn_count'], proj, wall, 1000.0 * S1['time'] / S1['N'], t_oat, 2.5 * wall))
    emit()
    emit('All Monte Carlo quantities are prior-predictive means over the stated priors; no output is '
         'conditioned on data except through the priors and the 2026-10 initial state.')
    emit()
    emit('Outcomes: E1 broad = human extinction or permanent loss of humanity\'s collective '
         'self-determination (paper sec. 1); E2 strict = population below 5,000 by Y (XPT definition), counting E1 '
         'worlds that become strict extinction after the E1 event as capability keeps growing; E2 is a subset of '
         'E1. Channels: accidental escape, deliberate misuse, power concentration or lock-in. Safe absorbing '
         'state: verified alignment (AI nature, gated on the threshold crossing), durable control regime or '
         'deliberate halt (human nature, calibrated to the historical 0.5-1%/yr).')
    emit()
    emit('params.yaml mirror check: %s' % ('all %d parameter and %d constant names found'
                                           % (len(PARAMS), len(CONSTANTS)) if not warnings else
                                           'WARNING: ' + '; '.join(warnings)))
    emit()
    nos = block_by_id(blocks, 'nosafe')
    anc = block_by_id(blocks, 'anchor')
    s1b = S1['snap'][n_block]
    b3 = dict((v['id'], v) for v in s3v)
    acc = channel_curve(S1, 'acc')
    side = channel_curve(S1, 'side')
    se100 = se_of([r['E1'][YP[2100]] for r in S1['recs']])
    hy = (2035, 2050, 2100)
    emit('Headline (S1, historical human nature): P(E1 by 2035 / 2050 / 2100) = %s; of which accidental escape %s '
         'and the side channels (deliberate misuse + lock-in) %s; P(E2 strict by 2035 / 2050 / 2100) = %s; P(safe '
         'exit by 2100) = %s (Monte Carlo SE of P(E1 by 2100) %.1f pp; tables to 0.1 pp in blocks 4 and 4b). The '
         'level is set mainly by e1 and the side-channel priors: draws with e1 >= 1e-2 (%s of the prior) carry %s of '
         'P(E1 by 2100), while draws with e1 < 1e-3 still reach %s by 2100, of which %s (%s of it) come through '
         'the side channels (block 4b); with the former, higher-hazard side-channel priors (S1-side-high) P(E1 by 2100) '
         '= %s.'
         % (' / '.join(pch(at(S1['E1'], Y)) for Y in hy), ' / '.join(pch(acc[YP[Y]]) for Y in hy),
            ' / '.join(pch(side[YP[Y]]) for Y in hy), ' / '.join(pch(at(S1['E2'], Y)) for Y in hy),
            pch(at(S1['safe'], 2100)), 100 * se100, pc(cond['prior_mass_hi']), pc(cond['share100_hi']),
            pch(cond['e1_low_2100']), pph(cond['side_low_2100']),
            pc(cond['side_low_2100'] / cond['e1_low_2100'] if cond['e1_low_2100'] > 0 else float('nan'), 0),
            pch(at(R['S1-side-high']['E1'], 2100))))
    emit()
    cal = R['S1-cal']
    emit('Context: on the first %d S1 draws P(E1 by 2100) = %s with all safe-exit routes, %s with the safe exit '
         'limited to the historical regime anchor (s_regime only), %s with no safe exit. S1-cal (e1 ~ LogU(1e-4, '
         '1e-2)) P(E1 by 2100) = %s, of which accidental escape %s, against the outside-view E1 pool %s (anchors '
         'mostly about loss of control; blocks 6 and 7). S2 (premise-driven) P(E1 by 2035) = %s; S3 P(E1 by 2100) '
         '= %s (%s with s_regime only, first %d draws).'
         % (n_block, pch(at(s1b['E1'], 2100)), pch(at(anc['E1'], 2100)), pch(at(nos['E1'], 2100)),
            pch(at(cal['E1'], 2100)), pch(channel_curve(cal, 'acc')[YP[2100]]), pch(ov_rows[('E1', 'raw')][YP[2100]]),
            pch(at(R['S2']['E1'], 2035)), pch(at(R['S3']['E1'], 2100)), pch(at(b3['S3anchor']['E1'], 2100)), n_block))
    emit()
    emit(reg_line)
    emit()
    if stale:
        emit('WARNING: the recorded human-nature effects in PARAMS and params.yaml are %s (block 11).' % rec_status)
        emit()
    for line in head + OUT_BODY:
        print(line)
    print('Wall time: %.1f s.' % wall)
    print()

    def sc_json(res):
        return {'E1': dict((str(Y), round(at(res['E1'], Y), 5)) for Y in YEARS),
                'E2': dict((str(Y), round(at(res['E2'], Y), 5)) for Y in YEARS),
                'E1_accident': dict((str(Y), round(v, 5)) for Y, v in zip(YEARS, channel_curve(res, 'acc'))),
                'E1_misuse_lockin': dict((str(Y), round(v, 5)) for Y, v in zip(YEARS, channel_curve(res, 'side'))),
                'safe_2100': round(at(res['safe'], 2100), 5),
                'E1_quantiles': [None if t_quantile(res['E1'], q) is None else round(t_quantile(res['E1'], q), 2)
                                 for q in (0.1, 0.5, 0.9)],
                'E2_quantiles': [None if t_quantile(res['E2'], q) is None else round(t_quantile(res['E2'], q), 2)
                                 for q in (0.1, 0.5, 0.9)],
                'N': res['N']}

    def rnd(d):
        return dict((k, round(v, 5) if isinstance(v, float) else v) for k, v in d.items())
    js = {'version': MC_VERSION, 'seed': seed, 'quick': quick, 'wall_s': round(wall, 1),
          'projected_s': round(proj, 1) if proj is not None else None, 'engine_paths': n_paths,
          's0_regression': {'pass': ok, 'worst_dP_pp': round(100 * worst_p, 5), 'worst_dmedian_yr': round(worst_m, 4)},
          's0_prior_averaged': dict((str(Y), round(avg['P'][Y], 4)) for Y in YEARS),
          'scenarios': dict((r['key'], sc_json(r)) for r in results),
          'block_variants': dict((b['id'], {'E1_2050': round(at(b['E1'], 2050), 5),
                                            'E1_2100': round(at(b['E1'], 2100), 5),
                                            'dE1_2050_vs_S1': round(bd[b['id']][2050][0], 5),
                                            'N': b['N']}) for b in blocks),
          's3_variants': dict((v['id'], {'E1_2050': round(at(v['E1'], 2050), 5),
                                         'E1_2100': round(at(v['E1'], 2100), 5), 'N': v['N']}) for v in s3v),
          'S1_conditional_e1': {'prior_mass_e1_ge_1e-2': round(cond['prior_mass_hi'], 4),
                                'share_of_P_E1_2050_from_e1_ge_1e-2': round(cond['share50_hi'], 4),
                                'share_of_P_E1_2100_from_e1_ge_1e-2': round(cond['share100_hi'], 4),
                                'P_E1_2050_given_e1_lt_1e-2': round(cond['p50_lo'], 5),
                                'P_E1_2050_given_e1_ge_1e-2': round(cond['p50_hi'], 5),
                                'points': [{'e1': pt['e1'], 'window': [round(w, 6) for w in pt['window']],
                                            'one_sided': pt['one_sided'], 'draws': pt['draws'],
                                            'E1': rnd(pt['E1']), 'E2': rnd(pt['E2']), 'E1_se': rnd(pt['E1_se']),
                                            'E2_se': rnd(pt['E2_se'])} for pt in cond['points']],
                                'bins': [{'label': b['label'], 'prior_mass': round(b['prior_mass'], 4),
                                          'draws': b['draws'], 'E1': rnd(b['E1']), 'E2': rnd(b['E2']),
                                          'accident_2050': round(b['accident_2050'], 5),
                                          'side_2050': round(b['side_2050'], 5),
                                          'accident_2100': round(b['accident_2100'], 5),
                                          'side_2100': round(b['side_2100'], 5)} for b in cond['bins']]},
          'S1_regime_route_2026_2050': round(dv['S1']['reg_cal'], 5),
          'S1_total_safe_hazard_2026_2050': round(dv['S1']['tot_cal'], 5),
          'safe_anchor': list(SAFE_ANCHOR),
          'S1_P_Tc_le_2050': round(dv['S1']['Tc_by'][1], 4),
          'S1_share_never_cross': round(dv['S1']['Tc_by'][4], 4),
          'human_nature_multipliers': dict((d, dict((k, round(v, 4)) for k, v in hnm[d].items())) for d in hnm),
          'learning_count_check_dE1_2050': {'S1_sampled': round(lchk['S1'][0][0], 5),
                                            'S1_lograte': round(lchk['S1'][1][0], 5),
                                            'S3_sampled': round(lchk['S3'][0][0], 5),
                                            'S3_lograte': round(lchk['S3'][1][0], 5)},
          'sign_check': dict((r['name'], {'dE1_2050': round(r['d50'][0], 5), 'se_2050': round(r['d50'][1], 5),
                                          'dE1_2100': round(r['d100'][0], 5), 'se_2100': round(r['d100'][1], 5),
                                          'dE2_2100': round(r['dE2'][0], 5), 'se_E2_2100': round(r['dE2'][1], 5)})
                             for r in sign_rows),
          'hn_effects_pp': dict((n, {'d2035': round(100 * e['d'][2035][0], 2), 'd2050': round(100 * e['d'][2050][0], 2),
                                     'd2100': round(100 * e['d'][2100][0], 2), 'se2100': round(100 * e['d'][2100][1], 2),
                                     'N': e['N'], 'inert': e['inert']}) for n, e in hne.items()),
          'hn_effect_record': rec_status,
          'hn_effect_record_current': dict((n, tuple(round(100.0 * e['d'][Y][0], 2) for Y in HNE_YEARS))
                                           for n, e in hne.items()),
          'S1_realised_safe_hazard_by_period': [round(x, 5) for x in dv['S1']['s']],
          'outside_view': {'E1_pool_2100': round(pools['E1']['pool'], 5), 'E2_pool_2100': round(pools['E2']['pool'], 5),
                           'E1_raw': dict((str(Y), round(v, 5)) for Y, v in zip(YEARS, ov_rows[('E1', 'raw')])),
                           'E1_x2.3': dict((str(Y), round(v, 5)) for Y, v in zip(YEARS, ov_rows[('E1', 'x2.3')])),
                           'E1_hn_pess': dict((str(Y), round(v, 5)) for Y, v in zip(YEARS, ov_rows[('E1', 'hn_pess')])),
                           'E1_hn_opt': dict((str(Y), round(v, 5)) for Y, v in zip(YEARS, ov_rows[('E1', 'hn_opt')]))},
          'oat_top5': [{'name': r['name'], 'hn': r['hn'], 'swing_pp': round(100 * r['swing'], 2),
                        'se_pp': round(100 * r['se'], 2)} for r in oat_sorted[:5]],
          'yaml_warnings': warnings}
    print('json: ' + json.dumps(js, sort_keys=True))
    sys.stdout.flush()
    progress('done')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv))
