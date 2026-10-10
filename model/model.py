# Reproduces every number in paper.md §5. Parameters mirror model/params.yaml.
# Run: python3 model/model.py
import math
t0, H0, r0 = 2026.5, 17.0, 4.0

def Tc(Hc, d_months):
    return t0 + math.log2(Hc / H0) * d_months / 12

def median_wait(p, grow):
    if not grow:
        return math.log(2) / (r0 * p)
    return math.log2(1 + math.log(2) ** 2 / (r0 * p))

def p_hit_by(Y, Tc_, p, grow):
    t = max(0.0, Y - Tc_)
    lam = r0 * p * t if not grow else r0 * p * (2 ** t - 1) / math.log(2)
    return 1 - math.exp(-lam)

def p_ever_counter(p0):   # r fixed, p halves every year
    return 1 - math.exp(-r0 * p0 / math.log(2))

for Hc in (167, 2000):
    print(Hc, [round(Tc(Hc, d), 2) for d in (3, 4.3, 7)])
for p in (0.01, 0.05, 0.20):
    for g in (False, True):
        print(p, g, round(2028.5 + median_wait(p, g), 1),
              [round(p_hit_by(Y, 2028.5, p, g), 2) for Y in (2030, 2035, 2040, 2050, 2100)])
    print('counter', p, round(p_ever_counter(p), 3))

# ===========================================================================
# Review-driven additions (2026-10-10; review digest §B: F01, F03, F10, F17,
# F18, F21, F23, F39). Everything above this line is the v0.1 script, byte for
# byte, and its eleven output lines are printed first. Everything below is
# appended and changes nothing above. The closed forms are the §4.1 ones,
# generalised in r_0, H_0, H_c, tau and p_min.
# Conventions: d in months, divided by 12 (F39 gap 2); the doubling clock
# starts at T_c, t = Y - T_c, as in p_hit_by() (F07); the §5.2 brackets use
# the unrounded T_c range (F39); the counter-scenario decay is a halving time,
# 2^(-t/tau), not an e-folding time (F18 refuter note).
# ===========================================================================
LN2 = math.log(2)
YEARS = (2027, 2028, 2029, 2030, 2032, 2035, 2040, 2050, 2075, 2100)
D_MONTHS = (3, 4.3, 7)
TC_CENTRAL = 2028.5                     # params.yaml threshold.Tc_central
TC_LO, TC_HI = Tc(167, 3), Tc(2000, 7)  # unrounded 2027.324 / 2030.512


def Tc_h0(Hc, d_months, H0_):
    """T_c with an explicit anchor H0_ in hours; same formula as Tc()."""
    return t0 + math.log2(Hc / H0_) * d_months / 12


def median_wait_r(p, grow, r):
    """§4.1 median waiting time after T_c with an explicit r_0 = r."""
    if not grow:
        return LN2 / (r * p)
    return math.log2(1 + LN2 ** 2 / (r * p))


def p_hit_by_r(Y, Tc_, p, grow, r):
    """§4.1 cumulative hit probability by year Y with an explicit r_0 = r."""
    t = max(0.0, Y - Tc_)
    lam = r * p * t if not grow else r * p * (2 ** t - 1) / LN2
    return 1 - math.exp(-lam)


def lambda_decay(t, p0, r, tau, pmin_frac):
    """Cumulative hazard with r fixed and p(t) = p_min + (p0 - p_min) 2^(-t/tau),
    p_min = pmin_frac * p0 (F18). tau = 1, pmin_frac = 0 is the §5.4 case."""
    pmin = pmin_frac * p0
    return r * (pmin * t + (p0 - pmin) * tau / LN2 * (1 - 2 ** (-t / tau)))


def p_ever_decay(p0, r, tau, pmin_frac):
    """P(ever hit) under lambda_decay: finite only when the floor is zero."""
    if pmin_frac > 0:
        return 1.0      # hazard floor r * p_min > 0: the integral diverges
    return 1 - math.exp(-r * p0 * tau / LN2)


def p_needed(P, r, Y=2100, Tc_=TC_CENTRAL):
    """§7 inversion, fixed rate: the p for which P(hit <= Y) = P."""
    return -math.log(1 - P) / (r * (Y - Tc_))


def md_table(header, rows):
    print('| ' + ' | '.join(header) + ' |')
    print('|' + '|'.join(' --- ' for _ in header) + '|')
    for row in rows:
        print('| ' + ' | '.join(str(c) for c in row) + ' |')


def pct(x, nd=1):
    return '{:.{}f}%'.format(100 * x, nd)


def pshort(p):
    return '{:g}%'.format(100 * p)


# --- Block 1: T_c over the H_0 CI, the 80% anchor and an H_c = 40 h row ----
# F10: H_0 = 17 h sits at METR's reliability ceiling; its 95% CI (8.5-55 h)
#      was never propagated into T_c.
# F21: the 80% horizon (3.1 h) is the reliability standard relevant to
#      surviving coordinated defense; same d assumed.
# F23: METR's 1-week rogue-replication tasks (~40-80 h) are the only
#      task-length anchor in the literature; 2,000 h has no external anchor.
# A T_c below t0 = 2026.5 means that anchor already exceeds that threshold.
print()
print('## (1) T_c = t0 + (d/12) log2(Hc/H0) over H0 in {3.1 (80% anchor), '
      '8.5, 17, 55} h and Hc in {40, 80, 167, 2000} h  [F10, F21, F23]')
H0_ROWS = ((3.1, '80% horizon anchor'), (8.5, '50% CI low'),
           (17.0, '50% point (v0.1)'), (55.0, '50% CI high'))
HC_ROWS = ((40, '1 working week (METR 1-week tasks)'),
           (80, 'upper end of METR 1-week tasks'),
           (167, '1 working month (v0.1)'), (2000, '1 working year (v0.1)'))
rows = []
for Hc, hc_label in HC_ROWS:
    for H0_, h0_label in H0_ROWS:
        rows.append([Hc, hc_label, H0_, h0_label,
                     '{:.2f}'.format(math.log2(Hc / H0_))]
                    + ['{:.2f}'.format(Tc_h0(Hc, d, H0_)) for d in D_MONTHS])
md_table(['Hc (h)', 'Hc anchor', 'H0 (h)', 'H0 anchor', 'doublings',
          'd = 3 mo', 'd = 4.3 mo', 'd = 7 mo'], rows)
print()
print('Note: negative doublings give T_c < t0 = {:g}, a back-extrapolation: '
      'under that anchor the threshold was already crossed at t0, and a '
      'longer d dates the crossing further back. The 80% anchor assumes the '
      'same d as the 50% horizon (F21).'.format(t0))

# --- Block 2: §5.2 medians with the brackets computed explicitly -----------
# F39 gap 3: the twelve bracket endpoints were never printed; they are built
#      from the unrounded T_c range 2027.324-2030.512.
# F17: Tc_central = 2028.5 is the mean of the six §5.1 cells (2028.52), not
#      the range midpoint (2028.92); the last column uses the midpoint.
print()
print('## (2) §5.2 median hit with brackets: central T_c = {}, brackets at '
      'T_c = {:.3f} and {:.3f}  [F17, F39]'.format(TC_CENTRAL, TC_LO, TC_HI))
cells = sorted(Tc(Hc, d) for Hc in (167, 2000) for d in D_MONTHS)
TC_MID = (TC_LO + TC_HI) / 2
print('T_c range midpoint = {:.2f}; mean of the six §5.1 cells = {:.2f}; '
      'median of the six = {:.2f}'.format(TC_MID, sum(cells) / 6,
                                          (cells[2] + cells[3]) / 2))
rows = []
for p in (0.01, 0.05, 0.20):
    for grow, label in ((False, 'fixed'), (True, 'doubling')):
        W = median_wait(p, grow)
        rows.append([pshort(p), label, '{:.2f}'.format(W),
                     '{:.1f}'.format(TC_CENTRAL + W),
                     '{:.1f}'.format(TC_LO + W), '{:.1f}'.format(TC_HI + W),
                     '{:.1f}'.format(TC_MID + W)])
md_table(['p', 'rate', 'W_med (yr)', 'median @ {}'.format(TC_CENTRAL),
          'bracket low', 'bracket high',
          'median @ midpoint {:.2f}'.format(TC_MID)], rows)

# --- Block 3: p and r_0 grids --------------------------------------------
# F03: the 1% floor is 1-3 orders above every analogous base rate and every
#      calibrated estimate; p = 0.01% (calibrated-forecaster-implied) and
#      0.1% (intermediate) are added; 1-20% is the upper tail.
# F01: r_0 = 4/yr counts one correlated cluster (n_eff ~ 1); r_0 in {1, 4, 10}
#      spans one root cause per year to the loose incident count.
# Under yearly doubling, cells past 2035 are model artifacts (§5.3 note; F07).
for grow, tag, label in ((False, 'a', 'fixed incident rate'),
                         (True, 'b', 'incident rate doubling yearly from T_c')):
    print()
    print('## (3{}) {}: median and P(hit <= Y), T_c = {}, p in {{0.01%, 0.1%, '
          '1%, 5%, 20%}}, r0 in {{1, 4, 10}}/yr  [F01, F03]'
          .format(tag, label, TC_CENTRAL))
    rows = []
    for r in (1, 4, 10):
        for p in (0.0001, 0.001, 0.01, 0.05, 0.20):
            rows.append([r, pshort(p),
                         '{:.1f}'.format(TC_CENTRAL + median_wait_r(p, grow, r))]
                        + [pct(p_hit_by_r(Y, TC_CENTRAL, p, grow, r))
                           for Y in YEARS])
    md_table(['r0/yr', 'p', 'median'] + ['<= {}'.format(Y) for Y in YEARS],
             rows)
    print()
    if not grow:
        print('Note: T_c = {} so the 2027 and 2028 columns are 0 by '
              'construction; 100.0% means >= 99.95%. Medians after 2100 '
              'extrapolate a constant hazard indefinitely; read them as '
              '"after 2100".'.format(TC_CENTRAL))
    else:
        print('Note: model artifact past ~2035 (§5.3 note; F07). The clock '
              'is r(Y) = r0 2^(Y - T_c), so at r0 = 4 the rate is {:.0f}/yr '
              'in 2035 and {:,.0f}/yr in 2040.'.format(
                  4 * 2 ** (2035 - TC_CENTRAL), 4 * 2 ** (2040 - TC_CENTRAL)))

# --- Block 4: counter-scenario generalised to a halving time and a floor --
# F18: documented learning curves halve in 2.2 yr (Mandiant median dwell
#      time 2011-2023), 7.6 yr (aviation fatal-accident rate, Boeing) and
#      10.3 yr (INL PWR transients 1988-2016), and decay to a non-zero floor
#      (Duffey-Saull universal learning curve; US nuclear plateau since ~2005).
#      p(t) = p_min + (p0 - p_min) 2^(-t/tau), r = r0 fixed, p_min = frac * p0.
#      tau = 1, frac = 0 reproduces §5.4. A floor > 0 makes P(ever) = 100%
#      on the timescale 1/(r0 p_min); what stays finite is P(hit <= 2100).
TAU_SRC = {1: 'v0.1 §5.4 (no analogue)', 2.2: 'Mandiant dwell time',
           7.6: 'aviation fatal-accident rate', 10.3: 'INL PWR transients'}
print()
print('## (4a) Counter-scenario generalised: tau (halving time, yr) x p_min '
      'fraction, r0 = {:g}, T_c = {}; cells are P(ever) / P(<= 2050) / '
      'P(<= 2100)  [F18]'.format(r0, TC_CENTRAL))
rows = []
for tau in (1, 2.2, 7.6, 10.3):
    for frac in (0, 0.01, 0.1):
        row = [tau, TAU_SRC[tau], frac]
        for p0 in (0.01, 0.05, 0.20):
            ever = p_ever_decay(p0, r0, tau, frac)
            by50 = 1 - math.exp(-lambda_decay(2050 - TC_CENTRAL, p0, r0, tau, frac))
            by100 = 1 - math.exp(-lambda_decay(2100 - TC_CENTRAL, p0, r0, tau, frac))
            row.append('{} / {} / {}'.format(pct(ever), pct(by50), pct(by100)))
        rows.append(row)
md_table(['tau (yr)', 'tau source', 'p_min/p0', 'p0 = 1%', 'p0 = 5%',
          'p0 = 20%'], rows)
print()
print('Note: tau = 1, p_min = 0 is v0.1 §5.4. With p_min > 0 the hazard never '
      'falls below r0 p_min, so P(ever) = 100% on the timescale 1/(r0 p_min) '
      '(e.g. {:,.0f} yr at p0 = 1%, p_min/p0 = 0.01); the finite comparison is '
      'P(<= Y). Decay is a halving time, 2^(-t/tau).'.format(
          1 / (r0 * 0.01 * 0.01)))
print()
print('## (4b) Same, zero floor, crossed with r0 in {1, 4, 10}/yr: '
      'P(ever) = 1 - exp(-r0 p0 tau / ln2)  [F01, F18]')
rows = []
for tau in (1, 2.2, 7.6, 10.3):
    for r in (1, 4, 10):
        rows.append([tau, r] + [pct(p_ever_decay(p0, r, tau, 0))
                                for p0 in (0.01, 0.05, 0.20)])
md_table(['tau (yr)', 'r0/yr', 'p0 = 1%', 'p0 = 5%', 'p0 = 20%'], rows)

# --- Block 5: §7 inversion -------------------------------------------------
# F39: p = -ln(1 - P) / (r0 (2100 - T_c)), fixed rate, T_c = 2028.5 (71.5 yr).
# Targets (digest §A, risk-estimates): 0.38% XPT superforecasters and 3% XPT
# AI domain experts (E2 strict: population < 5,000, by 2100); 5% AI Impacts
# 2023 survey median, 'extremely bad outcomes' (Grace et al. 2024); 10% AI
# Impacts Dec-2024 wave median (secondary coverage), also the 2023 survey's
# loss-of-control item ('extinction or permanent disempowerment', the E1-like
# phrasing); 23% Metaculus Pros (definition not restated in the review).
# 'oom' = orders of magnitude below the v0.1 floor p = 1%. F03: every target
# sits 0.4-3.3 orders below the grid floor.
print()
print('## (5) §7 inversion: per-incident p needed for P(hit <= 2100) = target, '
      'fixed rate, T_c = {}  [F03, F39]'.format(TC_CENTRAL))
TARGETS = ((0.0038, 'XPT superforecasters (E2 strict)'),
           (0.03, 'XPT AI domain experts (E2 strict)'),
           (0.05, 'AI Impacts 2023 survey median, extremely bad outcomes'),
           (0.10, 'AI Impacts Dec-2024 wave median / 2023 survey '
                  'loss-of-control item (E1-like)'),
           (0.23, 'Metaculus Pros'))
rows = []
for P, label in TARGETS:
    row = [pct(P, 2), label]
    for r in (1, 4, 10):
        p = p_needed(P, r)
        row.append('{:.3g}% ({:.1f} oom)'.format(100 * p, math.log10(0.01 / p)))
    rows.append(row)
md_table(['target by 2100', 'source', 'r0 = 1', 'r0 = 4', 'r0 = 10'], rows)

# --- Block 6: §5.5 sensitivities as explicit differences -------------------
# F39 gap 3: the four §5.5 differences were never printed (rows 1-4).
# Rows 5-9 add the sensitivities the review found missing: the H_0 CI (F10),
# the 80% anchor (F21), r_0 (F01, F17), the combined d x H_c span and the
# full T_c span once the H_0 CI is included (F10, F17).
print()
print('## (6) §5.5 sensitivities as explicit differences (years)  '
      '[F01, F10, F17, F21, F39]')


def W(p, grow, r=r0):
    return median_wait_r(p, grow, r)


def fmt_pair(a, b):
    return '{:.2f} -> {:.2f}'.format(a, b)


def direction(a, b):
    return 'later' if b > a else 'earlier'


tc_lo_ci = min(Tc_h0(Hc, d, 55.0) for Hc in (167, 2000) for d in D_MONTHS)
tc_hi_ci = max(Tc_h0(Hc, d, 8.5) for Hc in (167, 2000) for d in D_MONTHS)
rows = [
    ['p 1% -> 20%, fixed rate (median)',
     fmt_pair(TC_CENTRAL + W(0.01, False), TC_CENTRAL + W(0.20, False)),
     '{:.2f}'.format(W(0.01, False) - W(0.20, False)),
     direction(W(0.01, False), W(0.20, False))],
    ['rate fixed -> doubling, p = 1% (median)',
     fmt_pair(TC_CENTRAL + W(0.01, False), TC_CENTRAL + W(0.01, True)),
     '{:.2f}'.format(W(0.01, False) - W(0.01, True)),
     direction(W(0.01, False), W(0.01, True))],
    ['d 3 -> 7 months (T_c), Hc = 167 / 2000 h',
     '{} / {}'.format(fmt_pair(Tc(167, 3), Tc(167, 7)),
                      fmt_pair(Tc(2000, 3), Tc(2000, 7))),
     '{:.2f} / {:.2f}'.format(Tc(167, 7) - Tc(167, 3),
                              Tc(2000, 7) - Tc(2000, 3)),
     direction(Tc(167, 3), Tc(167, 7))],
    ['Hc 167 -> 2000 h (T_c), d = 3 / 7 months',
     '{} / {}'.format(fmt_pair(Tc(167, 3), Tc(2000, 3)),
                      fmt_pair(Tc(167, 7), Tc(2000, 7))),
     '{:.2f} / {:.2f}'.format(Tc(2000, 3) - Tc(167, 3),
                              Tc(2000, 7) - Tc(167, 7)),
     direction(Tc(167, 3), Tc(2000, 3))],
    ['H0 8.5 -> 55 h (T_c, any Hc), d = 3 / 4.3 / 7 months [F10]',
     'log2(55/8.5) = {:.2f} doublings'.format(math.log2(55 / 8.5)),
     ' / '.join('{:.2f}'.format(Tc_h0(167, d, 8.5) - Tc_h0(167, d, 55.0))
                for d in D_MONTHS),
     direction(Tc_h0(167, 3, 8.5), Tc_h0(167, 3, 55.0))],
    ['H0 17 -> 3.1 h, 50% -> 80% anchor (T_c, any Hc), d = 3 / 4.3 / 7 [F21]',
     'log2(17/3.1) = {:.2f} doublings'.format(math.log2(17 / 3.1)),
     ' / '.join('{:.2f}'.format(Tc_h0(167, d, 3.1) - Tc_h0(167, d, 17.0))
                for d in D_MONTHS),
     direction(Tc_h0(167, 3, 17.0), Tc_h0(167, 3, 3.1))],
    ['r0 1 -> 10/yr, p = 1%, fixed / doubling (median) [F01, F17]',
     '{} / {}'.format(fmt_pair(TC_CENTRAL + W(0.01, False, 1),
                               TC_CENTRAL + W(0.01, False, 10)),
                      fmt_pair(TC_CENTRAL + W(0.01, True, 1),
                               TC_CENTRAL + W(0.01, True, 10))),
     '{:.2f} / {:.2f}'.format(W(0.01, False, 1) - W(0.01, False, 10),
                              W(0.01, True, 1) - W(0.01, True, 10)),
     direction(W(0.01, False, 1), W(0.01, False, 10))],
    ['d x Hc combined span of §5.1 (T_c) [F17]',
     fmt_pair(TC_LO, TC_HI), '{:.2f}'.format(TC_HI - TC_LO), 'span'],
    ['full T_c span with the H0 CI, Hc in {167, 2000} h [F10]',
     fmt_pair(tc_lo_ci, tc_hi_ci), '{:.2f}'.format(tc_hi_ci - tc_lo_ci),
     'span'],
]
md_table(['sensitivity', 'from -> to', 'difference (yr, magnitude)',
          'moves'], rows)

# --- Index of the review finding ids used in the headings above -------------
# F01-F40 refer to the 2026-10-10 review of paper v0.1. Their titles, whether
# each survived adversarial verification, and the public sources are printed by
# python3 model/mc.py --sources and mirrored in params.yaml (review_findings:,
# sources:), so an outside reader can resolve every id printed by this script.
print()
print('Review finding ids in the headings above (F01-F40) refer to the 2026-10-10 review of paper v0.1. '
      'Titles, survival status and public sources: python3 model/mc.py --sources, or params.yaml keys '
      'review_findings: and sources:.')
