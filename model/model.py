# Reproduces every number in §5 of paper.md / paper.en.md. Parameters mirror model/params.yaml.
# 重算 paper.md / paper.en.md §5 的所有數字；參數與 model/params.yaml 一致。
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

def p_ever_counter(p0):   # r 固定、p 每年減半 / r fixed, p halves every year
    return 1 - math.exp(-r0 * p0 / math.log(2))

for Hc in (167, 2000):
    print(Hc, [round(Tc(Hc, d), 2) for d in (3, 4.3, 7)])
for p in (0.01, 0.05, 0.20):
    for g in (False, True):
        print(p, g, round(2028.5 + median_wait(p, g), 1),
              [round(p_hit_by(Y, 2028.5, p, g), 2) for Y in (2030, 2035, 2040, 2050, 2100)])
    print('counter', p, round(p_ever_counter(p), 3))
