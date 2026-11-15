#!/usr/bin/env python3
"""
Discard Hold'em bot -- "ATLAS" (Pokerbots 2026).

Design notes (what this bot implements and why)
================================================
Baseline: a GTO-flavoured core (pot-odds/MDF discipline, mixed strategies,
polarized sizing, capped bluff ratios) with a bounded exploitation layer
(opponent statistics + restricted-Nash-style blending: deviations are
shrunk toward the robust baseline by observation counts).

  1. Fast evaluation.  A numpy-vectorised 7-card evaluator (LUT-driven,
     ~1M+ evals/s) powers exact river equity, exact (or Monte-Carlo) turn
     equity vs. a weighted range, and flop/preflop Monte-Carlo equity.
  2. Preflop.  An embedded equity/percentile table over all 1755 canonical
     3-card classes (suit-symmetric, Monte-Carlo'd offline vs random
     opponents with perfect-information discards) drives a position-aware
     chart with mixed frequencies and opponent-adaptive thresholds.
  3. Ranges.  Bayesian-style weighting over the opponent's holdings:
     preflop line -> percentile prior; flop actions -> strength bump;
     the EXPOSED discard -> per-combo discard likelihood (softmax over
     keep-scores, sharpness learned from their observed discard
     predictability); turn/river actions -> exact made-hand likelihoods.
  4. Decisions.  Every legal action (fold/call/check/raise-to-x over a
     size grid) is scored with an EV model: fold probability from
     observed fold-rates blended with MDF priors (Laplace shrinkage),
     call -> showdown term, re-raise penalty branch.  Borderline spots
     are mixed with hand-seeded randomness (duplicate-leg consistency).
  5. Discard.  Each keep-option is Monte-Carlo'd against the opponent's
     triple range (they keep their best card pair per sampled runout);
     near-ties are randomised so our exposed discard leaks less.
  6. Safety.  Strict stdout protocol purity, stderr capped, every reply
     validated against legal_actions / min_raise / max_raise, deadline
     checks inside all compute loops, cheap fallback replies, exit on
     match_end.  Budget: ~30-40 ms per decision target, adaptive.

Only the standard library and numpy are used.
"""

import base64
import io
import json
import math
import os
import random
import sys
import time
import traceback

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import numpy as np

# stdout must carry protocol only: swap in at import time, before anything
# else can print.
_PROTO_OUT = sys.__stdout__ if sys.__stdout__ is not None else sys.stdout
sys.stdout = sys.stderr

RANKS = "23456789TJQKA"
SUITS = "shdc"
CARD_TEXT = [r + s for r in RANKS for s in SUITS]
CARD_ID = {c: i for i, c in enumerate(CARD_TEXT)}
ALL_CARDS = tuple(range(52))

_LOG_BYTES = 0
_LOG_MAX = 24000


def log_dbg(text):
    global _LOG_BYTES
    if _LOG_BYTES >= _LOG_MAX:
        return
    try:
        _PROTO = sys.stderr
        b = len(text) + 1
        if _LOG_BYTES + b > _LOG_MAX:
            return
        _LOG_BYTES += b
        print(text, file=sys.stderr)
    except Exception:
        pass


# ------------------------------------------------------------------ LUTs
def _build_luts():
    n = 1 << 13
    straight_high = np.full(n, -1, np.int64)
    top1 = np.full(n, -1, np.int64)
    top2 = np.zeros(n, np.int64)
    top3 = np.zeros(n, np.int64)
    top5 = np.zeros(n, np.int64)
    for m in range(n):
        bits = [i for i in range(13) if m & (1 << i)]
        if not bits:
            continue
        vals = [i + 2 for i in bits]
        vals.reverse()
        top1[m] = vals[0]
        if len(vals) >= 2:
            top2[m] = (vals[0] << 4) | vals[1]
        if len(vals) >= 3:
            top3[m] = (vals[0] << 8) | (vals[1] << 4) | vals[2]
        if len(vals) >= 5:
            top5[m] = (vals[0] << 16) | (vals[1] << 12) | (vals[2] << 8) | (vals[3] << 4) | vals[4]
        best = -1
        for top in range(12, 3, -1):
            win = 31 << (top - 4)
            if m & win == win:
                best = top + 2
                break
        if best < 0:
            wheel = (1 << 12) | 1 | 2 | 4 | 8
            if m & wheel == wheel:
                best = 5
        straight_high[m] = best
    return straight_high, top1, top2, top3, top5


STRAIGHT_HIGH, TOP1, TOP2, TOP3, TOP5 = _build_luts()

# straight-draw count table for keep heuristics
_SDRAW = [0.0] * 8192
for _m in range(8192):
    _n = 0
    for _i in range(9):
        _w = 31 << _i
        if (_m & _w).bit_count() == 4:
            _n += 1
    if (_m & ((1 << 12) | 1 | 2 | 4 | 8)).bit_count() == 4:
        _n += 1
    _SDRAW[_m] = 0.30 if _n >= 2 else (0.18 if _n == 1 else 0.0)


# ------------------------------------------------------------ batch eval
def eval7_batch(cards):
    """cards: (N,7) integer card ids. Returns (N,) int64 scores (higher=better)."""
    cards = np.asarray(cards, dtype=np.int64)
    n = cards.shape[0]
    nc = cards.shape[1]
    rows = np.arange(n)
    r = cards >> 2
    s = cards & 3
    cnt = np.zeros((n, 13), dtype=np.int8)
    for j in range(nc):
        cnt[rows, r[:, j]] += 1
    bits = (1 << np.arange(13, dtype=np.int64))
    cntb = cnt > 0
    pm_ = (cnt == 2)
    tm_ = (cnt == 3)
    qm_ = (cnt == 4)
    rmask = (cntb * bits).sum(axis=1)
    pmask = (pm_ * bits).sum(axis=1)
    tmask = (tm_ * bits).sum(axis=1)
    qmask = (qm_ * bits).sum(axis=1)
    scnt = np.zeros((n, 4), dtype=np.int8)
    sm = np.zeros((4, n), dtype=np.int64)
    for j in range(nc):
        scnt[rows, s[:, j]] += 1
        sm[s[:, j], rows] |= bits[r[:, j]]
    has_flush = (scnt >= 5).any(axis=1)
    fs = np.where(has_flush, (scnt >= 5).argmax(axis=1), 0)
    fmask = np.where(has_flush, sm[fs, rows], 0)

    sh = STRAIGHT_HIGH[fmask]
    sf = sh >= 0
    sh_n = np.where(sf, sh, 0)

    q = qmask > 0
    qv = np.where(q, TOP1[qmask], 0)
    qk = np.where(q, TOP1[rmask ^ qmask], 0)

    t = tmask > 0
    t_hi = np.where(t, TOP1[tmask], 0)
    hi_trip_bit = 1 << np.maximum(t_hi - 2, 0)
    low_trip_bit = tmask ^ hi_trip_bit
    fh_pair_mask = low_trip_bit | pmask
    fh = t & (fh_pair_mask > 0)
    fh_pair = np.where(fh, TOP1[fh_pair_mask], 0)

    st = STRAIGHT_HIGH[rmask]
    straight = st >= 0

    p2 = pmask & (pmask - 1)
    two_pair = (~t & ~q & (p2 > 0))
    ph = np.where(two_pair, TOP1[pmask], 0)
    pl = np.where(two_pair, TOP2[pmask] & 15, 0)
    chosen = (1 << np.maximum(ph - 2, 0)) | (1 << np.maximum(pl - 2, 0))
    kmask_2p = (rmask ^ pmask) | (pmask ^ chosen)
    pk = np.where(two_pair, TOP1[kmask_2p], 0)

    one_pair = (~t & ~q & (pmask > 0) & (p2 == 0))
    pv = np.where(one_pair, TOP1[pmask], 0)
    kp3 = np.where(one_pair, TOP3[rmask ^ pmask], 0)

    flush_score = (5 << 24) | (TOP5[fmask] << 4)

    out = np.zeros(n, dtype=np.int64)
    out = np.where(sf, (8 << 24) | (sh_n << 20), out)
    out = np.where(~sf & q, (7 << 24) | (qv << 20) | (qk << 16), out)
    out = np.where(~sf & ~q & fh, (6 << 24) | (t_hi << 20) | (fh_pair << 16), out)
    out = np.where(~sf & ~q & ~fh & has_flush, flush_score, out)
    out = np.where(~sf & ~q & ~fh & ~has_flush & straight, (4 << 24) | (st << 20), out)
    out = np.where(~sf & ~q & ~fh & ~has_flush & ~straight & t,
                   (3 << 24) | (t_hi << 20) | (TOP2[rmask ^ tmask] << 12), out)
    out = np.where(~sf & ~q & ~fh & ~has_flush & ~straight & ~t & two_pair,
                   (2 << 24) | (ph << 20) | (pl << 16) | (pk << 12), out)
    out = np.where(~sf & ~q & ~fh & ~has_flush & ~straight & ~t & ~two_pair & one_pair,
                   (1 << 24) | (pv << 20) | (kp3 << 8), out)
    out = np.where(~sf & ~q & ~fh & ~has_flush & ~straight & ~t & ~two_pair & ~one_pair,
                   TOP5[rmask] << 4, out)
    return out


def eval7_scalar(cards):
    cards = [int(c) for c in cards]
    mask = 0; pairs = 0; trips = 0; quads = 0
    s0 = s1 = s2 = s3 = 0
    for c in cards:
        b = 1 << (c >> 2)
        if mask & b:
            if pairs & b:
                if trips & b:
                    quads |= b
                else:
                    trips |= b
            else:
                pairs |= b
        mask |= b
        su = c & 3
        if su == 0: s0 |= b
        elif su == 1: s1 |= b
        elif su == 2: s2 |= b
        else: s3 |= b
    fm = 0
    if s0.bit_count() >= 5: fm = s0
    elif s1.bit_count() >= 5: fm = s1
    elif s2.bit_count() >= 5: fm = s2
    elif s3.bit_count() >= 5: fm = s3
    if fm:
        sh = STRAIGHT_HIGH[fm]
        if sh >= 0:
            return (8 << 24) | (int(sh) << 20)
        return (5 << 24) | int(TOP5[fm] << 4)
    if quads:
        r0 = quads.bit_length() - 1
        km = mask & ~(1 << r0)
        return (7 << 24) | ((r0 + 2) << 20) | ((km.bit_length() + 1) << 16)
    t3 = trips & ~quads
    if t3:
        r0 = t3.bit_length() - 1
        rest_t = t3 & ((1 << r0) - 1)
        if rest_t:
            return (6 << 24) | ((r0 + 2) << 20) | ((rest_t.bit_length() + 1) << 16)
        p2m = pairs & ~trips
        if p2m:
            return (6 << 24) | ((r0 + 2) << 20) | ((p2m.bit_length() + 1) << 16)
    st = STRAIGHT_HIGH[mask]
    if st >= 0:
        return (4 << 24) | (int(st) << 20)
    if t3:
        r0 = t3.bit_length() - 1
        km = mask & ~(1 << r0)
        b = km.bit_length() - 1
        k0 = b; km ^= 1 << b
        k1 = km.bit_length() - 1
        return (3 << 24) | ((r0 + 2) << 20) | ((k0 + 2) << 16) | ((k1 + 2) << 12)
    if pairs:
        r0 = pairs.bit_length() - 1
        rest = pairs & ((1 << r0) - 1)
        if rest:
            r1 = rest.bit_length() - 1
            km = mask & ~(1 << r0) & ~(1 << r1)
            return (2 << 24) | ((r0 + 2) << 20) | ((r1 + 2) << 16) | ((km.bit_length() + 1) << 12)
        km = mask & ~(1 << r0)
        b = km.bit_length() - 1
        k0 = b; km ^= 1 << b
        b = km.bit_length() - 1
        k1 = b; km ^= 1 << b
        k2 = km.bit_length() - 1
        return (1 << 24) | ((r0 + 2) << 20) | ((k0 + 2) << 16) | ((k1 + 2) << 12) | ((k2 + 2) << 8)
    return int(TOP5[mask] << 4)


# score -> rough percentile (for strength likelihoods)
_CAT_PCT = (0.16, 0.45, 0.70, 0.82, 0.875, 0.905, 0.950, 0.988, 0.997)


def hand_pct(score):
    base = _CAT_PCT[score >> 24]
    return base + (score & 0xFFFFFF) * 2.4e-9


# ----------------------------------------------------- keep-score (cheap)
def keep_score(k1, k2, f1, f2, f3):
    k1 = int(k1); k2 = int(k2); f1 = int(f1); f2 = int(f2); f3 = int(f3)
    r1 = k1 >> 2; r2 = k2 >> 2
    fr1 = f1 >> 2; fr2 = f2 >> 2; fr3 = f3 >> 2
    hp = 1 if r1 == r2 else 0
    m1 = (r1 == fr1) + (r1 == fr2) + (r1 == fr3)
    m2 = (r2 == fr1) + (r2 == fr2) + (r2 == fr3)
    c1 = 1 + hp + m1; c2 = 1 + hp + m2
    maxc = c1 if c1 > c2 else c2
    btrips = (fr1 == fr2) and (fr2 == fr3)
    if btrips:
        if maxc < 3: maxc = 3
    elif (fr1 == fr2) or (fr2 == fr3) or (fr1 == fr3):
        if maxc < 2: maxc = 2
    mask = (1 << r1) | (1 << r2) | (1 << fr1) | (1 << fr2) | (1 << fr3)
    nd = mask.bit_count()
    if maxc >= 4: made = 6.0
    elif maxc == 3:
        if nd == 2: made = 5.5
        elif c1 >= 3: made = 3.0 + 0.05 * r1
        else: made = 3.0 + 0.05 * r2
    elif maxc == 2:
        if nd == 3: made = 2.0
        else:
            if hp or m1: pr = r1
            elif m2: pr = r2
            elif fr1 == fr2: pr = fr1
            else: pr = fr2
            made = 1.0 + 0.05 * pr
    else:
        made = 0.06 * (r1 if r1 > r2 else r2)
    sb = [0, 0, 0, 0]
    sb[k1 & 3] |= 1 << r1; sb[k2 & 3] |= 1 << r2
    sb[f1 & 3] |= 1 << fr1; sb[f2 & 3] |= 1 << fr2; sb[f3 & 3] |= 1 << fr3
    fc = max(sb[0].bit_count(), sb[1].bit_count(), sb[2].bit_count(), sb[3].bit_count())
    if fc >= 5:
        if made < 5.0: made = 5.0
    elif fc == 4:
        made += 0.55
    st = STRAIGHT_HIGH[mask]
    if st >= 0:
        if made < 4.0: made = 4.0
    else:
        made += _SDRAW[mask]
    return made + (r1 + r2) * 0.004


def discard_prediction(hole3, flop3):
    """Index of the card a keep_score player would discard."""
    a, b, c = hole3
    f1, f2, f3 = flop3
    s_a = keep_score(b, c, f1, f2, f3)
    s_b = keep_score(a, c, f1, f2, f3)
    s_c = keep_score(a, b, f1, f2, f3)
    if s_a >= s_b and s_a >= s_c:
        return 0
    if s_b >= s_c:
        return 1
    return 2


def keep_softmax_scores(hole3, flop3, sharpness):
    """Softmax probabilities of discarding each of the 3 cards (model of opp)."""
    a, b, c = hole3
    f1, f2, f3 = flop3
    sc = [keep_score(b, c, f1, f2, f3),
          keep_score(a, c, f1, f2, f3),
          keep_score(a, b, f1, f2, f3)]
    mx = max(sc)
    if sharpness <= 0.01:
        return [1 / 3.0, 1 / 3.0, 1 / 3.0]
    ex = [math.exp(sharpness * (v - mx)) for v in sc]
    tot = sum(ex)
    return [e / tot for e in ex]


# ------------------------------------------------------------ preflop table
PREFLOP_B64 = (
    "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"
    "PpwyXxTTHpONZBTTS4aXq+nVvlN08NtzmuApenS4gQr3PqbR7j4Sy+6b4J9bVDB4l27L8qHi+hg9qv5A8mbDVBHzjTavm46cy7po"
    "XymJ64MhOOATgye1puSnWkcB66/Q8eZfdEKjlTpT1pB2YAXlHHtO22fzoND+AU14Es12uUqr17XR3tOfacn8Njo37zl9+vOD9ncL"
    "oHJAAiMFcqi6+I20TL5SKPhgOsYLfX9hTDAgrLhFMCNGDIfG1aDRJI3m5CmwW2aIV9Za+FtpiMlG6lDtjUf74WQsWLqd1h5/TTw8"
    "PHAYF0Kd4mTUF3fStsAh6gnlhvZvPmzY94wm+04CFxcXjGunoc1BEyuOKMFjeh+dM5HCkeVCsJBRwcNxfYh6T0bStDfkKCADN2El"
    "eG7nheV2bhgX6WCIi4PscC20XbHFSPN0NPMYQfm7FT7FueDRZTv0jOhCJW8W6tUc8PeMI0wHCSarkrGpKRU8H5ImyiMF8TeSMOCc"
    "hu59SfDWTIOkeRKiMlNRJJEEzZep4Oinwq0gBb+rUpGeno7c6ykQ3J6Og0KpeO6RDpl7YRjktaY/ZgOkFV/NlvY2MvkaIve9LtQi"
    "TGRHl6n0hzx+9Lxkio3ixFamUNzaFTR6yJIW8C0mra9F9JYdJo3cg5RisormjkzGpqoJLtX6ytqTHShkz0Z6M5JMOuYZ5CseTjW2"
    "O6hHaT1dz26nqEP/+WU9ZVffo6JFLylAY4Rect7QqSPaGPhthi6NTWyVri1Z/1hLBlkR9NM4ktJLbWhTSSvpSOyg95ObqIj/EE3U"
    "M/msbqRpHU9pbHsduQZXk1P1DUq+c4b8NvIg2/oOaRUN0NhPC2ysXoxXmtKUZGZHx+2DKchLior7nSlm+zTym+dA3wSLKbA4j+KC"
    "VIn3bQI5biigcputZL2piWrMuihS+CD1N1ym8t6bdKZ1Cjrqx8lptI22rRfBgTIxfMmdiR6+3/Rhuh1ufgqHr/hrJhxaTtwdVXTZ"
    "uJJMT28hl7QYUtzYQGvGeumU/FWa8eEUXTh/nnoSW+nz5Xay8btPw7KdE5d/jcT+4UF83yTYWnHjXyNxzPmHH31dEtj/XRmlNTNh"
    "WScG3wXTEfBbHO8aeRHROBs5e6RQ4x6G3/0xaImdR8tkGshLvpY+tN0jlx8dpJ+6l7iEblH3l1qavvINfY/tmGDeQ8Q/9z3hMw/2"
    "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"
)

_PF_KEYS = None
_PF_EQ = None
_PF_PCT = None


def _load_preflop():
    global _PF_KEYS, _PF_EQ, _PF_PCT
    try:
        raw = base64.b64decode(PREFLOP_B64)
        buf = io.BytesIO(raw)
        with np.load(buf) as npz:
            _PF_KEYS = np.sort(npz["keys"].astype(np.int64))
            order = np.argsort(npz["keys"].astype(np.int64))
            _PF_EQ = npz["equities"].astype(np.float64)[order]
            _PF_PCT = npz["percentiles"].astype(np.float64)[order]
    except Exception as e:
        log_dbg(f"preflop table load failed: {e}")
        _PF_KEYS = np.zeros(0, np.int64)
        _PF_EQ = np.full(0, 0.5)
        _PF_PCT = np.full(0, 0.5)


_load_preflop()


def _triple_key_scalar(cards):
    cs = sorted((int(c) for c in cards), key=lambda c: -c)
    r = [c >> 2 for c in cs]
    s = [c & 3 for c in cs]
    rv = [x + 2 for x in r]
    if rv[0] == rv[1] == rv[2]:
        pat = 0
    elif rv[0] == rv[1]:
        pat = 5 if (s[2] == s[0] or s[2] == s[1]) else 0
    elif rv[1] == rv[2]:
        pat = 2 if (s[0] == s[1] or s[0] == s[2]) else 0
    else:
        if s[0] == s[1] == s[2]:
            pat = 1
        elif s[0] == s[1]:
            pat = 2
        elif s[0] == s[2]:
            pat = 3
        elif s[1] == s[2]:
            pat = 4
        else:
            pat = 0
    return rv[0] * 100000 + rv[1] * 1000 + rv[2] * 10 + pat


def pf_lookup(cards3):
    """(equity, percentile) for a 3-card holding."""
    if len(_PF_KEYS) == 0:
        return 0.5, 0.5
    k = _triple_key_scalar(cards3)
    i = int(np.searchsorted(_PF_KEYS, k))
    if i < len(_PF_KEYS) and int(_PF_KEYS[i]) == k:
        return float(_PF_EQ[i]), float(_PF_PCT[i])
    return 0.5, 0.5


def pf_keys_batch(triples):
    """triples: (N,3) card ids -> class keys (N,). Matches scalar exactly."""
    t = np.asarray(triples, dtype=np.int64)
    t = -np.sort(-t, axis=1)  # descending per row
    rv = (t >> 2) + 2
    s = t & 3
    rv0, rv1, rv2 = rv[:, 0], rv[:, 1], rv[:, 2]
    s0, s1, s2 = s[:, 0], s[:, 1], s[:, 2]
    pat = np.zeros(t.shape[0], dtype=np.int64)
    trips = (rv0 == rv1) & (rv1 == rv2)
    topp = (rv0 == rv1) & (~trips)
    botp = (rv1 == rv2) & (~trips) & (~topp)
    distinct = ~trips & ~topp & ~botp
    pat = np.where(topp, np.where((s2 == s0) | (s2 == s1), 5, 0), pat)
    pat = np.where(botp, np.where((s0 == s1) | (s0 == s2), 2, 0), pat)
    pat = np.where(distinct & (s0 == s1) & (s1 == s2), 1, pat)
    pat = np.where(distinct & ~((s0 == s1) & (s1 == s2)) & (s0 == s1), 2, pat)
    pat = np.where(distinct & ~((s0 == s1) & (s1 == s2)) & (s0 != s1) & (s0 == s2), 3, pat)
    pat = np.where(distinct & ~((s0 == s1) & (s1 == s2)) & (s0 != s1) & (s0 != s2) & (s1 == s2), 4, pat)
    return rv0 * 100000 + rv1 * 1000 + rv2 * 10 + pat


def pf_pct_batch(triples):
    """percentiles for (N,3) triples (vectorised)."""
    if len(_PF_KEYS) == 0:
        return np.full(len(triples), 0.5)
    keys = pf_keys_batch(triples)
    idx = np.searchsorted(_PF_KEYS, keys)
    idx = np.clip(idx, 0, len(_PF_KEYS) - 1)
    ok = _PF_KEYS[idx] == keys
    return np.where(ok, _PF_PCT[idx], 0.5)


def pf_eq_batch(triples):
    if len(_PF_KEYS) == 0:
        return np.full(len(triples), 0.5)
    keys = pf_keys_batch(triples)
    idx = np.searchsorted(_PF_KEYS, keys)
    idx = np.clip(idx, 0, len(_PF_KEYS) - 1)
    ok = _PF_KEYS[idx] == keys
    return np.where(ok, _PF_EQ[idx], 0.5)


# warm up numpy evaluator paths once (startup time is not charged)
try:
    eval7_batch(np.arange(14, dtype=np.int64).reshape(2, 7) % 52)
    pf_keys_batch(np.array([[0, 1, 2], [48, 44, 41]], dtype=np.int64))
except Exception:
    pass


def clamp(x, lo, hi):
    return lo if x < lo else (hi if x > hi else x)


def hand_seed(cards, salt=""):
    """Deterministic roll in [0,1) keyed by the hole cards (+salt)."""
    key = salt + "".join(CARD_TEXT[int(c)] for c in sorted((int(c) for c in cards)))
    return random.Random(hash(key) & 0xFFFFFFFF).random()


# ============================================================ precomputed combos
import itertools as _it

_IDX3_46 = np.array(list(_it.combinations(range(46), 3)), dtype=np.int16)
_IDX2_44 = np.array(list(_it.combinations(range(44), 2)), dtype=np.int16)
_IDX2_45 = np.array(list(_it.combinations(range(45), 2)), dtype=np.int16)
_IDX2_46 = np.array(list(_it.combinations(range(46), 2)), dtype=np.int16)

_RNG = np.random.default_rng(0xA71A5)


# ------------------------------------------------------------------- clock
class Clock:
    def __init__(self):
        self.bank_ms = 30000.0
        self.total_hands = 200

    def update(self, bank_ms, total_hands):
        self.bank_ms = float(bank_ms)
        self.total_hands = int(total_hands)

    def budget_ms(self, time_left_ms, hand_id, weight=1.0):
        hands_left = max(1, self.total_hands - hand_id + 1)
        reserve = max(2500.0, 0.12 * self.bank_ms)
        usable = max(0.0, float(time_left_ms) - reserve)
        return clamp(min(usable / hands_left * weight, 900.0), 2.0, 900.0)

    @staticmethod
    def panic(time_left_ms):
        return time_left_ms < 3500.0

    @staticmethod
    def emergency(time_left_ms):
        return time_left_ms < 1200.0


# ------------------------------------------------------------ opponent stats
class OppStats:
    """Lightweight per-leg opponent statistics with MDF-shrunk estimates."""

    def __init__(self):
        # postflop: (street, size_bucket) -> [folds, calls, raises, bets_faced]
        self.facing = {}
        # (street,) -> [opens, checks_when_open]  (opportunities to open)
        self.open = {"flop": [0, 0], "turn": [0, 0], "river": [0, 0]}
        # preflop line stats
        self.pf = {
            "btn_open": [0, 0],      # [opens, opportunities]
            "btn_fold_sb": [0, 0],
            "bb_3bet_vs_open": [0, 0],
            "bb_fold_vs_open": [0, 0],
            "btn_fold_vs_3bet": [0, 0],
            "btn_call_vs_3bet": [0, 0],
            "bb_fold_vs_4bet": [0, 0],
            "bb_call_vs_4bet": [0, 0],
            "btn_vs_3bet": [0, 0],
            "btn_vs_4bet": [0, 0],
        }
        # showdown revealed hands: pct by last-aggro street
        self.sd_pct = {"flop": [0.0, 0], "turn": [0.0, 0], "river": [0.0, 0], "none": [0.0, 0]}
        self.sd_won = {"flop": [0.0, 0], "turn": [0.0, 0], "river": [0.0, 0], "none": [0.0, 0]}
        # discard predictability
        self.disc_match = [0, 0]
        self.hands_seen = 0
        # shown percentiles of their bet lines (bluff-catch calibration)
        self.bet_sd = []
        # adaptive jam model: shown strength of their 120+ raises, by street
        self.jam_pf_sd = []     # preflop jams
        self.jam_flop_sd = []   # flop/postflop jams
        self.jam_count = 0

    def jam_range_mu(self, default=0.86, pf=False):
        """Mean strength of their observed big-raise (jam) lines."""
        src = self.jam_pf_sd if pf else self.jam_flop_sd
        if len(src) < 2:
            return default, 0.12
        mean = sum(src) / len(src)
        return clamp(mean - 0.04, 0.52, 0.92), 0.16

    # ---------- preflop ----------
    def pf_rate(self, key, prior):
        a, n = self.pf[key]
        if n <= 0:
            return prior
        return (a + prior * 6.0) / (n + 6.0)

    # ---------- postflop ----------
    @staticmethod
    def size_bucket(bet, pot_before):
        if pot_before <= 0:
            return "big"
        f = bet / pot_before
        if f < 0.40:
            return "small"
        if f < 0.90:
            return "mid"
        return "big"

    def observe_facing(self, street, bucket, action):
        d = self.facing.setdefault((street, bucket), [0, 0, 0, 0])
        d[3] += 1
        if action == "fold":
            d[0] += 1
        elif action == "call":
            d[1] += 1
        elif action == "raise":
            d[2] += 1

    def observe_open(self, street, opened):
        self.open[street][1] += 1
        if opened:
            self.open[street][0] += 1

    def fold_rate(self, street, bucket, mdf_floor):
        """Estimated fold-vs-bet probability, MDF-shrunk."""
        d = self.facing.get((street, bucket))
        if d is None or d[3] < 1:
            return mdf_floor
        n = d[3]
        obs = d[0] / n
        return (obs * n + mdf_floor * 10.0) / (n + 10.0)

    def call_rate(self, street, bucket, prior):
        d = self.facing.get((street, bucket))
        if d is None or d[3] < 1:
            return prior
        n = d[3]
        return (d[1] + prior * 10.0) / (n + 10.0)

    def raise_rate(self, street, bucket, prior):
        d = self.facing.get((street, bucket))
        if d is None or d[3] < 1:
            return prior
        n = d[3]
        return (d[2] + prior * 14.0) / (n + 14.0)

    def open_rate(self, street, prior):
        o, n = self.open[street]
        if n < 1:
            return prior
        return (o + prior * 10.0) / (n + 10.0)

    # ---------- personas ----------
    def total_facing(self):
        return sum(d[3] for d in self.facing.values())

    def total_postflop_aggr(self):
        o = sum(self.open[s][0] for s in self.open)
        r = sum(d[2] for d in self.facing.values())
        return o + r

    def total_postflop_passive(self):
        c = sum(d[1] for d in self.facing.values())
        n = sum(self.open[s][1] for s in self.open)
        return c + n

    def persona(self):
        """Returns one of: station, maniac, nit, normal (postflop lens)."""
        tf = self.total_facing()
        if tf < 6 and self.hands_seen < 8:
            return "normal"
        folds = sum(d[0] for d in self.facing.values()) / max(1, tf)
        aggr = self.total_postflop_aggr()
        passive = self.total_postflop_passive()
        af = (aggr + 1.0) / (passive + 1.0)
        calls = sum(d[1] for d in self.facing.values()) / max(1, tf)
        if calls > 0.62 and folds < 0.22:
            return "station"
        if af > 2.2 and calls < 0.42:
            return "maniac"
        if folds > 0.62:
            return "nit"
        return "normal"

    def observe_showdown(self, aggro_street, pct, won):
        if aggro_street in self.sd_pct:
            self.sd_pct[aggro_street][0] += pct
            self.sd_pct[aggro_street][1] += 1
            self.sd_won[aggro_street][0] += 1.0 if won else 0.0
            self.sd_won[aggro_street][1] += 1

    def sd_avg_pct(self, street, default):
        s, n = self.sd_pct.get(street, (0.0, 0))
        if n < 2:
            return default
        return (s / n + default * 3.0) / (n + 3.0)


# ------------------------------------------------------- betting replay
def replay_history(history, my_seat_is_button):
    """Annotate each history entry with betting context.
    Yields dicts: {street, actor, action, amount, pot_before, inc, to_call, f}.
      pot_before: total committed before the action (incl. current street bets)
      inc:        this action's chip increment (bets/raises)
      f:          inc / pot_before (bet fraction), 0 for check/call/fold
    """
    out = []
    bets = {"me": 0, "opp": 0}
    pot_carry = 0.0
    acted = {"me": False, "opp": False}
    cur = "preflop"
    if my_seat_is_button:
        bets = {"me": 1, "opp": 2}
    else:
        bets = {"me": 2, "opp": 1}
    first = True
    for e in history:
        st = e.get("street", "preflop")
        if st != cur:
            pot_carry += bets["me"] + bets["opp"]
            cur = st
            bets = {"me": 0, "opp": 0}
            acted = {"me": False, "opp": False}
            first = True
        actor = e.get("actor", "opp")
        act = e.get("action")
        amt = e.get("amount")
        pot_before = pot_carry + bets["me"] + bets["opp"]
        other = "opp" if actor == "me" else "me"
        to_call = bets[other] - bets[actor]
        inc = 0
        f = 0.0
        if act == "raise":
            inc = int(amt) - bets[actor]
            f = inc / max(1.0, pot_before)
            bets[actor] = int(amt)
            acted[other] = False
        elif act == "call":
            inc = to_call
            bets[actor] = bets[other]
        acted[actor] = True
        out.append({
            "street": st, "actor": actor, "action": act, "amount": amt,
            "pot_before": pot_before, "inc": inc, "to_call": to_call, "f": f,
        })
        if acted["me"] and acted["opp"] and bets["me"] == bets["opp"]:
            pot_carry += bets["me"] + bets["opp"]
            bets = {"me": 0, "opp": 0}
            acted = {"me": False, "opp": False}
            # next street detection handled by street tag on next entry
            if st == "preflop":
                cur = "flop_pending"
    return out




# ============================================================ range models

_POPCNT = np.zeros(8192, dtype=np.int8)
for _m in range(8192):
    _POPCNT[_m] = bin(_m).count("1")

_SDRAW_ARR = np.zeros(8192, dtype=np.float64)
for _m in range(8192):
    _SDRAW_ARR[_m] = _SDRAW[_m]


def gauss_w(x, mu, sigma):
    z = (x - mu) / max(0.02, sigma)
    return math.exp(-0.5 * z * z)


def _board_bits(flop3):
    """(rank_bitmask, suit_counts[4], is_board_rank[13]) for a 3-card flop."""
    mask = 0
    fs = np.zeros(4, dtype=np.int64)
    isb = np.zeros(13, dtype=bool)
    for c in flop3:
        r = c >> 2
        mask |= 1 << r
        isb[r] = True
        fs[c & 3] += 1
    return mask, fs, isb


def flop_trip_strength(triples, flop3):
    """Cheap additive strength bump [0..~0.6] for triples on this flop."""
    t = np.asarray(triples, dtype=np.int64)
    mask, fs, isb = _board_bits(flop3)
    r = t >> 2
    s = t & 3
    hi_board = max(int(c) >> 2 for c in flop3)
    hits = isb[r].sum(axis=1).astype(np.float64)      # cards pairing the board
    pair_rank_bonus = np.zeros(t.shape[0], dtype=np.float64)
    board_ranks = sorted({int(c) >> 2 for c in flop3}, reverse=True)
    for br in board_ranks:
        has = (r == br).sum(axis=1) > 0
        pair_rank_bonus += has * (0.06 + 0.008 * br)
    flushy = np.zeros(t.shape[0], dtype=np.float64)
    for su in range(4):
        if fs[su] >= 2:
            cnt = (s == su).sum(axis=1)
            flushy += (cnt >= 1) * 0.03 * fs[su] * cnt
    overs = (r > hi_board).sum(axis=1) * 0.018
    hi = r.max(axis=1) * 0.010
    return np.minimum(hits * 0.0 + pair_rank_bonus + flushy + overs + hi, 0.6)


def build_flop_range(my3, flop3, mu, sigma, agg_bump=0.0):
    """Weighted triple range for the opponent at flop time.
    Returns (triples (M,3), weights (M,) summing to 1)."""
    mine = set(int(c) for c in my3)
    fl = set(int(c) for c in flop3)
    unknown = np.array([c for c in range(52) if c not in mine and c not in fl],
                       dtype=np.int64)
    m = len(unknown)
    if m == 46:
        idx = _IDX3_46
    else:
        idx = np.array(list(_it.combinations(range(m), 3)), dtype=np.int16)
    triples = unknown[idx]
    pct = pf_pct_batch(triples)
    mu_eff = mu + agg_bump
    sigma_eff = sigma * (1.0 - 0.45 * min(1.0, abs(agg_bump)))
    w = np.exp(-0.5 * ((pct - mu_eff) / max(0.03, sigma_eff)) ** 2)
    if agg_bump != 0.0:
        w *= (1.0 + flop_trip_strength(triples, flop3) * min(2.5, abs(agg_bump) * 6.0))
    tot = float(w.sum())
    if tot <= 1e-9:
        w = np.full(len(triples), 1.0 / max(1, len(triples)))
    else:
        w = w / tot
    return triples, w.astype(np.float64)


def _pair_index_table(m):
    if m == 44:
        return _IDX2_44
    if m == 45:
        return _IDX2_45
    if m == 46:
        return _IDX2_46
    return np.array(list(_it.combinations(range(m), 2)), dtype=np.int16)


def keep_score_vec(k1, k2, flop3):
    """Vectorised keep_score mirroring the scalar heuristic exactly."""
    f1, f2, f3 = int(flop3[0]), int(flop3[1]), int(flop3[2])
    fr1, fr2, fr3 = f1 >> 2, f2 >> 2, f3 >> 2
    fs1, fs2, fs3 = f1 & 3, f2 & 3, f3 & 3
    k1 = np.asarray(k1, dtype=np.int64)
    k2 = np.asarray(k2, dtype=np.int64)
    r1 = k1 >> 2; r2 = k2 >> 2
    s1 = k1 & 3; s2 = k2 & 3
    hp = (r1 == r2).astype(np.float64)
    m1 = ((r1 == fr1).astype(np.float64) + (r1 == fr2) + (r1 == fr3))
    m2 = ((r2 == fr1).astype(np.float64) + (r2 == fr2) + (r2 == fr3))
    c1 = 1.0 + hp + m1
    c2 = 1.0 + hp + m2
    maxc = np.maximum(c1, c2)
    btrips = (fr1 == fr2) and (fr2 == fr3)
    pairf = (fr1 == fr2) or (fr2 == fr3) or (fr1 == fr3)
    if btrips:
        maxc = np.maximum(maxc, 3.0)
    elif pairf:
        maxc = np.maximum(maxc, 2.0)
    fm = (1 << fr1) | (1 << fr2) | (1 << fr3)
    both = np.left_shift(1, r1) | np.left_shift(1, r2) | fm
    nd = _POPCNT[both].astype(np.float64)
    made = np.where(nd == 2, 5.5,
                    np.where(c1 >= 3, 3.0 + 0.05 * r1, 3.0 + 0.05 * r2))   # maxc==3
    pr = np.where(hp > 0, r1,
                  np.where(m1 > 0, r1,
                           np.where(m2 > 0, r2,
                                    np.where(fr1 == fr2, fr1, fr2))))
    made2 = np.where(nd == 3, 2.0, 1.0 + 0.05 * pr)
    made = np.where(maxc == 2, made2, made)
    made = np.where(maxc < 2, 0.06 * np.maximum(r1, r2), made)
    made = np.where(maxc >= 4, 6.0, made)
    # flush: per-suit rank masks
    fc = np.zeros_like(r1, dtype=np.float64)
    for su in range(4):
        m = ((s1 == su) * np.left_shift(1, r1) + (s2 == su) * np.left_shift(1, r2)).astype(np.int64)
        fbit = 0
        if fs1 == su: fbit |= 1 << fr1
        if fs2 == su: fbit |= 1 << fr2
        if fs3 == su: fbit |= 1 << fr3
        m |= fbit
        fc = np.maximum(fc, _POPCNT[m])
    made = np.where(fc >= 5, np.maximum(made, 5.0), made)
    made = np.where(fc == 4, made + 0.55, made)
    st = STRAIGHT_HIGH[both]
    sd = _SDRAW_ARR[both]
    made = np.where(st >= 0, np.maximum(made, 4.0), made + sd)
    return made + (r1 + r2) * 0.004




def build_pair_range(my2, board, opp_discard, mu_pre, sig_pre,
                     street_mu, street_sigma, flop3, sharpness=2.0,
                     flop_agg_bump=0.0):
    """Weighted pair range post-discard, using the exposed discard as evidence.
    Returns (pairs (M,2), weights (M,) summing to 1, unknown cards array)."""
    known = set(int(c) for c in my2) | set(int(c) for c in board)
    known.add(int(opp_discard))
    unknown = np.array([c for c in range(52) if c not in known], dtype=np.int64)
    m = len(unknown)
    idx = _pair_index_table(m)
    pairs = unknown[idx].astype(np.int64)
    triples = np.zeros((pairs.shape[0], 3), dtype=np.int64)
    triples[:, 0:2] = pairs
    triples[:, 2] = int(opp_discard)
    pct = pf_pct_batch(triples)
    w = np.exp(-0.5 * ((pct - mu_pre) / max(0.03, sig_pre)) ** 2)
    if flop_agg_bump != 0.0:
        w *= (1.0 + flop_trip_strength(triples, flop3) * min(2.5, abs(flop_agg_bump) * 6.0))
    # discard likelihood: softmax over the 3 keep options; they dropped the 3rd
    if sharpness > 0.01:
        sc = np.empty((pairs.shape[0], 3), dtype=np.float64)
        for i in range(3):
            if i == 0:
                k1 = triples[:, 1]; k2 = triples[:, 2]
            elif i == 1:
                k1 = triples[:, 0]; k2 = triples[:, 2]
            else:
                k1 = triples[:, 0]; k2 = triples[:, 1]
            sc[:, i] = keep_score_vec(k1, k2, flop3)
        mx = sc.max(axis=1, keepdims=True)
        ex = np.exp(np.clip(sharpness * (sc - mx), -60.0, 60.0))
        ex /= ex.sum(axis=1, keepdims=True)
        w *= ex[:, 2]
    # current made-strength likelihood from their street actions
    if street_mu is not None and len(board) == 5:
        cards7 = np.zeros((len(pairs), 7), dtype=np.int64)
        cards7[:, 0:2] = pairs
        cards7[:, 2:7] = np.array(list(board), dtype=np.int64)
        sc7 = eval7_batch(cards7)
        spct = np.array([hand_pct(int(v)) for v in sc7])
        w *= np.exp(-0.5 * ((spct - street_mu) / max(0.03, street_sigma)) ** 2)
    elif street_mu is not None and len(board) == 4:
        # turn: real made-hand strength via 6-card evaluation
        cards6 = np.zeros((pairs.shape[0], 6), dtype=np.int64)
        cards6[:, 0:2] = pairs
        cards6[:, 2:6] = np.array(list(board), dtype=np.int64)
        sc6 = eval7_batch(cards6)
        spct6 = np.array([hand_pct(int(v)) for v in sc6])
        w *= np.exp(-0.5 * ((spct6 - street_mu) / max(0.03, street_sigma)) ** 2)
    tot = float(w.sum())
    if tot <= 1e-9:
        w = np.full(len(pairs), 1.0 / max(1, len(pairs)))
    else:
        w = w / tot
    return pairs, w.astype(np.float64), unknown


# ------------------------------------------------------------ equity engine
def equity_flop(my3, flop3, triples, w, n, rng, hero_keep=None):
    """MC equity on the flop (pre-discard) of my 3 cards vs weighted triples.
    Both sides keep the best 2 of 3 given the sampled runout."""
    m = len(triples)
    if m == 0:
        return 0.5
    p = w / w.sum()
    ti = rng.choice(m, size=n, p=p) if m > 1 else np.zeros(n, dtype=np.int64)
    opp_t = triples[ti]
    known = set(int(c) for c in my3) | set(int(c) for c in flop3)
    unknown_all = np.array([c for c in range(52) if c not in known], dtype=np.int64)
    u = len(unknown_all)
    rand = rng.random((n, u))
    mem = np.zeros((n, u), dtype=bool)
    rows = np.arange(n)
    for j in range(3):
        pos = np.minimum(np.searchsorted(unknown_all, opp_t[:, j]), u - 1)
        hit = unknown_all[pos] == opp_t[:, j]
        if hit.any():
            mem[rows[hit], pos[hit]] = True
    order = np.argsort(rand + mem * 2.0, axis=1)
    future = unknown_all[order[:, :2]]                     # (n,2) runout cards
    board = np.zeros((n, 5), dtype=np.int64)
    board[:, 0:3] = np.array(list(flop3), dtype=np.int64)
    board[:, 3:5] = future
    if hero_keep is not None:
        cards = np.zeros((n, 7), dtype=np.int64)
        cards[:, 0:2] = np.array(list(hero_keep), dtype=np.int64)
        cards[:, 2:7] = board
        hero = eval7_batch(cards)
    else:
        hero = np.zeros(n, dtype=np.int64)
        for drop in range(3):
            keep = [my3[j] for j in range(3) if j != drop]
            cards = np.zeros((n, 7), dtype=np.int64)
            cards[:, 0] = keep[0]; cards[:, 1] = keep[1]
            cards[:, 2:7] = board
            hero = np.maximum(hero, eval7_batch(cards))
    opp = np.zeros(n, dtype=np.int64)
    for drop in range(3):
        keep = np.zeros((n, 2), dtype=np.int64)
        keep[:, 0] = opp_t[:, (drop + 1) % 3]
        keep[:, 1] = opp_t[:, (drop + 2) % 3]
        cards = np.zeros((n, 7), dtype=np.int64)
        cards[:, 0:2] = keep
        cards[:, 2:7] = board
        opp = np.maximum(opp, eval7_batch(cards))
    return float(((hero > opp).sum() + 0.5 * (hero == opp).sum()) / n)


def equity_river(my2, board5, pairs, w):
    """Exact weighted equity at the river."""
    if len(pairs) == 0:
        return 0.5
    cards7 = np.zeros((len(pairs), 7), dtype=np.int64)
    cards7[:, 0:2] = np.asarray(pairs, dtype=np.int64)
    cards7[:, 2:7] = np.array(list(board5), dtype=np.int64)
    opp = eval7_batch(cards7)
    hero = eval7_scalar([int(c) for c in my2] + [int(c) for c in board5])
    win = (hero > opp).astype(np.float64)
    tie = (hero == opp).astype(np.float64)
    return float((w * (win + 0.5 * tie)).sum() / max(1e-9, w.sum()))


def equity_turn_mc(my2, board4, pairs, w, unknown, n):
    m = len(pairs)
    if m == 0:
        return 0.5
    p = w / w.sum()
    ti = _RNG.choice(m, size=n, p=p)
    pr = pairs[ti]
    u = np.asarray(unknown, dtype=np.int64)
    ru = len(u)
    rand = _RNG.random((n, ru))
    mem = np.zeros((n, ru), dtype=bool)
    rows = np.arange(n)
    for j in range(2):
        pos = np.minimum(np.searchsorted(u, pr[:, j]), ru - 1)
        hit = u[pos] == pr[:, j]
        if hit.any():
            mem[rows[hit], pos[hit]] = True
    order = np.argsort(rand + mem * 2.0, axis=1)
    riv = u[order[:, 0]]
    cards7 = np.zeros((n, 7), dtype=np.int64)
    cards7[:, 0:2] = np.array(list(my2), dtype=np.int64)
    cards7[:, 2:6] = np.array(list(board4), dtype=np.int64)
    cards7[:, 6] = riv
    hero = eval7_batch(cards7)
    cards7[:, 0:2] = pr
    opp = eval7_batch(cards7)
    return float(((hero > opp).sum() + 0.5 * (hero == opp).sum()) / n)


def equity_turn_exact(my2, board4, pairs, w, unknown, deadline=None):
    """Exact weighted turn equity (river x pairs grid). MC fallback on deadline."""
    if len(pairs) == 0:
        return 0.5
    if deadline is not None and time.perf_counter() > deadline:
        return equity_turn_mc(my2, board4, pairs, w, unknown, 1600)
    m = len(pairs)
    r = len(unknown)
    rivers = np.asarray(unknown, dtype=np.int64)
    hb = np.zeros((r, 7), dtype=np.int64)
    hb[:, 0:2] = np.array(list(my2), dtype=np.int64)
    hb[:, 2:6] = np.array(list(board4), dtype=np.int64)
    hb[:, 6] = rivers
    hero_r = eval7_batch(hb)
    b4 = np.array(list(board4), dtype=np.int64)
    pair0 = np.repeat(pairs[:, 0], r).astype(np.int64)
    pair1 = np.repeat(pairs[:, 1], r).astype(np.int64)
    riv = np.tile(rivers, m).astype(np.int64)
    cards7 = np.zeros((m * r, 7), dtype=np.int64)
    cards7[:, 0] = pair0
    cards7[:, 1] = pair1
    cards7[:, 2:6] = np.tile(b4, m * r).reshape(m * r, 4)
    cards7[:, 6] = riv
    if deadline is not None and time.perf_counter() > deadline:
        return equity_turn_mc(my2, board4, pairs, w, unknown, 1600)
    opp = eval7_batch(cards7)
    hero = np.tile(hero_r, m)
    wr = np.repeat(w, r)
    ok = (riv != pair0) & (riv != pair1)
    win = ((hero > opp) & ok).astype(np.float64)
    tie = ((hero == opp) & ok).astype(np.float64)
    tot = float((wr * ok).sum())
    if tot <= 1e-9:
        return 0.5
    return float((wr * (win + 0.5 * tie)).sum() / tot)


def equity_preflop_range(my3, mu, sigma, n, rng, continue_shift=0.0):
    """Weighted MC preflop equity vs a percentile-modelled range."""
    mine = set(int(c) for c in my3)
    unknown = np.array([c for c in range(52) if c not in mine], dtype=np.int64)
    u = len(unknown)
    rand = rng.random((n, u))
    order = np.argsort(rand, axis=1)[:, :8]
    draw = unknown[order]
    opp_t = draw[:, 0:3]
    board = draw[:, 3:8]
    pct = pf_pct_batch(opp_t)
    mu_eff = mu + continue_shift
    wgt = np.exp(-0.5 * ((pct - mu_eff) / max(0.03, sigma)) ** 2)
    hero = np.zeros(n, dtype=np.int64)
    for drop in range(3):
        keep = [int(my3[j]) for j in range(3) if j != drop]
        cards = np.zeros((n, 7), dtype=np.int64)
        cards[:, 0] = keep[0]; cards[:, 1] = keep[1]
        cards[:, 2:7] = board
        hero = np.maximum(hero, eval7_batch(cards))
    opp = np.zeros(n, dtype=np.int64)
    for drop in range(3):
        keep = np.zeros((n, 2), dtype=np.int64)
        keep[:, 0] = opp_t[:, (drop + 1) % 3]
        keep[:, 1] = opp_t[:, (drop + 2) % 3]
        cards = np.zeros((n, 7), dtype=np.int64)
        cards[:, 0:2] = keep
        cards[:, 2:7] = board
        opp = np.maximum(opp, eval7_batch(cards))
    win = (hero > opp).astype(np.float64)
    tie = (hero == opp).astype(np.float64)
    tw = float(wgt.sum())
    if tw <= 1e-9:
        return 0.5
    return float((wgt * (win + 0.5 * tie)).sum() / tw)


def discard_option_equities(my3, flop3, triples, w, n, rng):
    """Equity of each of the 3 keep options vs the weighted triple range."""
    m = len(triples)
    if m == 0:
        return [0.5, 0.5, 0.5]
    p = w / w.sum()
    ti = rng.choice(m, size=n, p=p) if m > 1 else np.zeros(n, dtype=np.int64)
    opp_t = triples[ti]
    known = set(int(c) for c in my3) | set(int(c) for c in flop3)
    unknown_all = np.array([c for c in range(52) if c not in known], dtype=np.int64)
    u = len(unknown_all)
    rand = rng.random((n, u))
    mem = np.zeros((n, u), dtype=bool)
    rows = np.arange(n)
    for j in range(3):
        pos = np.minimum(np.searchsorted(unknown_all, opp_t[:, j]), u - 1)
        hit = unknown_all[pos] == opp_t[:, j]
        if hit.any():
            mem[rows[hit], pos[hit]] = True
    order = np.argsort(rand + mem * 2.0, axis=1)
    future = unknown_all[order[:, :2]]
    board = np.zeros((n, 5), dtype=np.int64)
    board[:, 0:3] = np.array(list(flop3), dtype=np.int64)
    board[:, 3:5] = future
    opp = np.zeros(n, dtype=np.int64)
    for drop in range(3):
        keep = np.zeros((n, 2), dtype=np.int64)
        keep[:, 0] = opp_t[:, (drop + 1) % 3]
        keep[:, 1] = opp_t[:, (drop + 2) % 3]
        cards = np.zeros((n, 7), dtype=np.int64)
        cards[:, 0:2] = keep
        cards[:, 2:7] = board
        opp = np.maximum(opp, eval7_batch(cards))
    eqs = []
    for drop in range(3):
        keep = [my3[j] for j in range(3) if j != drop]
        cards = np.zeros((n, 7), dtype=np.int64)
        cards[:, 0] = keep[0]; cards[:, 1] = keep[1]
        cards[:, 2:7] = board
        hero = eval7_batch(cards)
        eqs.append(float(((hero > opp).sum() + 0.5 * (hero == opp).sum()) / n))
    return eqs


# ======================================================= range/response v2
def range_scores(pairs, board):
    """Exact best-5 scores for (N,2) pairs on the given board (any length)."""
    pairs = np.asarray(pairs, dtype=np.int64)
    n = len(pairs)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    cards = np.zeros((n, 2 + len(board)), dtype=np.int64)
    cards[:, 0:2] = pairs
    for k, c in enumerate(board):
        cards[:, 2 + k] = c
    return eval7_batch(cards)


def _cluster_scores(scores, w, max_clusters=340):
    """Collapse a weighted score distribution to <=max_clusters representatives."""
    scores = np.asarray(scores, dtype=np.int64)
    w = np.asarray(w, dtype=np.float64)
    n = len(scores)
    if n == 0:
        return scores, w
    if n <= max_clusters:
        return scores, w
    lo = int(scores.min()); hi = int(scores.max())
    if hi <= lo:
        return np.array([lo], dtype=np.int64), np.array([w.sum()])
    edges = np.linspace(lo, hi + 1, max_clusters + 1)
    bin_id = np.clip(np.searchsorted(edges, scores, side="right") - 1, 0, max_clusters - 1)
    wts = np.zeros(max_clusters)
    np.add.at(wts, bin_id, w)
    wsc = np.zeros(max_clusters)
    np.add.at(wsc, bin_id, w * scores)
    rep = (wsc / np.maximum(wts, 1e-12)).astype(np.int64)
    keep = wts > 1e-12
    return rep[keep], wts[keep]


def eq_vs_range(my_score, opp_scores, opp_w):
    """Exact weighted equity of one score vs a weighted score distribution."""
    tot = float(opp_w.sum())
    if len(opp_scores) == 0 or tot <= 1e-9:
        return 0.5
    win = opp_scores < my_score
    tie = opp_scores == my_score
    return float((opp_w * (win + 0.5 * tie)).sum() / tot)


def eq_matrix(opp_scores, us_scores, us_w):
    """Each opp score's equity vs our weighted score distribution."""
    tot = float(us_w.sum())
    if len(us_scores) == 0 or tot <= 1e-9:
        return np.full(len(opp_scores), 0.5)
    gt = opp_scores[:, None] > us_scores[None, :]
    eq = gt + 0.5 * (opp_scores[:, None] == us_scores[None, :])
    return (eq * us_w[None, :]).sum(axis=1) / tot


def _cluster_kv(scores, w, vals, max_clusters=340):
    """Cluster weighted (score, value) pairs into <=max_clusters score bins.
    Returns (rep_scores, weights, mean_vals)."""
    scores = np.asarray(scores, dtype=np.int64)
    w = np.asarray(w, dtype=np.float64)
    vals = np.asarray(vals, dtype=np.float64)
    n = len(scores)
    if n == 0:
        return scores, w, vals
    if n <= max_clusters:
        # still merge identical scores
        su, inv = np.unique(scores, return_inverse=True)
        if len(su) == n:
            return scores, w, vals
        k = len(su)
        wts = np.zeros(k)
        np.add.at(wts, inv, w)
        vsc = np.zeros(k)
        np.add.at(vsc, inv, w * vals)
        vm = vsc / np.maximum(wts, 1e-12)
        keep = wts > 1e-12
        return su[keep], wts[keep], vm[keep]
    lo, hi = int(scores.min()), int(scores.max())
    if hi <= lo:
        return (np.array([lo], dtype=np.int64),
                np.array([w.sum()]),
                np.array([float((w * vals).sum() / max(1e-9, w.sum()))]))
    edges = np.linspace(lo, hi + 1, max_clusters + 1)
    bid = np.clip(np.searchsorted(edges, scores, side="right") - 1, 0, max_clusters - 1)
    wts = np.zeros(max_clusters)
    np.add.at(wts, bid, w)
    wsc = np.zeros(max_clusters)
    np.add.at(wsc, bid, w * scores)
    wvc = np.zeros(max_clusters)
    np.add.at(wvc, bid, w * vals)
    rep = (wsc / np.maximum(wts, 1e-12)).astype(np.int64)
    vm = wvc / np.maximum(wts, 1e-12)
    keep = wts > 1e-12
    return rep[keep], wts[keep], vm[keep]


class LineModel:
    """Shared Bayesian line model used for BOTH players (opponent and self-mirror)."""
    __slots__ = ("mu", "sigma", "flop_agg", "street_mu", "barrels", "floor")

    def __init__(self):
        self.reset()

    def reset(self):
        self.mu, self.sigma = 0.50, 0.30
        self.flop_agg = 0.0
        self.street_mu = None
        self.barrels = 0
        self.floor = 0.0


def _line_preflop(ln, action, ann, lvl, first_vol):
    """Update a LineModel from one preflop action (mirrors the opp branches)."""
    tc = ann["to_call"]; inc = ann["inc"]; f = ann["f"]
    if action == "raise":
        if inc >= 190:
            ln.mu, ln.sigma = 0.88, 0.12
        elif tc == 1:
            ln.mu = clamp(0.47 + 0.02 * (ann["amount"] - 4), 0.42, 0.68)
            ln.sigma = 0.27
        elif tc == 0:
            ln.mu, ln.sigma = 0.72, 0.16
        else:
            if lvl == 1:
                ln.mu, ln.sigma = 0.80, 0.11
            elif lvl == 2:
                ln.mu, ln.sigma = 0.955, 0.035
            else:
                ln.mu, ln.sigma = 0.93, 0.05
    elif action == "call":
        if tc <= 1:
            ln.mu, ln.sigma = 0.30, 0.24
        else:
            if lvl == 0:
                ln.mu = clamp(0.40 + 0.015 * ann["amount"], 0.40, 0.55)
                ln.sigma = 0.24
            elif lvl == 1:
                ln.mu, ln.sigma = 0.58, 0.18
            else:
                ln.mu, ln.sigma = 0.78, 0.12
    elif action == "fold":
        pass


def _line_postflop(ln, street, action, ann):
    """Update a LineModel from one postflop action."""
    if street == "flop":
        if action == "raise":
            if ann["inc"] >= 60:
                ln.flop_agg += 0.55; ln.floor = max(ln.floor, 0.90)
            elif ann["f"] >= 0.9:
                ln.flop_agg += 0.40; ln.floor = max(ln.floor, 0.86)
            elif ann["f"] >= 0.4:
                ln.flop_agg += 0.42; ln.floor = max(ln.floor, 0.78)
            else:
                ln.flop_agg += 0.32; ln.floor = max(ln.floor, 0.74)
        elif action == "call":
            ln.flop_agg += 0.03
    elif street in ("turn", "river"):
        ln.street_mu = _line_street_mu(ln, action, ann)
        if action == "raise":
            ln.floor = max(ln.floor, 0.80 if street == "turn" else 0.82)
            ln.barrels += 1
            if ann["inc"] >= 60:
                ln.street_mu = (0.88, 0.09)
        elif action == "check":
            ln.barrels = 0
        elif action == "call" and ann["f"] >= 0.5:
            ln.floor = max(ln.floor, 0.66)


def _line_street_mu(ln, action, ann):
    f = ann["f"]
    if action == "raise":
        esc = 0.05 * max(0, ln.barrels - 1)
        if f >= 0.9:
            base = 0.80 + 0.10 * clamp((f - 0.9) / 1.5, 0.0, 1.0) + esc
            sig = 0.10
        else:
            base = 0.72 + 0.05 * f + esc
            sig = 0.13
    elif action == "call":
        base = 0.50 + 0.10 * clamp(f, 0.0, 1.5)
        sig = 0.18
    elif action == "check":
        base, sig = 0.42, 0.24
    else:
        return None
    return clamp(base, 0.25, 0.93), sig


def build_self_range(my_discard, board, flop3, ln, sharpness=1.8):
    """Range a rational opponent assigns US after the discard (we may hold any pair
    of unknown cards; our true hole stays inside with its proper weight)."""
    known = set(int(c) for c in board)
    if my_discard is not None:
        known.add(int(my_discard))
    unknown = np.array([c for c in range(52) if c not in known], dtype=np.int64)
    m = len(unknown)
    idx = _pair_index_table(m)
    pairs = unknown[idx].astype(np.int64)
    triples = np.zeros((pairs.shape[0], 3), dtype=np.int64)
    triples[:, 0:2] = pairs
    triples[:, 2] = int(my_discard) if my_discard is not None else -1
    pct = pf_pct_batch(triples)
    pre_sigma = max(0.30, ln.sigma)
    w = np.exp(-0.5 * ((pct - ln.mu) / max(0.03, pre_sigma)) ** 2)
    if ln.flop_agg != 0.0:
        w *= (1.0 + flop_trip_strength(triples, flop3) * min(2.5, abs(ln.flop_agg) * 6.0))
    if sharpness > 0.01 and my_discard is not None:
        sc = np.empty((pairs.shape[0], 3), dtype=np.float64)
        for i in range(3):
            if i == 0:
                k1 = triples[:, 1]; k2 = triples[:, 2]
            elif i == 1:
                k1 = triples[:, 0]; k2 = triples[:, 2]
            else:
                k1 = triples[:, 0]; k2 = triples[:, 1]
            sc[:, i] = keep_score_vec(k1, k2, flop3)
        mx = sc.max(axis=1, keepdims=True)
        ex = np.exp(np.clip(sharpness * (sc - mx), -60.0, 60.0))
        ex /= ex.sum(axis=1, keepdims=True)
        w *= ex[:, 2]
    if ln.street_mu is not None and len(board) >= 4:
        cards = np.zeros((pairs.shape[0], 2 + len(board)), dtype=np.int64)
        cards[:, 0:2] = pairs
        for k, c in enumerate(board):
            cards[:, 2 + k] = c
        spct = np.array([hand_pct(int(v)) for v in eval7_batch(cards)])
        mu, sig = ln.street_mu
        w *= np.exp(-0.5 * ((spct - mu) / max(0.03, sig)) ** 2)
    floor = ln.floor
    if floor > 0.0 and (ln.street_mu is None or ln.street_mu[0] < floor):
        pass  # floor only used for their model of us when they saw aggression
    tot = float(w.sum())
    if tot <= 1e-9:
        w = np.full(len(pairs), 1.0 / max(1, len(pairs)))
    else:
        w = w / tot
    return pairs, w.astype(np.float64)


def build_jam_flop_range(my3, flop3, frac=0.28, sharp=1.6):
    """Range of hands that HIT THIS FLOP hard (for evaluating vs jams/monster raises):
    triples weighted by softmax over their best keep-score on this board."""
    mine = set(int(c) for c in my3) | set(int(c) for c in flop3)
    unknown = np.array([c for c in range(52) if c not in mine], dtype=np.int64)
    m = len(unknown)
    if m == 46:
        idx = _IDX3_46
    else:
        idx = np.array(list(_it.combinations(range(m), 3)), dtype=np.int16)
    triples = unknown[idx]
    # best keep-score over the 3 discard options
    best = np.full(len(triples), -1e9)
    for drop in range(3):
        cols = [j for j in range(3) if j != drop]
        k1 = triples[:, cols[0]].astype(np.int64)
        k2 = triples[:, cols[1]].astype(np.int64)
        sc = keep_score_vec(k1, k2, flop3)
        best = np.maximum(best, sc)
    w = np.exp(sharp * (best - best.max()))
    k = max(8, int(len(triples) * frac))
    order = np.argsort(-w)
    sel = order[:k]
    wsel = w[sel]
    tot = float(wsel.sum())
    if tot <= 1e-9:
        wsel = np.full(len(sel), 1.0 / max(1, len(sel)))
    else:
        wsel = wsel / tot
    return triples[sel].astype(np.int64), wsel.astype(np.float64)


# ============================================================ decision engine
class AtlasBot:
    def __init__(self):
        self.clock = Clock()
        self.stats = OppStats()
        self.rng = np.random.default_rng(0x5EED5)
        # match/leg state
        self.total_hands = 200
        # per-hand state
        self.hand_id = 0
        self.hole3 = []
        self.seat_button = True
        self.board = []
        self.my_discard = None
        self.opp_discard = None
        self.seen_hist = 0
        self.last_hist = []
        self.replay_cache = []
        # opp-line model
        self.opp_mu = 0.50
        self.opp_sigma = 0.30
        self.flop_agg = 0.0
        self.my_pf_aggr = False      # we were the preflop aggressor
        self.last_req = None         # last action_request (raw)
        self.disc_roll_cache = None
        self.turn_street_mu = None
        self.flop_range_cache = None
        # self-mirror line model (what a rational opponent believes about US)
        self.self_ln = LineModel()
        self.self_range_cache = None

    # ------------------------------------------------------ history/stats
    def update_from_history(self, req):
        hist = req.get("history") or []
        if len(hist) < self.seen_hist:
            self.seen_hist = 0
        new = hist[self.seen_hist:]
        self.seen_hist = len(hist)
        self.last_hist = hist
        if not new:
            return
        rep = replay_history(hist, self.seat_button)
        # only look at entries that are new
        new_idx = len(hist) - len(new)
        for i in range(new_idx, len(hist)):
            e = hist[i]
            actor = e.get("actor")
            if actor not in ("opp", "me"):
                continue
            a = e.get("action")
            street = e.get("street", "preflop")
            ann = rep[i]
            if actor == "opp":
                self.observe_opp_action(street, a, ann)
            else:
                self.observe_self_action(street, a, ann)

    def observe_self_action(self, street, action, ann):
        """Mirror-update the model of OUR line (what they infer from our actions)."""
        if street == "preflop":
            lvl = self.my_pf_raise_level()
            _line_preflop(self.self_ln, action, ann, lvl, False)
            if action == "raise":
                self.my_pf_aggr = True
            return
        _line_postflop(self.self_ln, street, action, ann)
        if street in ("turn", "river") and action == "raise":
            self.self_range_cache = None

    def my_pf_raise_level(self):
        """Preflop raise count before now, from our own perspective."""
        n = 0
        for e in self.last_hist:
            if e.get("street") != "preflop":
                break
            if e.get("action") == "raise":
                n += 1
        return n - 1

    def jam_floor(self, default):
        """Strength floor for their big raises, adapted to observed jam widths."""
        if len(self.stats.jam_flop_sd) >= 2:
            mean = sum(self.stats.jam_flop_sd) / len(self.stats.jam_flop_sd)
            if mean < 0.62:
                return min(default, 0.68)
        return default

    def fold_vs_big(self, street):
        """Observed fold rate vs big bets (for giant-overbet discipline)."""
        d = self.stats.facing.get((street, "big"))
        if d is None or d[3] < 4:
            return None
        return d[0] / d[3]

    def observe_opp_action(self, street, action, ann):
        stats = self.stats
        if street == "preflop":
            # classify the line
            tc = ann["to_call"]
            inc = ann["inc"]
            f = ann["f"]
            pot_before = ann["pot_before"]
            first_vol = (self.opp_pf_first() and action in ("raise", "call"))
            if action == "raise":
                if inc >= 190:
                    self.opp_mu, self.opp_sigma = 0.88, 0.12
                elif tc == 1:
                    stats.pf["btn_open"][0] += 1
                    stats.pf["btn_open"][1] += 1
                    self.opp_mu = clamp(0.47 + 0.02 * (ann["amount"] - 4), 0.42, 0.68)
                    self.opp_sigma = 0.27
                elif tc == 0:
                    # big-blind option raise after our limp
                    self.opp_mu, self.opp_sigma = 0.72, 0.16
                else:
                    # 3bet/4bet; scale by size
                    lvl = self.opp_pf_raise_level()
                    if lvl == 1:
                        stats.pf["bb_3bet_vs_open"][0] += 1
                        stats.pf["bb_3bet_vs_open"][1] += 1
                        stats.pf["bb_fold_vs_open"][1] += 1
                        r3 = stats.pf_rate("bb_3bet_vs_open", 0.10)
                        self.opp_mu = 0.86 if r3 < 0.16 else 0.76
                        self.opp_sigma = 0.09 if r3 < 0.16 else 0.15
                    elif lvl == 2:
                        self.opp_mu, self.opp_sigma = 0.955, 0.035
                        stats.pf["btn_vs_4bet"][1] += 1
                    else:
                        self.opp_mu, self.opp_sigma = 0.93, 0.05
            elif action == "call":
                if tc <= 1:
                    # button limp or sb complete
                    self.opp_mu, self.opp_sigma = 0.30, 0.24
                    stats.pf["btn_open"][1] += 1
                else:
                    lvl = self.opp_pf_raise_level()
                    if lvl == 0:
                        self.opp_mu = clamp(0.40 + 0.015 * ann["amount"], 0.40, 0.55)
                        self.opp_sigma = 0.24
                        stats.pf["bb_fold_vs_open"][1] += 1
                    elif lvl == 1:
                        stats.pf["btn_call_vs_3bet"][0] += 1
                        stats.pf["btn_vs_3bet"][1] += 1
                        self.opp_mu, self.opp_sigma = 0.58, 0.18
                    else:
                        self.opp_mu, self.opp_sigma = 0.78, 0.12
            elif action == "fold":
                if tc > 1:
                    lvl = self.opp_pf_raise_level()
                    if lvl == 0:
                        stats.pf["bb_fold_vs_open"][0] += 1
                        stats.pf["bb_fold_vs_open"][1] += 1
                    elif lvl == 1:
                        stats.pf["btn_fold_vs_3bet"][0] += 1
                        stats.pf["btn_vs_3bet"][1] += 1
                    elif lvl == 2:
                        stats.pf["bb_fold_vs_4bet"][0] += 1
                        stats.pf["bb_fold_vs_4bet"][1] += 1
            return
        # postflop
        tc = ann["to_call"]
        bucket = OppStats.size_bucket(ann["inc"] if ann["inc"] else 0, ann["pot_before"])
        if action in ("fold", "call", "raise") and tc > 0:
            stats.observe_facing(street, bucket, action)
        elif action == "raise":
            stats.observe_open(street, True)
        elif action == "check":
            stats.observe_open(street, False)
        # bump flop aggression model and persistent strength floor
        if street == "flop":
            if action == "raise":
                rr = (ann["amount"] / max(1.0, ann["to_call"])) \
                    if ann["to_call"] > 0 else 0.0
                if ann["inc"] >= 60 or (rr >= 2.4 and ann["to_call"] >= 8):
                    self.flop_agg += 0.55       # shove or fat re-raise: very strong
                    self.strength_floor = max(self.strength_floor,
                                              self.jam_floor(0.90))
                elif ann["f"] >= 0.9:
                    self.flop_agg += 0.40
                    self.strength_floor = max(self.strength_floor, 0.86)
                elif ann["f"] >= 0.4:
                    self.flop_agg += 0.42
                    self.strength_floor = max(self.strength_floor, 0.78)
                else:
                    self.flop_agg += 0.32
                    self.strength_floor = max(self.strength_floor, 0.74)
                self.flop_range_cache = None
            elif action == "call":
                self.flop_agg += 0.03
        elif street == "turn":
            self.turn_street_mu = self.street_mu_after(action, ann, "turn")
            if action == "raise":
                self.strength_floor = max(self.strength_floor,
                                          self.jam_floor(0.80))
                self.opp_barrels += 1
                if ann["inc"] >= 60:
                    self.turn_street_mu = (0.88, 0.09)
                    if len(self.stats.jam_flop_sd) >= 2 and \
                            sum(self.stats.jam_flop_sd) / len(self.stats.jam_flop_sd) < 0.62:
                        self.turn_street_mu = (0.74, 0.16)
            elif action == "check":
                self.opp_barrels = 0
            elif action == "call" and ann["f"] >= 0.5:
                self.strength_floor = max(self.strength_floor, 0.66)
        elif street == "river":
            self.turn_street_mu = self.street_mu_after(action, ann, "river")
            if action == "raise":
                self.strength_floor = max(self.strength_floor,
                                          self.jam_floor(0.82))
                self.opp_barrels += 1
                if ann["inc"] >= 60:
                    self.turn_street_mu = (0.88, 0.09)
                    if len(self.stats.jam_flop_sd) >= 2 and \
                            sum(self.stats.jam_flop_sd) / len(self.stats.jam_flop_sd) < 0.62:
                        self.turn_street_mu = (0.74, 0.16)
            elif action == "check":
                self.opp_barrels = 0

    def street_mu_after(self, action, ann, street):
        """(mu, sigma) for their current made strength given their street action."""
        f = ann["f"]
        if action == "raise":
            esc = 0.05 * max(0, getattr(self, "opp_barrels", 0) - 1)
            if f >= 0.9:
                base = 0.80 + 0.10 * clamp((f - 0.9) / 1.5, 0.0, 1.0) + esc
                sig = 0.10
            else:
                base = 0.72 + 0.05 * f + esc
                sig = 0.13
        elif action == "call":
            base = 0.50 + 0.10 * clamp(f, 0.0, 1.5)
            sig = 0.18
        elif action == "check":
            base, sig = 0.42, 0.24
        else:
            return None
        obs = self.stats.sd_avg_pct(street, base)
        mu = clamp((base + obs * 1.5) / 2.5, 0.25, 0.93)
        return mu, sig

    def opp_pf_first(self):
        # has the opp voluntarily acted preflop before (any raise/call)?
        return not any(e.get("actor") == "opp" and e.get("action") in ("raise", "call")
                       for e in self.last_hist if e.get("street") == "preflop")

    def opp_pf_raise_level(self):
        """How many voluntary preflop raises have happened before now (0=open)."""
        n = 0
        for e in self.last_hist:
            if e.get("street") != "preflop":
                break
            if e.get("action") == "raise":
                n += 1
        # BB counts as raise #1
        return n - 1

    def infer_flop_final(self, disc_req):
        """Infer opp's last flop action (usually a call) from pot deltas."""
        if self.last_req is None:
            return
        lr = self.last_req
        if lr.get("street") != "flop":
            return
        mb = lr.get("my_bet", 0)
        ob = lr.get("opp_bet", 0)
        pot = lr.get("pot", 0)
        # what did we reply? find our last flop action from history
        hist = self.last_hist
        my_last = None
        for e in hist:
            if e.get("street") == "flop" and e.get("actor") == "me":
                my_last = e
        if my_last is None:
            return
        a = my_last.get("action")
        pot_disc = disc_req.get("pot", 0)
        if a == "check":
            if pot_disc > pot:
                pass  # they bet?? impossible: we would have gotten a request
            return
        if a in ("call", "raise"):
            amt = my_last.get("amount") or 0
            if a == "call":
                after = pot  # our call closed the street; pot already includes both
                if pot_disc == after:
                    return
            else:
                after = pot + (amt - mb)
                if pot_disc == after + (amt - ob):
                    # they called our bet
                    self.flop_agg += 0.02
                    self.flop_range_cache = None
                    bucket = OppStats.size_bucket(amt - ob, max(1, after))
                    self.stats.observe_facing("flop", bucket, "call")
                    return

    # ------------------------------------------------------ equity plumbing
    def flop_range(self):
        if self.flop_range_cache is None:
            trip, w = build_flop_range(self.hole3, self.board[:3],
                                       self.opp_mu, self.opp_sigma,
                                       agg_bump=self.flop_agg)
            self.flop_range_cache = (trip, w)
        else:
            trip, w = self.flop_range_cache
        return trip, w

    def rebuild_flop_range(self):
        self.flop_range_cache = None
        return self.flop_range()

    def my_cards_now(self):
        if self.my_discard is not None and len(self.hole3) == 3:
            return [c for c in self.hole3 if c != self.my_discard]
        if len(self.hole3) == 2:
            return list(self.hole3)
        return list(self.hole3)

    def pair_range(self, street_mu, cond=True):
        my2 = self.my_cards_now()
        if isinstance(street_mu, (tuple, list)):
            mu, sig = street_mu[0], street_mu[1]
        else:
            mu, sig = street_mu, 0.26
        # after several streets the preflop prior is weak evidence: flatten it
        pre_sigma = max(0.30, self.opp_sigma)
        floor = getattr(self, "strength_floor", 0.0)
        if cond and floor > 0.0 and (mu is None or mu < floor):
            mu = floor
            sig = min(sig, 0.13) if mu == floor else sig
        if not cond:
            mu, sig = None, 0.26
        return build_pair_range(
            my2, self.board, self.opp_discard, self.opp_mu, pre_sigma,
            mu, sig, self.board[:3],
            sharpness=1.0 + 4.0 * self.discard_predictability(),
            flop_agg_bump=self.flop_agg)

    def discard_predictability(self):
        m, n = self.stats.disc_match
        if n < 3:
            return 0.5
        return clamp((m + 2.0) / (n + 4.0), 0.1, 0.95)

    def current_equity(self, req, budget_ms):
        street = req.get("street")
        deadline = time.perf_counter() + max(2.0, budget_ms / 1000.0)
        if street == "flop":
            trip, w = self.flop_range()
            n = 700 if budget_ms < 12 else (1200 if budget_ms < 25 else 1800)
            return equity_flop(self.hole3, self.board[:3], trip, w, n, self.rng)
        if street == "turn":
            smu = self.turn_street_mu
            pairs, w, unknown = self.pair_range(smu)
            if budget_ms >= 55:
                return equity_turn_exact(self.my_cards_now(), self.board, pairs, w,
                                         unknown, deadline=deadline)
            return equity_turn_mc(self.my_cards_now(), self.board, pairs, w, unknown, 1600)
        if street == "river":
            smu = self.turn_street_mu
            pairs, w, unknown = self.pair_range(smu)
            return equity_river(self.my_cards_now(), self.board, pairs, w)
        return 0.5

    # ------------------------------------------------------ response model
    def fold_prob(self, street, f, pot_before):
        stats = self.stats
        bucket = OppStats.size_bucket(f * pot_before, pot_before) if pot_before > 0 else "big"
        prior = clamp(f / (1.0 + f), 0.12, 0.80) * 0.92
        if street == "river" and f >= 0.8:
            prior *= 0.72          # people bluff-catch rivers more than MDF says
        persona = stats.persona()
        if persona == "station":
            prior *= 0.55
        elif persona == "maniac":
            prior *= 0.5
        elif persona == "nit":
            prior = min(0.9, prior * 1.30)
        p = stats.fold_rate(street, bucket, prior)
        return clamp(p, 0.02, 0.93)

    def reraise_prob(self, street, f):
        prior = 0.10 if f < 0.5 else 0.04
        persona = self.stats.persona()
        if persona == "maniac":
            prior = min(0.4, prior * 2.5)
        elif persona == "nit":
            prior *= 0.35
        bucket = "small" if f < 0.4 else ("mid" if f < 0.9 else "big")
        return clamp(self.stats.raise_rate(street, bucket, prior), 0.0, 0.45)

    def call_thresh_adj(self):
        persona = self.stats.persona()
        if persona == "station":
            return 0.045      # they value bet more, bluff less
        if persona == "maniac":
            return -0.060     # they bluff a lot
        if persona == "nit":
            return 0.030
        return 0.0

    # ------------------------------------------------------ EV machinery
    def ev_bet(self, eq, P, mb, ob, X, street, max_raise, pot_before):
        """EV of raising to X (a 'bet' when ob == 0)."""
        inc = X - mb
        f = inc / max(1.0, pot_before)
        F = self.fold_prob(street, f, pot_before)
        R = self.reraise_prob(street, f)
        if R + F > 0.97:
            R = 0.97 - F
        Cw = 1.0 - F - R
        ev_f = P
        pot_call = P + inc + (X - ob)
        ev_c = eq * pot_call - inc
        if R > 0.003 and X < max_raise - 1:
            R_to = min(max_raise, int(X + max(X - ob, 4) * 1.4 + 4))
            eq_r = eq * 0.9 - 0.02
            pot_rr = P + (R_to - mb) + (R_to - ob)
            ev_r = eq_r * pot_rr - (R_to - mb)
        else:
            ev_r = ev_c
        ev = F * ev_f + Cw * ev_c + R * ev_r
        return ev, F

    def ev_check(self, eq, P, street, in_position):
        rl = 1.0 if street == "river" else (0.93 if street == "turn" else 0.86)
        if in_position:
            rl += 0.03
        persona = self.stats.persona()
        if persona == "station":
            rl += 0.03   # we realize equity when they call down
        elif persona == "maniac" and not in_position:
            rl -= 0.06
        return eq * P * clamp(rl, 0.70, 1.10)

    def bet_candidates(self, req):
        P = req.get("pot", 0)
        mb = req.get("my_bet", 0)
        ob = req.get("opp_bet", 0)
        mn = req.get("min_raise", 2)
        mx = req.get("max_raise", mb + req.get("my_stack", 0))
        pot_before = P - (ob - mb) if ob > mb else P
        pot_before = max(1, pot_before)
        cands = []
        for fr in (0.35, 0.55, 0.8, 1.2):
            X = int(ob + fr * pot_before)
            cands.append(X)
        cands.append(mx)     # all-in
        out = []
        for X in cands:
            X = int(clamp(X, mn, mx))
            if not (mn <= X <= mx):
                continue
            if X <= ob and ob > 0:
                continue       # not actually a raise
            if X > mx * 0.62 and mx not in out:
                X = mx         # snap near-all-in to all-in
            if X not in out and X > ob:
                out.append(X)
        out.sort()
        return out, pot_before

    # ------------------------------------------------------ v2 range engine
    def self_pair_range(self):
        """What a rational opponent currently believes we hold (post-discard)."""
        if len(self.board) < 4 or self.my_discard is None:
            return None, None
        ln = self.self_ln
        try:
            return build_self_range(self.my_discard, self.board, self.board[:3], ln)
        except Exception:
            return None, None

    def build_view(self, street, deadline):
        """Exact range snapshot for turn/river decisions.
        Opponent range = preflop prior x discard evidence x board-relative
        action conditioning (their combos' equity vs our perceived range)."""
        my2 = [int(c) for c in self.my_cards_now()]
        board = [int(c) for c in self.board]
        hero_sc = int(eval7_scalar(my2 + board))
        pairs, w, unknown = self.pair_range(None, cond=False)
        opp_sc_full = range_scores(pairs, board)
        us_pairs, us_w = self.self_pair_range()
        if us_pairs is None or len(us_pairs) == 0:
            us_sc_full = opp_sc_full.copy()
            us_w_full = w.copy()
        else:
            us_sc_full = range_scores(us_pairs, board)
            us_w_full = us_w
        usc, usw = _cluster_scores(us_sc_full, us_w_full, 340)
        # their combos' equity vs our perceived range = board-relative strength
        opp_eq_full = eq_matrix(opp_sc_full, usc, usw)
        smu = self.turn_street_mu
        cw_full = w
        if smu is not None:
            if isinstance(smu, (tuple, list)):
                mu, sig = smu[0], smu[1]
            else:
                mu, sig = float(smu), 0.26
            floor = getattr(self, "strength_floor", 0.0)
            if floor > 0.0 and (mu is None or mu < floor):
                mu, sig = floor, min(sig, 0.13)
            mult = np.exp(-0.5 * ((opp_eq_full - mu) / max(0.03, sig)) ** 2)
            w2 = w * mult
            if float(w2.sum()) > 1e-9:
                cw_full = w2 / float(w2.sum())
        csc, cw, ceq = _cluster_kv(opp_sc_full, cw_full, opp_eq_full, 340)
        eq_hero = eq_vs_range(hero_sc, csc, cw)
        view = {"hero_sc": hero_sc, "csc": csc, "cw": cw, "usc": usc, "usw": usw,
                "opp_eq": ceq, "eq": eq_hero, "unknown": unknown,
                "pairs": pairs, "pw": cw_full, "my2": my2, "board": board}
        if street == "turn" and len(board) == 4:
            # river-aware hero equity: exact over a weighted sample of their range
            try:
                p = cw_full / max(1e-9, cw_full.sum())
                n_s = 150
                idx = self.rng.choice(len(pairs), size=n_s, p=p) if len(pairs) > 1 \
                    else np.zeros(n_s, dtype=np.int64)
                sp = np.asarray(pairs, dtype=np.int64)[idx]
                if time.perf_counter() < deadline - 0.008:
                    eq_t = equity_turn_exact(my2, board, sp, p[idx], unknown,
                                             deadline=deadline)
                else:
                    eq_t = equity_turn_mc(my2, board, sp, p[idx], unknown, 1200)
                view["eq"] = clamp(0.45 * eq_hero + 0.55 * eq_t, 0.01, 0.99)
            except Exception:
                pass
        # our own combos' equity vs their range (for equilibrium bluff frequency)
        try:
            us_eq = eq_matrix(usc, csc, cw)
            view["us_eq"] = us_eq
        except Exception:
            view["us_eq"] = np.full(len(usc), 0.5)
        return view

    def response_probs(self, street, X, mb, ob, P, view, persona, tighten=0.0):
        """Their fold / raise weights vs our bet (raise-to X), from exact ranges."""
        c = X - ob
        pot_final = P + (X - mb) + c
        pot_before = max(1.0, P - mb - ob)
        if c <= 0:
            return 0.0, 1.0, 1.0, 1.0
        alpha = c / max(1.0, pot_final)
        kappa = 1.0
        if persona == "station":
            kappa = 0.78
        elif persona == "maniac":
            kappa = 0.85
        elif persona == "nit":
            kappa = 1.22
        f = c / pot_before
        # overbets and raises are called only by hands that beat a polarized range
        theta_call = clamp(alpha * kappa + 0.30 * max(0.0, f - 0.9) + tighten, 0.02, 0.95)
        theta_raise = clamp(0.74 + 0.12 * min(2.0, f), 0.74, 0.95)
        if persona == "maniac":
            theta_raise -= 0.12
        elif persona == "nit":
            theta_raise += 0.08
        oeq = view["opp_eq"]
        cw = view["cw"]
        F_r = float(cw[oeq < theta_call].sum())
        R_r = float(cw[oeq > theta_raise].sum())
        bucket = OppStats.size_bucket(c, pot_before) if pot_before > 0 else "big"
        stats = self.stats
        d = stats.facing.get((street, bucket))
        n = d[3] if d else 0
        w_obs = min(1.0, n / 12.0)
        F_o = stats.fold_rate(street, bucket, max(0.02, F_r))
        R_o = stats.raise_rate(street, bucket, R_r)
        F = clamp((1.0 - w_obs) * F_r + w_obs * F_o, 0.01, 0.96)
        R = clamp((1.0 - w_obs) * R_r + w_obs * R_o, 0.0, 0.60)
        if F + R > 0.97:
            R = max(0.0, 0.97 - F)
        return F, R, theta_call, theta_raise

    def ev_bet_v2(self, street, X, P, mb, ob, mx_r, view, persona, tighten=0.0):
        """EV of betting/raising to X using the exact range response model."""
        inc = X - mb
        c = X - ob
        pot_final = P + inc + c
        F, R, theta_call, theta_raise = self.response_probs(
            street, X, mb, ob, P, view, persona, tighten=tighten)
        hero_sc = view["hero_sc"]
        csc = view["csc"]; cw = view["cw"]; oeq = view["opp_eq"]
        callers = (oeq >= theta_call) & (oeq <= theta_raise)
        eq_call = eq_vs_range(hero_sc, csc[callers], cw[callers])
        if street == "turn":
            eq_call = 0.55 * eq_call + 0.45 * view["eq"]
        ev_c = eq_call * pot_final - inc
        if R > 0.004 and X < mx_r - 1:
            R_to = min(mx_r, int(X + max(X - ob, 4) * 1.5 + 4))
            raisers = oeq > theta_raise
            eq_rr = eq_vs_range(hero_sc, csc[raisers], cw[raisers])
            pot_rr = P + (R_to - mb) + (R_to - ob)
            ev_r = max(-inc, eq_rr * pot_rr - (R_to - mb))
        else:
            ev_r = ev_c
            R = 0.0
        ev = F * P + (1.0 - F - R) * ev_c + R * ev_r
        return ev, F, R, eq_call

    def eq_bluff_freq(self, street, X, P, mb, ob, view, F):
        """Equilibrium-consistent bluff probability for a weak hand at size X."""
        c = X - ob
        pot_final = P + (X - mb) + c
        if c <= 0:
            return 0.0
        alpha = c / max(1.0, pot_final)
        us_eq = view.get("us_eq")
        usw = view.get("usw")
        if us_eq is None or usw is None or len(us_eq) == 0 or len(usw) == 0:
            return clamp(F * 0.5, 0.02, 0.8)
        value_w = float(usw[us_eq > alpha + 0.15].sum())
        bluff_w = float(usw[us_eq < 0.45].sum())
        if bluff_w <= 1e-9 or value_w <= 1e-9:
            return clamp(F * 0.6, 0.02, 0.85)
        p = alpha * value_w / ((1.0 - alpha) * bluff_w)
        # bounded exploitation: overfolders invite more bluffs, callers fewer
        if F > alpha + 0.02:
            p *= clamp(0.6 + 1.1 * (F - alpha), 1.0, 2.2)
        elif F < alpha - 0.05:
            p *= clamp(0.55 + (F / max(0.02, alpha)), 0.4, 1.0)
        return clamp(p, 0.01, 0.92)

    # ------------------------------------------------------ preflop chart
    def decide_preflop(self, req, budget_ms):
        legal = req.get("legal_actions", [])
        P = req.get("pot", 3)
        C = req.get("to_call", 1)
        mb = req.get("my_bet", 1 if self.seat_button else 2)
        ob = req.get("opp_bet", 2)
        mn = req.get("min_raise", 4)
        mx = req.get("max_raise", 200)
        raises_left = req.get("raises_left", 3)
        eq, pct = pf_lookup(self.hole3)
        cards = self.hole3
        persona = self.stats.persona()
        stats = self.stats
        can_raise = "raise" in legal
        lvl = self.opp_pf_raise_level()
        facing_allin = (req.get("opp_stack", 0) == 0) or (C >= 190)

        def reply(a, amt=None):
            if a == "raise":
                amt = int(clamp(amt, mn, mx))
                return {"type": "action", "action": "raise", "amount": amt}
            return {"type": "action", "action": a}

        # ---------- facing an all-in bet: pure equity decision ----------
        if facing_allin and C > 0:
            mu, jsig = self.stats.jam_range_mu(0.82 if self.opp_mu <= 0.55 else self.opp_mu,
                                               pf=True)
            need = C / max(1.0, P + C)
            eq_all = equity_preflop_range(self.hole3, mu, jsig, 900, self.rng)
            slack = -0.015 if len(self.stats.jam_pf_sd) < 2 else 0.010
            if eq_all > need + slack:
                return reply("call")
            return reply("fold")

        # ---------- we are the button (SB): first action ----------
        # HU with 3 cards: position + fold equity make ~90% opens correct
        if self.seat_button and lvl <= 0 and C <= 1:
            opp_fold = stats.pf_rate("bb_fold_vs_open", 0.30)
            opp_3b = stats.pf_rate("bb_3bet_vs_open", 0.10)
            open_thr = 0.10
            if opp_fold > 0.55:
                open_thr = 0.03
            elif opp_fold > 0.42:
                open_thr = 0.06
            elif opp_3b > 0.18:
                open_thr = 0.16
            if persona == "station":
                open_thr = min(0.28, open_thr + 0.06)
            if pct >= open_thr:
                roll = hand_seed(cards, "os")
                size = 4
                if roll < 0.16:
                    size = 5
                elif roll < 0.04:
                    size = 6
                if opp_fold > 0.5:
                    size = min(5, mx)
                if not can_raise:
                    return reply("call")
                return reply("raise", size)
            # limp only the near-open band; fold the true trash
            if pct >= max(0.02, open_thr - 0.07):
                return reply("call")
            return reply("fold")

        # ---------- button facing the big blind's limp-raise ----------
        if self.seat_button and lvl <= 0 and C > 1:
            if pct >= 0.86 and can_raise:
                return reply("raise", int(clamp(3.0 * ob, mn, mx)))
            need_l = C / max(1.0, P + C)
            if pct >= 0.30 or (need_l < 0.45 and pct >= 0.20):
                return reply("call")
            return reply("fold")

        # ---------- button facing a 3bet ----------
        if self.seat_button and lvl == 1 and C > 0:
            r3 = ob
            opp_3b = stats.pf_rate("bb_3bet_vs_open", 0.10)
            call_thr = 0.55 if r3 <= 12 else (0.62 if r3 <= 25 else 0.70)
            if opp_3b > 0.18:
                call_thr -= 0.06
            if stats.jam_count >= 3:
                call_thr -= 0.05      # they punish 4bets with jams: keep it small
            if r3 >= 100:
                mu = 0.86
                eq_all = equity_preflop_range(self.hole3, mu, 0.10, 900, self.rng)
                need = C / max(1.0, P + C)
                if eq_all > need - 0.012:
                    return reply("call")
                return reply("fold")
            if pct >= 0.93 and can_raise:
                roll4 = hand_seed(cards, "4b")
                if roll4 < 0.85:
                    return reply("raise", min(int(2.7 * r3), mx))
            if 0.70 <= pct < 0.93 and can_raise and hand_seed(cards, "4bl") < 0.12:
                return reply("raise", min(int(2.7 * r3), mx))
            if pct >= call_thr:
                return reply("call")
            return reply("fold")

        # ---------- button facing a 4bet ----------
        if self.seat_button and lvl >= 2 and C > 0:
            r4 = ob
            if r4 >= 90 or req.get("opp_stack", 0) == 0:
                mu = 0.93
                eq_all = equity_preflop_range(self.hole3, mu, 0.07, 900, self.rng)
                need = C / max(1.0, P + C)
                if eq_all > need - 0.01 and pct > 0.80:
                    return reply("call")
                return reply("fold")
            if pct >= 0.965 and can_raise:
                return reply("raise", mx)
            if pct >= 0.82 and can_raise and hand_seed(cards, "5bl") < 0.10:
                return reply("raise", mx)
            if pct >= (0.78 if r4 <= 26 else 0.88):
                return reply("call")
            return reply("fold")

        # ---------- we are the big blind ----------
        if not self.seat_button:
            # BB option after a limp (to_call == 0): iso-raise weak limp ranges
            if C == 0 and lvl <= 0:
                iso_size = 6 if mn <= 6 else mn
                if pct >= 0.72 and can_raise:
                    return reply("raise", iso_size)
                if pct >= 0.40 and can_raise and hand_seed(cards, "bx") < 0.30:
                    return reply("raise", iso_size)
                return reply("check")
            # BB facing an open (their single raise => lvl == 0)
            if lvl == 0 and C > 0:
                X = ob
                opp_open = stats.pf_rate("btn_open", 0.65)
                opp_fold = stats.pf_rate("btn_fold_vs_3bet", 0.55)
                three_thr = 0.86
                if opp_open > 0.85:
                    three_thr = 0.82
                elif opp_open < 0.45:
                    three_thr = 0.92
                if persona == "station":
                    three_thr = 0.90  # value 3bets only
                # vs jam-happy opponents keep preflop pots small: our edge is postflop
                if stats.jam_count >= 3:
                    three_thr = 0.955
                call_thr = clamp(0.30 + 0.010 * (X - 4), 0.26, 0.50)
                if stats.jam_count >= 3:
                    call_thr -= 0.05
                if opp_open > 0.85:
                    call_thr -= 0.03
                if opp_open < 0.45:
                    call_thr += 0.04
                if opp_fold > 0.62:
                    call_thr -= 0.04
                if pct >= three_thr and can_raise:
                    target = int(clamp(3.0 * X, mn, mx))
                    return reply("raise", target)
                if pct >= 0.52 and can_raise and hand_seed(cards, "3bl") < 0.12 \
                        and opp_fold > 0.45:
                    target = int(clamp(3.0 * X, mn, mx))
                    return reply("raise", target)
                if pct >= call_thr:
                    return reply("call")
                return reply("fold")
            # BB facing 4bet after our 3bet
            if lvl == 2 and C > 0:
                r4 = ob
                if r4 >= 120 or req.get("opp_stack", 0) == 0:
                    mu = 0.90
                    eq_all = equity_preflop_range(self.hole3, mu, 0.08, 900, self.rng)
                    need = C / max(1.0, P + C)
                    if eq_all > need - 0.01:
                        return reply("call")
                    return reply("fold")
                if pct >= 0.965 and can_raise:
                    return reply("raise", mx)
                if pct >= 0.84 and can_raise and hand_seed(cards, "5bb") < 0.10:
                    return reply("raise", mx)
                if pct >= (0.80 if r4 <= 30 else 0.90):
                    return reply("call")
                return reply("fold")
        # fallback: cheap call
        if C > 0:
            if pct >= 0.42:
                return reply("call")
            need = C / max(1.0, P + C)
            if need < 0.35:
                return reply("call")
            return reply("fold")
        return reply("check" if "check" in legal else "call")

    # ------------------------------------------------------ postflop logic
    def decide_postflop(self, req, budget_ms):
        street = req.get("street")
        if street in ("turn", "river") and not Clock.emergency(
                req.get("time_left_ms", 30000)):
            try:
                return self.decide_turn_river(req, budget_ms, street)
            except Exception:
                log_dbg("v2 engine fail " + traceback.format_exc()[-300:])
        return self.decide_postflop_v1(req, budget_ms)

    @staticmethod
    def board_danger(board):
        """0..1: how many straights/flushes/boats the completed board enables."""
        if len(board) < 4:
            return 0.0
        ranks = sorted((int(c) >> 2 for c in board), reverse=True)
        uniq = sorted(set(ranks), reverse=True)
        d = 0.0
        for i in range(len(uniq) - 3):
            if uniq[i] - uniq[i + 3] == 4:
                d = max(d, 0.7)
        if {2, 3, 4, 5}.issubset(set(ranks)) or {10, 11, 12, 13}.issubset(set(ranks)):
            d = max(d, 0.5)
        suits = [int(c) & 3 for c in board]
        for s in range(4):
            if suits.count(s) >= 4:
                d = max(d, 0.7)
        from collections import Counter as _C
        rc = _C(ranks)
        if rc.most_common(1)[0][1] >= 2:
            d = max(d, 0.35)
        return d

    def decide_turn_river(self, req, budget_ms, street):
        legal = req.get("legal_actions", [])
        P = req.get("pot", 0)
        mb = req.get("my_bet", 0)
        ob = req.get("opp_bet", 0)
        C = req.get("to_call", 0)
        my_stack = req.get("my_stack", 0)
        opp_stack = req.get("opp_stack", 0)
        in_pos = self.seat_button
        persona = self.stats.persona()
        mx_r = mb + my_stack
        deadline = time.perf_counter() + max(3.0, budget_ms / 1000.0)

        def reply(a, amt=None):
            if a == "raise":
                mn = req.get("min_raise", 2)
                mx = req.get("max_raise", mx_r)
                amt = int(clamp(amt, mn, mx))
                return {"type": "action", "action": "raise", "amount": amt}
            return {"type": "action", "action": a}

        # facing an all-in shove
        if opp_stack == 0 and C > 0:
            view = self.build_view(street, deadline)
            eq = clamp(view["eq"], 0.01, 0.99)
            need = C / max(1.0, P + C)
            adj = self.call_thresh_adj() + 0.025
            if street == "flop":
                adj += 0.035          # pre-discard equity is optimistic
            if self.opp_sigma < 0.15:
                adj += 0.07           # their 4bet+ line is monsters
            # wide jammers get called lighter (their jams are not all monsters)
            if len(self.stats.jam_flop_sd) >= 2 and C > 100 and \
                    sum(self.stats.jam_flop_sd) / len(self.stats.jam_flop_sd) < 0.62:
                adj -= 0.05
            if eq > need + adj - 0.008:
                return reply("call")
            return reply("fold")

        view = self.build_view(street, deadline)
        eq = clamp(view["eq"], 0.01, 0.99)
        pot_before = max(1.0, P - mb - ob)
        seed_cards = self.hole3 if len(self.hole3) == 3 else self.my_cards_now()

        # ---------------- facing a bet ----------------
        if C > 0:
            need = C / max(1.0, P + C)
            adj = self.call_thresh_adj()
            if persona == "station" and street == "river":
                adj += 0.02
            thr = need + adj
            if C > max(1.0, P - C):
                thr += 0.05          # big raises concentrate their range
            if street == "turn":
                thr += 0.01
            if street == "river" and C >= 0.6 * max(1.0, P - C) and \
                    getattr(self, "opp_barrels", 0) >= 2:
                thr += 0.055         # double-barrel+ big river: value-heavy line
            elif street == "river" and C >= 0.8 * max(1.0, P - C):
                thr += 0.03          # any near-pot river jam: concentrated range
            # discipline: never call a huge bet planning to fold later
            if C > 1.25 * max(1.0, P - C) and C > 0.45 * max(1.0, C + my_stack):
                thr += 0.09          # committing half our stack: need real equity
            if getattr(self, "opp_barrels", 0) >= 2 and street == "river" and C > 0:
                thr += 0.03
            # showdown calibration: are their aggressive lines value-heavy?
            sd_adj = self.showdown_adjust(street)
            thr += sd_adj
            # dominated second-best hands on scary boards
            bdanger = self.board_danger(self.board)
            if bdanger >= 0.5 and eq < 0.70:
                thr += 0.045
            margin = eq - thr
            p_call = clamp(0.5 + margin / 0.045, 0.0, 1.0)
            # MDF floor: bounded unexploitability vs their bets (never fold ~everything)
            f_bet = C / max(1.0, P - C)
            mdf = 1.0 / (1.0 + f_bet)
            kdef = {"normal": 0.40, "station": 0.30, "maniac": 0.72, "nit": 0.15}.get(persona, 0.40)
            if bdanger >= 0.5 and eq < 0.62:
                kdef *= 0.55           # bluff-catching on scary boards loses more
            if eq > need - 0.24:
                p_call = max(p_call, kdef * mdf)
            roll = hand_seed(seed_cards, "fc" + street)
            call_it = roll < p_call
            if need < 0.10 and eq > 0.22:
                call_it = True
            # value / semi-bluff raises with the exact response model
            best_raise, best_ev_r, best_F = None, None, None
            if "raise" in legal and time.perf_counter() < deadline - 0.002:
                cands, _pb = self.bet_candidates(req)
                for X in cands:
                    f = (X - ob) / max(1.0, pot_before)
                    if X >= mx_r - 1 and f > 1.3 and not (eq >= 0.90 or eq <= 0.12):
                        continue        # jam only with monsters or air
                    if f > 1.25 and not (eq >= 0.84 or eq <= 0.15):
                        continue        # no fat overbets with medium hands
                    ev, F, R, eq_call = self.ev_bet_v2(street, X, P, mb, ob, mx_r, view,
                                                        persona, tighten=0.15)
                    if f > 1.25 and 0.15 <= eq < 0.84 and eq_call < 0.72:
                        continue        # overbetting into a caller range that beats us
                    if best_ev_r is None or ev > best_ev_r:
                        best_ev_r, best_raise, best_F = ev, X, F
            ev_call = eq * (P + C) - C
            if best_raise is not None and best_ev_r > ev_call + \
                    (0.20 if eq >= 0.70 else 0.32) * (P + C) and eq > 0.70:
                return reply("raise", best_raise)
            # river bluff-raise vs chronic overbetting weak ranges (rare, bounded)
            if best_raise is not None and street == "river" and eq < 0.30 and \
                    best_F is not None and best_F > 0.62 and \
                    self.stats.persona() != "station":
                p_br = clamp((best_F - 0.55) * 0.6, 0.0, 0.18)
                if hand_seed(seed_cards, "br" + street) < p_br:
                    return reply("raise", best_raise)
            if call_it:
                return reply("call")
            return reply("fold")

        # ---------------- we can check or bet ----------------
        # discipline: after committing vs a big flop re-raise, no turn donk-barrels
        # with less-than-nut equity (their range is board-hit heavy)
        if street == "turn" and getattr(self, "paid_big_rr_flop", False):
            if eq < 0.82:
                return reply("check")
            self.paid_big_rr_flop = True   # keep flag for river too
        else:
            self.paid_big_rr_flop = False
        cands, pot_before = self.bet_candidates(req)
        # polarized sizing: with monsters or pure air add big overbets
        if street == "river" and (eq >= 0.86 or eq <= 0.14):
            for fr in (1.8,):
                X = int(ob + fr * pot_before)
                if X not in cands:
                    cands.append(X)
            cands.sort()
        ev_check = self.ev_check(eq, P, street, in_pos)
        best_a, best_ev, best_X, best_F = "check", ev_check, None, None
        if time.perf_counter() < deadline - 0.002:
            for X in cands:
                f = (X - ob) / max(1.0, pot_before)
                if X >= mx_r - 1 and f > 1.3 and not (eq >= 0.90 or eq <= 0.12):
                    continue          # jam only with monsters or air
                if f > 1.25 and not (eq >= 0.84 or eq <= 0.15):
                    continue          # no fat overbets with medium hands
                if f > 0.9 and not (eq >= 0.75 or eq <= 0.15):
                    X = int(ob + 0.8 * pot_before)   # cap at 0.8 pot
                if not (X > ob) or X == best_X:
                    continue
                ev, F, R, eq_call = self.ev_bet_v2(street, X, P, mb, ob, mx_r, view, persona)
                if f > 1.25 and 0.15 <= eq < 0.84 and eq_call < 0.72:
                    continue          # overbetting into a caller range that beats us
                # giant overbets: only with real value, or proven fold equity
                if f > 2.2 and X >= mx_r - 1:
                    fb = self.fold_vs_big(street)
                    if eq_call < 0.75 and not (eq <= 0.15 and fb is not None and fb >= 0.5):
                        continue
                if ev > best_ev + 0.5:
                    best_ev, best_a, best_X, best_F = ev, "raise", X, F
        # equilibrium bluff frequency gating (range-computed)
        if best_a == "raise" and eq < 0.40:
            p_bluff = self.eq_bluff_freq(street, best_X, P, mb, ob, view,
                                         best_F if best_F is not None else 0.4)
            roll = hand_seed(seed_cards, "bl" + street)
            if roll > p_bluff:
                best_a, best_X = "check", None
        # thin value nudge vs stations on the river
        if best_a == "check" and street == "river" and eq > 0.62 and persona == "station":
            best_a, best_X = "raise", int(ob + 0.75 * pot_before)
        if best_a == "check":
            return reply("check")
        return reply("raise", best_X)

    def showdown_adjust(self, street):
        """Shift call thresholds from showdown-revealed strengths of their bets."""
        d = getattr(self.stats, "bet_sd", None)
        if not d or len(d) < 4:
            return 0.0
        vals = list(d)
        mean = sum(vals) / len(vals)
        # mean shown percentile of their bet lines: low => bluff-heavy => call more
        # clamp widened: value-heavy rivers must be respected faster (payoff leak)
        return clamp((mean - 0.52) * 0.16, -0.06, 0.08)

    def decide_postflop_v1(self, req, budget_ms):
        legal = req.get("legal_actions", [])
        street = req.get("street")
        P = req.get("pot", 0)
        mb = req.get("my_bet", 0)
        ob = req.get("opp_bet", 0)
        C = req.get("to_call", 0)
        my_stack = req.get("my_stack", 0)
        opp_stack = req.get("opp_stack", 0)
        eq = self.current_equity(req, budget_ms)
        eq = clamp(eq, 0.01, 0.99)
        in_pos = self.seat_button     # button acts last postflop
        persona = self.stats.persona()

        def reply(a, amt=None):
            if a == "raise":
                mn = req.get("min_raise", 2)
                mx = req.get("max_raise", mb + my_stack)
                amt = int(clamp(amt, mn, mx))
                return {"type": "action", "action": "raise", "amount": amt}
            return {"type": "action", "action": a}

        # facing an all-in shove
        if opp_stack == 0 and C > 0:
            need = C / max(1.0, P + C)
            adj = self.call_thresh_adj() + 0.025
            if street == "flop":
                # jams = hands that hit THIS flop hard (keep-score top slice)
                frac = 0.28
                if len(self.stats.jam_flop_sd) >= 2:
                    jm = sum(self.stats.jam_flop_sd) / len(self.stats.jam_flop_sd)
                    if jm >= 0.80:
                        frac = 0.16      # they jam only real monsters
                    elif jm < 0.62:
                        frac = 0.42      # wide jammers
                try:
                    trip, wj = build_jam_flop_range(self.hole3, self.board[:3], frac=frac)
                    eqj = equity_flop(self.hole3, self.board[:3], trip, wj,
                                      900, self.rng)
                except Exception:
                    eqj = eq
                if eqj > need + adj + 0.015:
                    return reply("call")
                return reply("fold")
            if eq > need + adj - 0.012:
                return reply("call")
            return reply("fold")

        # ---------- facing a bet ----------
        if C > 0:
            need = C / max(1.0, P + C)
            adj = self.call_thresh_adj()
            if persona == "station" and street == "river":
                adj += 0.02
            thr = need + adj
            if C > max(1.0, P - C):
                thr += 0.05      # respect big raises: their range is concentrated
            if street == "flop":
                thr += 0.02      # imperfect realization OOP / future barrels
            elif street == "turn":
                thr += 0.01
            pot_before = max(1.0, P - C)
            if street == "river" and C >= 0.7 * pot_before and \
                    getattr(self, "opp_barrels", 0) >= 3:
                thr += 0.05      # triple-barrel pot-sized river: value-heavy
            if street == "flop" and self.opp_sigma < 0.15 and C > 0.5 * pot_before:
                thr += 0.07              # their 4bet+ preflop line is monster-heavy
            # mixing near the threshold (MDF aggregate discipline)
            margin = eq - thr
            p_call = clamp(0.5 + margin / 0.045, 0.0, 1.0)
            # monster flop raises: call only hands strong enough to commit
            # trigger: huge raise vs pot, OR a raise-to >= 2.4x our own bet
            # (re-raise over our lead: in this game that range is trips-heavy)
            big_rr = C > 1.5 * pot_before or (
                mb >= 4 and ob >= 2.4 * mb + 2 and C >= 0.5 * pot_before)
            if street == "flop" and big_rr:
                pv = self.stats.persona()
                commit_eq = 0.66 if pv == "normal" else (0.55 if pv == "maniac" else 0.72)
                if self.opp_sigma < 0.15:
                    commit_eq += 0.08
                # adaptive: their observed big-raise showdowns were weak -> lighter
                if len(self.stats.jam_flop_sd) >= 3 and \
                        sum(self.stats.jam_flop_sd) / len(self.stats.jam_flop_sd) < 0.62:
                    commit_eq -= 0.08
                # evaluate vs the "hit this flop hard" range, not the wide one
                try:
                    trip, wj = build_jam_flop_range(self.hole3, self.board[:3], frac=0.30)
                    eqj = equity_flop(self.hole3, self.board[:3], trip, wj,
                                      700, self.rng)
                except Exception:
                    eqj = eq
                if eqj < commit_eq:
                    p_call = 0.0
                else:
                    p_call = max(p_call, 0.7)
                    self.paid_big_rr_flop = True   # only set when we DO commit
            roll = hand_seed(self.hole3 if len(self.hole3) == 3 else self.my_cards_now(),
                            "fc" + street)
            call_it = roll < p_call
            # tiny bets: almost never fold
            if need < 0.10 and eq > 0.22:
                call_it = True
            # evaluate raises (semi-bluff or value)
            best_raise, best_ev_r = None, None
            if "raise" in legal:
                cands, pot_before = self.bet_candidates(req)
                mx_r = mb + my_stack
                for X in cands:
                    f = (X - ob) / max(1.0, pot_before)
                    eq_c = eq - 0.05 * min(1.0, f) - 0.10 * min(1.2, max(0.0, f - 1.0))
                    if X >= mx_r - 1 and f > 1.3 and not (eq >= 0.90 or eq <= 0.12):
                        continue          # jam only with monsters or air
                    if f > 1.25 and not (eq >= 0.84 or eq <= 0.15):
                        continue          # no fat overbets with medium hands
                    if f > 0.9 and not (eq >= 0.75 or eq <= 0.15):
                        X = int(ob + 0.8 * pot_before)   # cap at 0.8 pot
                    ev, F = self.ev_bet(eq_c, P, mb, ob, X, street, mx_r, pot_before)
                    if best_ev_r is None or ev > best_ev_r:
                        best_ev_r, best_raise = ev, X
            ev_call = eq * (P + C) - C
            ev_fold = 0.0
            if best_raise is not None and best_ev_r > ev_call + \
                    (0.15 if eq >= 0.70 else 0.30) * (P + C) and \
                    eq > 0.58 and street in ("turn", "river"):
                return reply("raise", best_raise)
            if call_it:
                return reply("call")
            return reply("fold")

        # ---------- we can check or bet ----------
        # solver-style IP c-bet frequencies on the flop: uncap our range
        if street == "flop" and in_pos and C == 0 and ob == 0 and P >= 5:
            roll = hand_seed(self.hole3, "cb")
            trap = eq >= 0.78 and hand_seed(self.hole3, "tp") < 0.22
            if not trap:
                if eq >= 0.54:
                    p_bet = 1.0
                elif eq >= 0.38:
                    p_bet = 0.72
                elif eq >= 0.25:
                    p_bet = 0.34
                else:
                    p_bet = 0.12
                stats_f = self.stats.fold_rate("flop", "mid", 0.40)
                p_bet = clamp(p_bet + 0.9 * (stats_f - 0.40), 0.05, 1.0)
                if persona == "nit":
                    p_bet += 0.08
                if roll < p_bet:
                    fr = 0.55 if eq >= 0.60 else 0.48
                    X = int(ob + fr * P)
                    mn = req.get("min_raise", 2)
                    mx = req.get("max_raise", mb + my_stack)
                    X = int(clamp(X, mn, mx))
                    if X > ob:
                        return reply("raise", X)
            return reply("check")
        cands, pot_before = self.bet_candidates(req)
        ev_check = self.ev_check(eq, P, street, in_pos)
        best_a, best_ev, best_X = "check", ev_check, None
        mx_r = mb + my_stack
        for X in cands:
            f = (X - ob) / max(1.0, pot_before)
            eq_c = eq - 0.05 * min(1.0, f) - 0.10 * min(1.2, max(0.0, f - 1.0))
            if X >= mx_r - 1 and f > 1.3 and not (eq >= 0.90 or eq <= 0.12):
                continue          # jam only with monsters or air
            if f > 1.25 and not (eq >= 0.84 or eq <= 0.15):
                continue          # no fat overbets with medium hands
            if f > 0.9 and not (eq >= 0.75 or eq <= 0.15):
                X = int(ob + 0.8 * pot_before)   # cap at 0.8 pot
            # giant flop overbets need proven fold equity or real value
            if f > 2.2 and X >= mx_r - 1 and street == "flop":
                fb = self.fold_vs_big("flop")
                if not (eq >= 0.75 or (eq <= 0.15 and fb is not None and fb >= 0.5)):
                    continue
            ev, F = self.ev_bet(eq_c, P, mb, ob, X, street, mx_r, pot_before)
            if ev > best_ev + 0.5:
                best_ev, best_a, best_X = ev, "raise", X
        # bluff gating: don't barrel hopeless hands too often
        if best_a == "raise" and eq < 0.34:
            Xf = (best_X - ob) / max(1.0, pot_before)
            F = self.fold_prob(street, Xf, pot_before)
            mdf_req = Xf / (1.0 + Xf)
            if F < mdf_req + 0.04:
                roll = hand_seed(self.hole3 if len(self.hole3) == 3 else self.my_cards_now(),
                                 "bl" + street)
                if roll > 0.30:
                    best_a, best_X = "check", None
        # thin value nudge vs stations on the river
        if best_a == "check" and street == "river" and eq > 0.62 and persona == "station":
            best_a, best_X = "raise", int(ob + 0.75 * pot_before)
        if best_a == "check":
            return reply("check")
        return reply("raise", best_X)

    # ------------------------------------------------------ discard choice
    def choose_discard(self, req):
        hole = [CARD_ID.get(c, 0) for c in req.get("hole_cards", [])]
        flop = [CARD_ID.get(c, 0) for c in req.get("board", [])]
        budget_ms = self.clock.budget_ms(req.get("time_left_ms", 30000), self.hand_id, 0.6)
        if len(hole) != 3 or len(flop) != 3:
            # degenerate fallback: lowest card
            worst = min(hole, key=lambda c: (c >> 2, c & 3))
            return {"type": "discard", "card": CARD_TEXT[worst]}
        emergency = Clock.emergency(req.get("time_left_ms", 30000))
        if emergency:
            i = discard_prediction(hole, flop)
            return {"type": "discard", "card": CARD_TEXT[hole[i]]}
        try:
            trip, w = self.flop_range()
            n = 1000 if budget_ms < 12 else (1400 if budget_ms < 22 else 1800)
            eqs = discard_option_equities(hole, flop, trip, w, n, self.rng)
        except Exception:
            log_dbg("discard eq fail " + traceback.format_exc()[-400:])
            i = discard_prediction(hole, flop)
            return {"type": "discard", "card": CARD_TEXT[hole[i]]}
        best = max(range(3), key=lambda i: eqs[i])
        near = [i for i in range(3) if eqs[i] >= eqs[best] - 0.020]
        if len(near) > 1:
            roll = hand_seed(hole, "dj")
            pick = near[int(roll * len(near)) % len(near)]
        else:
            pick = best
        return {"type": "discard", "card": CARD_TEXT[hole[pick]]}

    # ------------------------------------------------------ message handling
    def on_init(self, msg):
        self.clock.update(msg.get("time_bank_ms", 30000), msg.get("total_hands", 200))
        self.total_hands = msg.get("total_hands", 200)
        self.stats = OppStats()
        self.rng = np.random.default_rng(0x5EED5)

    def on_new_hand(self, msg):
        self.hand_id = msg.get("hand_id", 0)
        self.hole3 = [CARD_ID.get(c, 0) for c in msg.get("hole_cards", [])]
        self.seat_button = msg.get("seat") == "button"
        self.board = []
        self.my_discard = None
        self.opp_discard = None
        self.seen_hist = 0
        self.last_hist = []
        self.replay_cache = []
        self.opp_mu, self.opp_sigma = 0.50, 0.30
        self.flop_agg = 0.0
        self.my_pf_aggr = False
        self.last_req = None
        self.turn_street_mu = None
        self.flop_range_cache = None
        self.opp_barrels = 0
        self.strength_floor = 0.0
        self.paid_big_rr_flop = False
        self.self_ln = LineModel()
        self.self_range_cache = None
        self.stats.hands_seen += 1

    def on_discard_result(self, msg):
        d = msg.get("opp_discard")
        if d:
            self.opp_discard = CARD_ID.get(d, None)
        mine = msg.get("my_discard")
        if mine:
            self.my_discard = CARD_ID.get(mine, None)
        self.hole3 = [CARD_ID.get(c, 0) for c in msg.get("hole_cards", self.hole3)]

    def on_hand_result(self, msg):
        # showdown calibration
        res = msg.get("result")
        opp_hole = msg.get("opp_hole_cards")
        if res == "showdown" and opp_hole and self.opp_discard is not None:
            triple = [CARD_ID.get(c, 0) for c in opp_hole] + [self.opp_discard]
            eq, pct = pf_lookup(triple)
            # last aggressive street by opp
            last_aggro = "none"
            opp_bet_any = False
            for e in (msg.get("history") or []):
                if e.get("actor") == "opp" and e.get("action") == "raise":
                    last_aggro = e.get("street", "none")
                    opp_bet_any = True
            won = (msg.get("net", 0) or 0) > 0
            self.stats.observe_showdown(last_aggro, pct, won)
            if opp_bet_any:
                # shown strength of their betting lines -> bluff-catching calibration
                self.stats.bet_sd.append(pct)
                if len(self.stats.bet_sd) > 40:
                    self.stats.bet_sd = self.stats.bet_sd[-40:]
            # adaptive jam model: what do their 120+ raises actually show down as?
            jam_street = None
            for e in (msg.get("history") or []):
                if e.get("actor") == "opp" and e.get("action") == "raise":
                    amt = e.get("amount") or 0
                    if amt >= 120:
                        jam_street = e.get("street", "flop")
            if jam_street is not None:
                self.stats.jam_count += 1
                # final made-hand percentile on the board (not preflop strength)
                try:
                    ob5 = [CARD_ID.get(c, 0) for c in (msg.get("board") or [])]
                    o2 = [CARD_ID.get(c, 0) for c in (opp_hole or [])]
                    if len(ob5) == 5 and len(o2) == 2:
                        fp = hand_pct(int(eval7_scalar(o2 + ob5)))
                        if jam_street == "preflop":
                            self.stats.jam_pf_sd.append(fp)
                            if len(self.stats.jam_pf_sd) > 25:
                                self.stats.jam_pf_sd = self.stats.jam_pf_sd[-25:]
                        else:
                            self.stats.jam_flop_sd.append(fp)
                            if len(self.stats.jam_flop_sd) > 25:
                                self.stats.jam_flop_sd = self.stats.jam_flop_sd[-25:]
                except Exception:
                    pass
            # discard predictability
            flop3 = self.board[:3]
            if len(flop3) == 3:
                pred = discard_prediction(triple, flop3)
                actual = self.opp_discard
                self.stats.disc_match[1] += 1
                if triple[pred] == actual:
                    self.stats.disc_match[0] += 1
        if res == "fold":
            pass

    def on_action_request(self, msg):
        self.last_req = msg
        street = msg.get("street")
        if street and len(self.board) < len(msg.get("board", [])):
            self.board = [CARD_ID.get(c, 0) for c in msg.get("board", [])]
        if street == "preflop":
            self.board = []
        self.update_from_history(msg)
        if street == "flop" and self.flop_range_cache is not None:
            self.rebuild_flop_range()
        budget_ms = self.clock.budget_ms(msg.get("time_left_ms", 30000),
                                          self.hand_id,
                                          1.0 if street in ("turn", "river") else 0.7)
        if Clock.emergency(msg.get("time_left_ms", 30000)):
            budget_ms = 3.0
        legal = msg.get("legal_actions", [])
        try:
            if street == "preflop":
                return self.decide_preflop(msg, budget_ms)
            return self.decide_postflop(msg, budget_ms)
        except Exception:
            log_dbg("decide fail: " + traceback.format_exc()[-600:])
            if "check" in legal:
                return {"type": "action", "action": "check"}
            if "call" in legal:
                return {"type": "action", "action": "call"}
            return {"type": "action", "action": "fold"}


# ------------------------------------------------------------------- main
def validate_action(reply, req):
    legal = req.get("legal_actions", [])
    a = reply.get("action")
    if a not in legal:
        if "check" in legal:
            return {"type": "action", "action": "check"}
        if "call" in legal:
            return {"type": "action", "action": "call"}
        return {"type": "action", "action": "fold"}
    if a == "raise":
        mn = req.get("min_raise")
        mx = req.get("max_raise")
        amt = reply.get("amount")
        if not isinstance(amt, int) or mn is None or mx is None or not (mn <= amt <= mx):
            # clamp into legality instead of sending an illegal raise
            amt2 = int(clamp(amt if isinstance(amt, (int, float)) else mn, mn, mx))
            return {"type": "action", "action": "raise", "amount": amt2}
    return reply


def validate_discard(reply, req):
    card = reply.get("card")
    hole = req.get("hole_cards", [])
    if card not in hole:
        worst = min(hole, key=lambda c: (RANKS.index(c[0]), "shdc".index(c[1])))
        return {"type": "discard", "card": worst}
    return reply


def fallback_action(req):
    legal = req.get("legal_actions", [])
    if "check" in legal:
        return {"type": "action", "action": "check"}
    if "call" in legal:
        return {"type": "action", "action": "call"}
    return {"type": "action", "action": "fold"}


def main():
    bot = AtlasBot()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except Exception:
            continue
        t = msg.get("type")
        try:
            if t == "init":
                bot.on_init(msg)
                continue
            if t == "new_hand":
                bot.on_new_hand(msg)
                continue
            if t == "discard_result":
                bot.on_discard_result(msg)
                continue
            if t == "hand_result":
                bot.on_hand_result(msg)
                continue
            if t == "notice":
                continue
            if t == "match_end":
                break
            if t == "action_request":
                try:
                    reply = bot.on_action_request(msg)
                except Exception:
                    log_dbg("action fail " + traceback.format_exc()[-400:])
                    reply = fallback_action(msg)
                reply = validate_action(reply, msg)
                _PROTO_OUT.write(json.dumps(reply, separators=(",", ":")) + "\n")
                _PROTO_OUT.flush()
                continue
            if t == "discard_request":
                try:
                    bot.infer_flop_final(msg)
                    reply = bot.choose_discard(msg)
                except Exception:
                    log_dbg("discard fail " + traceback.format_exc()[-400:])
                    worst = min(msg.get("hole_cards", []),
                                key=lambda c: (RANKS.index(c[0]), "shdc".index(c[1])))
                    reply = {"type": "discard", "card": worst}
                reply = validate_discard(reply, msg)
                _PROTO_OUT.write(json.dumps(reply, separators=(",", ":")) + "\n")
                _PROTO_OUT.flush()
                continue
            # unknown message type: ignore
        except Exception:
            log_dbg("top fail " + traceback.format_exc()[-400:])
            try:
                _PROTO_OUT.write(json.dumps(fallback_action(msg)) + "\n")
                _PROTO_OUT.flush()
            except Exception:
                pass
    try:
        _PROTO_OUT.flush()
    except Exception:
        pass


if __name__ == "__main__":
    main()
