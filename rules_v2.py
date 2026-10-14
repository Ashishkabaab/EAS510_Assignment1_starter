"""EAS 510 - Project 1 - Phase 2: the V2 rule set.

Rules 1-3 are kept exactly as in `rules.py`. Your job is to ADD Rule 4 that
targets the systematic weakness you diagnosed from results_v1_hard.txt.

Constraint: the rules' `out_of` weights must still sum to 100 so the Final
Score stays on a /100 scale (the format validator enforces "Final Score:
<s>/100" equals the sum of the block's rule scores).

Example split that keeps the original spirit: Rules 1-3 -> 25/25/40 and
Rule 4 -> 10 (or trim the losers more aggressively).
"""

import functools
from functools import lru_cache

import numpy as np
import cv2
import rules

#Weights must sum to 100
W1, W2, W3, W4 = 25, 30, 10, 35
assert W1 + W2 + W3 + W4 == 100, "rule weights must sum to 100"

#Rule 4 Parameters
R4_MAX_DIM = 256        #this size: fast, and smooths compression noise
R4_SCALES = np.geomspace(1.0, 0.30, 18)   # relative template sizes to try
R4_MIN_TEMPLATE = 40    # px; smaller templates match anything by accident
R4_FLOOR = 0.80         # correlation at/below this counts as no evidence
R4_FIRE = 0.10          # rescaled metric needed to fire

#: Import the original rules and append your new one.
RULES = ("rule1_metadata_v2", "rule2_histogram_v2", "rule3_template_v2",
         "rule4_multiscale")

def _rescaled(fn, new_out_of):
    """Same rule, same logic: only the points budget changes."""
    @functools.wraps(fn)
    def wrapped(target, input_path):
        r = dict(fn(target, input_path))
        r["out_of"] = new_out_of
        r["score"] = int(round(new_out_of * r["metric"])) if r["fired"] else 0
        return r
    return wrapped

rule1_metadata_v2 = _rescaled(rules.rule1_metadata, W1)
rule2_histogram_v2 = _rescaled(rules.rule2_histogram, W2)
rule3_template_v2 = _rescaled(rules.rule3_template, W3)

@lru_cache(maxsize=256)
def _small_gray(path):
  '''Grayscale copy with long side = R4_MAX_DIM'''
  wh = rules._size(path)
  if not wh:
    return None
  longest = max(wh)
  if longest >= 4 * R4_MAX_DIM:
      flag = cv2.IMREAD_REDUCED_GRAYSCALE_4
  elif longest >= 2 * R4_MAX_DIM:
      flag = cv2.IMREAD_REDUCED_GRAYSCALE_2
  else:
      flag = cv2.IMREAD_GRAYSCALE
  g = cv2.imread(path, flag)
  if g is None:
      return None
  h, w = g.shape
  s = R4_MAX_DIM / max(h, w)
  return cv2.resize(g, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)


def rule4_multiscale(target, input_path):
    """TODO: replace with your Phase 2 rule that fixes the V1 weakness.

    Starter returns a no-op so the pipeline still runs before you implement it.
    Parity with rules.py rule dict: rule/name/fired/score/out_of/note/metric.

    Multi-scale template matching. 
    Both images first normalized to same long side, so pure resize lines up at scale ~ 1 and crop shows up as template smaller
    than original. Slide suspect shrunk by each factor in R4_SCALES over original and keep best correlation. 
    """
    out = {"rule": 4, "name": "Multiscale", "fired": False, "score": 0,
           "out_of": W4, "note": "Match score 0.00", "metric": 0.0}
    try:
      src = _small_gray(target["path"])
      sus = _small_gray(input_path)
      if src is None or sus is None:
          return out
      sh, sw = src.shape
      nh, nw = sus.shape
      fit = min(sw / nw, sh / nh)        # largest scale that still fits inside
      best, best_s = 0.0, 1.0
      for f in R4_SCALES:
          w, h = int(round(nw * fit * f)), int(round(nh * fit * f))
          if min(w, h) < R4_MIN_TEMPLATE or w > sw or h > sh:
              continue
          t = cv2.resize(sus, (w, h), interpolation=cv2.INTER_AREA)
          if t.std() < 5:                # flat patch: correlation is meaningless
              continue
          ncc = float(np.nan_to_num(cv2.matchTemplate(
              src, t, cv2.TM_CCOEFF_NORMED).max()))
          if ncc > best:
              best, best_s = ncc, f
      metric = max(0.0, (best - R4_FLOOR) / (1.0 - R4_FLOOR))
      out["metric"] = round(min(1.0, metric), 3)
      out["note"] = f"Match score {out['metric']:.2f} at scale {best_s:.2f}"
      if out["metric"] >= R4_FIRE:
          out["fired"] = True
          out["score"] = int(round(W4 * out["metric"]))
    except Exception:
        pass
    return out