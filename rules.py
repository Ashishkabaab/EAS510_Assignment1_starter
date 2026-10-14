"""EAS 510 - Project 1: Rule functions for the Digital Forensics Apprentice.

Each rule compares one suspect image against one registered original and
returns a dict:

    {
        "rule": 1,
        "name": "Metadata",
        "fired": bool,          # True if the rule found supporting evidence
        "score": int,           # points awarded out of "out_of"
        "out_of": int,
        "note": "Size ratio 0.85",   # short human-readable metric
        "metric": 0.85,              # raw similarity measure in [0, 1]
    }

The points budgets are fixed by the assignment: Rule 1 = 30, Rule 2 = 30,
Rule 3 = 40. The detector sums them into a 0-100 confidence score.

These starter implementations are deliberately WEAK baselines (they hinge on
simple thresholds). Improving them is the assignment.

Reads are cached per path so one suspect against many originals does not
re-decode images needlessly.
"""

import os
from functools import lru_cache

import numpy as np
import cv2
from PIL import Image

#: Rule names used by SimpleDetector.evaluate() (V1).
RULES = ("rule1_metadata", "rule2_histogram", "rule3_template")

# Knobs to be able to tune as constants
R1_FIRE = 0.5          # rule 1 fires at/above this metric
R2_FLOOR = 0.65        # histogram correlation at/below this = no evidence
R2_FIRE = 0.2          # rule 2 fires at/above this (rescaled) metric
R3_FLOOR = 0.5         # template match at/below this = no evidence
R3_FIRE = 0.1          # rule 3 fires at/above this (rescaled) metric
MIN_INLIERS = 15       # feature matches needed to trust an ORB match


@lru_cache(maxsize=256)
def _arr(path):
    return cv2.imread(path)


@lru_cache(maxsize=256)
def _hist(path):
    img = _arr(path)
    if img is None:
        return None
    hists = [cv2.calcHist([img], [i], None, [32], [0, 256]) for i in range(3)]
    hist_all = np.concatenate(hists).ravel().astype(np.float32)
    cv2.normalize(hist_all, hist_all)
    return hist_all


def _downscale(img, max_dim=512):
    """Fit an image into `max_dim` before expensive cv2 work."""
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return img


@lru_cache(maxsize=256)
def _gray(path):
    img = _arr(path)
    if img is None:
        return None
    return _downscale(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))


@lru_cache(maxsize=256)
def _size(path):
    try:
        with Image.open(path) as img:
            return img.size
    except Exception:
        return None

# function to Examine first 16 bytes to determine file type
def _magic(filepath):
  with open(filepath, 'rb') as f:
      header_bytes = f.read(16)
  if header_bytes.startswith(b'\x89PNG'):
      return "png"
  if header_bytes.startswith(b'\xFF\xD8'):
      return "jpeg"
  return "other"

#Full resolution grayscale used by rule 3 so both images share a scale
@lru_cache(maxsize=256)
def _gray_full(path):
    img = _arr(path)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Oriented FAST and Rotated BRIEF keypoints to handle rotation + scale. Cached like rest
@lru_cache(maxsize=256)
def _orb(path):
  g = _gray(path) #load as grayscale
  if g is None:
    return None
  #create detector and keep 1500 strongest keypoints
  #kp = list of keypoints
  #des = Numpy array with one fingerprint per row
  kp, des = cv2.ORB_create(nfeatures=1500).detectAndCompute(g, None)
  #discard images with no features
  if des is None or len(kp) < 10:
    return None
  #return each keypoint (x,y) position as an array and the fingerprints
  return np.float32([k.pt for k in kp]), des

def rule1_metadata(target, input_path):
    """changed to be header + dimensions instead of raw file-size ratio."""
    out = {"rule": 1, "name": "Metadata", "fired": False, "score": 0,
           "out_of": 30, "note": "Size ratio 0.00", "metric": 0.0}
    try:
        sw, sh = _size(target["path"]) or (0, 0)
        iw, ih = _size(input_path) or (0, 0)

        if (iw, ih) == (sw, sh):
            # same dimensions
            if _magic(target["path"]) != _magic(input_path):
                metric = 1.0 # JPEG / PNG format change
            else:
              a = os.path.getsize(target["path"])
              b = os.path.getsize(input_path)
              metric = 0.7 + 0.3 * min(a, b) / max(a, b) #recompression
        elif (iw, ih) == (sh, sw):
          metric = 0.8 #rotated 90 degrees
        elif sw and sh and ih and abs((iw/ih) / (sw / sh) -1 ) < 0.02:
          metric = 0.85 #same aspect ratio -> resized, with 2% tolerance
        elif iw <= sw and ih <= sh and sw * sh > 0:
          metric = 0.4 + 0.4 * (iw * ih) / (sw * sh) #crop
        else:
          metric = 0.2
        out["metric"] = round(max(0.0, min(1.0, metric)), 3)
        out["note"] = f"Size ratio {out['metric']:.2f}"
        if out["metric"] >= R1_FIRE:
            out["fired"] = True
            out["score"] = int(round(out["out_of"] * out["metric"]))
    except Exception:
        pass
    return out


def rule2_histogram(target, input_path):
    """Color histogram correlation. Robust to crops and spatial changes.
    Changed scoring. Unrelated images correlate ~0.5-0.8, so
    rescale [R2_FLOOR, 1] -> [0, 1] instead of using the raw correlation."""
    out = {"rule": 2, "name": "Histogram", "fired": False, "score": 0,
           "out_of": 30, "note": "Correlation 0.00", "metric": 0.0}
    try:
        hs, hi = _hist(target["path"]), _hist(input_path)
        if hs is None or hi is None:
            return out
        corr = float(cv2.compareHist(hs, hi, cv2.HISTCMP_CORREL))
        raw = max(0.0, min(1.0, corr))
        metric = max(0.0, (raw - R2_FLOOR) / (1.0 - R2_FLOOR))
        out["metric"] = round(min(1.0, metric), 3)
        out["note"] = f"Correlation {raw:.2f}"
        if out["metric"] >= R2_FIRE:
            out["fired"] = True
            out["score"] = int(round(out["out_of"] * out["metric"]))
    except Exception:
        pass
    return out


def rule3_template(target, input_path):
    """Template matching. Detects when the suspect is contained in the original.

    cv2.matchTemplate needs the template (suspect) smaller than the target
    (original); on rotated/resized suspects this rule degrades fast. Fixing
    that becomes Rule 4 territory.

    Change: both images shrunk by same factor instead of independently. score also
    rescaled so weak matches count as 0. 
    """
    out = {"rule": 3, "name": "Template", "fired": False, "score": 0,
           "out_of": 40, "note": "Match score 0.00", "metric": 0.0}
    try:
        src = _gray_full(target["path"])
        nd = _gray_full(input_path)
        if src is None or nd is None:
            return out

        #One shared scale taken from original 
        s = min(1.0, 512 / max(src.shape))
        if s < 1.0:
            src = cv2.resize(src, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            nd = cv2.resize(nd, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)

        # template must fit inside the original: shrink uniformly if it doesn't
        fit = min(src.shape[1] / nd.shape[1], src.shape[0] / nd.shape[0])
        if fit < 1.0:
            nd = cv2.resize(nd, None, fx=fit, fy=fit, interpolation=cv2.INTER_AREA)

        res = cv2.matchTemplate(src, nd, cv2.TM_CCOEFF_NORMED)
        best = float(np.nan_to_num(res.max()))
        metric = max(0.0, (best - R3_FLOOR) / (1.0 - R3_FLOOR))
        out["metric"] = round(min(1.0, metric), 3)
        out["note"] = f"Match score {out['metric']:.2f}"
        if out["metric"] >= R3_FIRE:
            out["fired"] = True
            out["score"] = int(round(out["out_of"] * out["metric"]))
    except Exception:
        pass
    return out