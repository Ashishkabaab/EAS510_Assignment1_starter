# EAS 510 - Assignment 1: Digital Forensics Apprentice

A rule-based expert system that matches modified images back to their originals.

This is the **starter repository** for Project 1. Two repositories matter:

| Repository | Role |
|------------|------|
| `delveccj/EAS510_Assignment1_starter` (this repo) | Your **code** lives here. You will fork it, work in your fork, and push your final submission here. |
| `delveccj/EAS510_Assignment1` | The **dataset** (read-only). Clone it for the images; never commit it to your fork. |

## Setup on your Codio box

```bash
# 1. Fork this repo on GitHub, then in your box:
git clone https://github.com/YOUR-USERNAME/EAS510_Assignment1_starter.git
cd EAS510_Assignment1_starter

# 2. Clone the read-only dataset as a sibling folder (images live there):
cd ~/workspace
git clone https://github.com/delveccj/EAS510_Assignment1.git

# 3. Install dependencies:
cd ~/workspace/EAS510_Assignment1_starter
python3 -m pip install -r requirements.txt

# 4. Point this box at YOUR fork (run once):
./setup_git.sh
```

## Repository layout

```
forensics_detective.py   SimpleDetector: register targets + find_best_match
rules.py                 Rule functions (Rule 1: Metadata, Rule 2: Histogram, Rule 3: Template)
test_system.py           Runs the detector over the data folders and writes results_*.txt
scripts/check_output_format.py   Validates a results file against the required format
setup_git.sh, submit.sh  One-time setup + commit/push helper ("backup button")
```

## The task in one paragraph

Implement three interpretable rules (metadata, color histogram, template matching)
that combine into a 0-100 confidence score. Run your system on `modified_images/`
and `random/` and save the full output as `results_v1.txt`. In Phase 2, run on
`hard/`, diagnose a systematic failure, add a **Rule 4**, and save
`results_v1_hard.txt` and `results_v2.txt`. Full instructions are in the Codio guide
and in the assignment PDF in UBLearns.

## Running

```bash
# Phase 1 (easy + random) -> results_v1.txt
python3 test_system.py --modified --random --output results_v1.txt

# Phase 1 on hard cases -> results_v1_hard.txt
python3 test_system.py --hard --output results_v1_hard.txt

# Phase 2 (everything) -> results_v2.txt
python3 test_system.py --modified --hard --random --output results_v2.txt

# Validate any results file against the required format:
python3 scripts/check_output_format.py results_v1.txt
```

The data folders are resolved from a sibling clone of `EAS510_Assignment1`.
If your data is elsewhere, pass `--data-dir /path/to/EAS510_Assignment1`.

## Output format (do not alter)

```
Processing: modified_image_01.jpg
Rule 1 (Metadata): FIRED - Size ratio 0.85 -> 20/30 points
Rule 2 (Histogram): FIRED - Correlation 0.92 -> 25/30 points
Rule 3 (Template): FIRED - Match score 0.76 -> 30/40 points
Final Score: 75/100 -> MATCH to original_03.jpg
```

## Backup doctrine

Your Codio box can be reset at any time. **Your fork on GitHub is the only safe
copy.** After every milestone run `./submit.sh "describe what you did"`. A box
restart never touches GitHub.

## License

Apache 2.0. See `LICENSE`.

## Observed weakness in V1: what failed and why
There are 20 images in v3 and v6. These two variants produced 9 rejections and 1 wrong match. All 20 images from these two variants had a score of 0 from 
Rule 3. Some examples:
-original_00__crop_keep60__resized__q45__v6.jpg
-original_08__crop_keep60__resized__q45__v6.jpg
-original_00__resize_scale114__compress__q45__v3.jpg
This is because Rule 3 lines things up at one scale, and both v3 and v6 involve resizing. Positions don't line up between the suspect and the original. 
The scale is not preserved so Rule 3 gives a score of 0. 
## Design decision for V2: what Rule 4 is and why you chose it
Rule 4 is about normalizing the same long side so a pure resize lines up. It starts by loading a small grayscale of each image. Then we find the biggest
scale that fits so that the template/suspect is no bigger than the original it slides over. The suspect is tried at many sizes. We slide and score, with
normalized cross correlation ignoring brightness and contrast. The best result across all sizes is stored. NCC gets rescaled into a good metric and the 
rule fires accordingly. This was chosen because cv2.matchTemplate in rule 3 only works if the suspect and the original are at the same pixel scale. So we
try many sizes to find the right scale. 
## Effect of the change: accuracy before/after on easy vs hard
On Easy, the rules identified the 60 matches using V1 and only 52 matches using V2, so a slight decrease in accuracy. 
On Hard, the rules identified 50 correct matches using V1 and 52 correct matches using V2, so a marginal increase. 
## Trade-offs: what new costs or risks did Rule 4 introduce
After Rule 4 was introduced, v6 was 10/10 correct, which was giving problems before. resize_scale114 was also fixed, and wrong matches were down from 0 to 1.
However, results got worse with crop_25pct and crop_50pct. With crop_50pct, one image was now rejected after Rule 4, and crop_25pct went from 10/10 correct to 
3/10 correct. These were carried by Rule 3 in V1 but it's weight was reduced in V2 and Rule 4 scores 0 on most of these. 