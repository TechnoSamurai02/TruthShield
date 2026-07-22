# TruthShield image v4 pilot status

## Decision

The July 21, 2026 image-fusion pilot is **not promoted** to production. The
locked Midjourney-6 suite is consumed and must not be used to tune another
candidate.

## Locked result

- 500 records: 250 authentic and 250 generated.
- Generated-verdict precision: 97.83%.
- Authentic false-AI rate: 0.8%.
- Generated recall: 36%.
- Overall decisive coverage: 18.4%; the promotion requirement is 70%.
- Two records were forced inconclusive because controlled-view instability
  exceeded 0.18.

The candidate met its precision and false-warning requirements but failed the
decisive-coverage requirement. Manipulation and authentic outcomes also remain
disabled because no manipulation specialist passed its own promotion gate.

## Component diagnosis

| Component | Locked ROC-AUC | Locked average precision |
| --- | ---: | ---: |
| Community Forensics | 0.898368 | 0.901886 |
| TruthShield comparison model | 0.482688 | 0.498690 |
| Regularized fusion | 0.876480 | 0.884889 |

The comparison model failed to generalize to the unseen generator and reduced
the stronger Community Forensics signal when fused. The next candidate should
not use this comparison model as an authority. It requires broader development
and calibration populations, a manipulation specialist that meets its gates,
and a newly constructed locked generator-family suite.

## Reproducibility

- Community Forensics model: `OwensLab/commfor-model-224`
- Reviewed upstream commit: `ee5b71d43db0f3779e1edd64ee927b13f2dd6ad4`
- Decision policy: `truthshield-media-policy-v4.0.0`
- Fusion fit split: tuning
- Calibration family: DALL-E 3
- Consumed locked family: Midjourney 6

Only aggregate reports are committed. Per-file locked predictions and media are
kept out of the repository to reduce the chance of accidental failure-driven
tuning.

## Manipulation localizer v3 full-data result

The expanded LR-ASPP localizer was trained on 2,440 validated training records
and evaluated on 750 editor-isolated tuning records. Controlled-view evaluation
produced ROC-AUC 0.814688 and average precision 0.704737. Its best constrained
rule achieved 95.24% precision, 0.4% authentic false warnings, no generated
false-manipulation warnings, and 8% manipulation recall.

This is an improvement over the v2 pilot's 1.6% safe recall, but remains too low
for production promotion. The candidate is retained as a reproducible experiment
and is not calibrated, locked-tested, packaged, or deployed.
