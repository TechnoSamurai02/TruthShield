# Image decision evaluation summary

Evaluation date: 2026-07-14

These results measure the packaged `truthshield-image-detector-v2` ResNet classifier and the conservative decision policy used by the application. The classifier's `ai_generated` softmax output is an AI-class score, not a calibrated real-world probability.

## Decision policy

- `AI-class score <= 0.15`: likely authentic, unless independent evidence conflicts.
- `AI-class score >= 0.95`: likely AI-generated or manipulated, unless camera/provenance evidence conflicts without forensic corroboration.
- Scores between the thresholds: inconclusive, with a narrow authentic exception when a score no higher than `0.30` is corroborated by camera metadata or verified provenance.
- Missing models, invalid scores, failed web/provenance services, and unavailable optional tools are neutral and lead to abstention when the remaining evidence is insufficient.

## Held-out score evaluation

Source: `decision_report_test_1800.json`. This balanced, leakage-filtered test sample contains 600 AI-generated, 600 real-camera, and 600 real-edited/captioned images.

| Metric | Former frontend rule (`>= 0.70` means AI) | Three-way policy (`<= 0.15` / `>= 0.95`) |
| --- | ---: | ---: |
| False AI alarms among 1,200 non-AI images | 62 (5.17%) | 7 (0.58%) |
| Correct AI decisions among 600 AI images | 498 (83.00%) | 253 (42.17%) |
| AI images incorrectly called authentic | Not represented by the forced rule | 14 (2.33%) |
| Inconclusive | 0 | 664 (36.89%) |
| Decisive coverage | 100% | 63.11% |
| Accuracy among decisive outcomes | Not applicable to a one-sided UI rule | 98.15% |
| Precision of a likely-AI decision | 88.93% | 97.31% |

The stricter threshold materially lowers false accusations, but it also lowers AI recall. The new behavior deliberately exposes that tradeoff as an inconclusive outcome.

## Full production-path evaluation

Source: `full_pipeline_report_test_300.json`. This seeded sample contains 100 images per class and runs byte decoding, metadata inspection, pixel/compression forensics, the classifier, neutral web/provenance fallbacks, aggregation, and response-schema validation.

- Outcomes: 53 likely AI, 154 likely authentic, and 93 inconclusive.
- Non-AI false-positive rate: 1/200 = 0.50%.
- AI incorrectly labeled authentic: 5/100 = 5.00%.
- AI recall at the likely-AI threshold: 52.00%.
- Authentic recall across both non-AI classes: 74.50%.
- Inconclusive rate: 31.00%; decisive coverage: 69.00%.
- Decisive accuracy: 97.10%.
- Likely-AI precision: 98.11%; likely-authentic precision: 96.75%.
- In this environment, web search was not configured for all 300 samples and C2PA tooling was unavailable for all 300; both remained neutral.

## Transformation robustness sample

Sources: `real_transform_robustness.json` and `ai_transform_robustness.json`. Five real and five AI source images were each tested as the original plus JPEG 95/75/50, 75% and 50% resizes, an 85% center crop, and a social-style resize/JPEG transform.

- Real variants: 7 likely authentic, 33 inconclusive, 0 likely AI.
- AI variants: 24 likely AI, 16 inconclusive, 0 likely authentic.
- The sample is small and selected, so it is a regression/robustness check rather than a population estimate.

## Scope and limitations

- The test export groups diverse scenes but does not preserve a separate scene-category label, so per-people/animal/landscape/building rates are not available.
- Generator-family labels are not preserved per image in the prepared test export; the evaluation cannot claim separate performance for each generator.
- The model remains capable of confidently misclassifying both real and AI images. Neither outcome is proof.
- The three-way thresholds are an operating policy selected from held-out behavior, not a formal probability calibration guarantee.
- Face swaps, local retouching, and adversarial edits were not separately benchmarked here.
- Reverse-image context depends on external configuration. A missing key or failed service contributes no evidence.
- C2PA presence can support provenance, but absence cannot prove manipulation or generation.
