# MedVision Data Card

## Dataset role

MedVision uses a small chest X-ray dataset for research and educational experimentation. The repository treats the dataset as an input to an ML pipeline rather than embedding a claim of clinical representativeness.

## Data quality audit

The recorded quality audit contains:

| Quality check | Count |
|---|---:|
| Total images | 454 |
| Valid images | 430 |
| Corrupted images | 6 |
| Duplicates | 12 |
| Blank images | 3 |
| Invalid channel images | 3 |
| Missing labels | 0 |
| Extreme brightness cases | 0 |
| Invalid dimensions | 0 |

The audit is intentionally surfaced because data quality can materially affect medical-imaging evaluation.

## Labels

The model predicts eight thoracic findings:

- Atelectasis
- Cardiomegaly
- Effusion
- Infiltration
- Mass
- Nodule
- Pneumonia
- Pneumothorax

The task is **multi-label**, so one image may contain multiple positive findings.

## Split strategy

The repository uses patient-level splitting rather than image-level random splitting.

Target split fractions:

- 70% train
- 15% validation
- 15% test
- Seed: 42

The split implementation asserts that train, validation, and test patient IDs are pairwise disjoint.

This is a critical medical-imaging safeguard because multiple images from the same patient can otherwise cross the split boundary and inflate evaluation performance.

## Preprocessing and augmentation

The current training configuration uses:

- Model-specific input resolution.
- Standard image preprocessing associated with the selected pretrained backbone.
- Horizontal flip.
- Small rotation augmentation.
- Brightness/contrast augmentation.

Augmentation should be reviewed carefully for clinical plausibility before any clinical application.

## Class imbalance

The project explicitly compares:

- Binary cross-entropy.
- Weighted binary cross-entropy.
- Focal loss.

The strongest F1/sensitivity configuration in the current benchmark is Weighted BCE, but it comes with a substantial specificity trade-off.

## Data governance and privacy

The repository's prediction database stores metadata such as image hashes, model version, findings, probability, latency, and timestamps rather than retaining uploaded patient images as prediction records.

The project should still be treated as handling potentially sensitive medical data when deployed. Real-world use requires appropriate consent, access controls, retention policies, encryption, and regulatory review.

## Known limitations

- Small experimental sample size.
- No demonstrated external validation.
- Unknown population representativeness.
- Potential dataset and label noise.
- Class prevalence may differ substantially from real deployment environments.
- Performance should not be interpreted as clinical sensitivity/specificity without a properly designed validation study.

## Reproducibility checklist

- [x] Data quality checks are implemented.
- [x] Patient-level splitting is implemented.
- [x] Split seed is explicit.
- [x] Model and loss configurations are version controlled.
- [x] Evaluation results are stored in `reports/`.
- [x] Automated tests cover split integrity.
- [ ] Independent external validation.
- [ ] Full data provenance/version manifest.
