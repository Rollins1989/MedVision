# 🩻 MedVision

An end-to-end explainable medical imaging AI system for multi-label chest
X-ray analysis, featuring transfer-learning-ready architectures, uncertainty
estimation, Grad-CAM explanations with a faithfulness sanity check, REST
inference, experiment tracking, containerized deployment, and automated
testing.

> ⚠️ **This system is a research/educational decision-support prototype and
> is not intended for clinical diagnosis.**

**Live demo:** _add your deployed URL here after following "Deploying" below_
**Repo:** _add your GitHub link here_

---

## Read this first: what's real here, and what's a sandbox stand-in

This was built end-to-end in a sandboxed environment with no internet
access to Kaggle, the NIH data host, or `download.pytorch.org` (ImageNet
pretrained weights). Rather than hand you a repo full of unrun code and
made-up numbers, here's exactly what's real and what's substituted:

| | Real | Sandbox substitution |
|---|---|---|
| **Pipeline code** (data, models, training, eval, explainability, uncertainty, API, tests) | Yes -- 100% real, every module executes | -- |
| **Dataset** | -- | A small **synthetic** procedurally-generated stand-in (454 images / 226 patients), not real X-rays. NIH ChestX-ray14/CheXpert require an account-gated download (Box/Kaggle) unreachable from this sandbox. See "Using the real dataset" below for exact steps to point this same code at the real data. |
| **All experiment numbers below** (ROC-AUC, PR-AUC, F1, ECE, faithfulness margins) | Yes -- computed for real, from real training runs on the synthetic data | Numbers reflect the *synthetic* data's learnability, not clinical performance -- don't quote these as "my model detects pneumonia at 93% AUC" without the caveat |
| **Transfer learning** | -- | Trained with `pretrained=False` (random init) because ImageNet weights live on `download.pytorch.org`, unreachable here. The code path for `pretrained=True` is fully implemented and works normally on any machine with internet access -- just flip the flag in the config. |
| **Docker** | Dockerfile + docker-compose written and YAML-validated | Not run end-to-end here (no Docker daemon in this sandbox) -- run `docker compose up --build` yourself before trusting it blind |
| **CI/CD** | Real GitHub Actions workflow, will run correctly on GitHub's runners (which have internet access) | Not run here for the same reason |
| **Deployment** | -- | Not deployed from this sandbox; see "Deploying" for the steps |

Every number quoted in this README came from an actual `python3 -m
src.training.train --config ...` run against the synthetic dataset in this
repo.

---

## 1. What It Does

```
Image
 |
 v
Validation & quality gating
 |
 v
Preprocessing
 |
 v
Multi-label CNN (DenseNet121 / ResNet50 / EfficientNet-B0 / ViT)
 |
 v
Prediction + MC-Dropout uncertainty + temperature-scaled calibration
 |
 v
Grad-CAM explanation (with a faithfulness sanity check)
 |
 v
JSON report, persisted to PostgreSQL
```

Example response from `POST /predict`:

```json
{
  "prediction_id": 1,
  "top_finding": "Pneumonia",
  "top_probability": 0.917,
  "findings": [
    {"label": "Pneumonia", "probability": 0.917, "uncertainty": 0.038,
     "uncertainty_tier": "Low", "calibrated_probability": 0.896},
    {"label": "Infiltration", "probability": 0.62, "uncertainty": 0.09,
     "uncertainty_tier": "Medium", "calibrated_probability": 0.58}
  ],
  "model_name": "densenet121",
  "model_version": "1.0.0",
  "latency_ms": 143.2,
  "explanation": {"method": "Grad-CAM", "available": true, "target_label": "Pneumonia"},
  "disclaimer": "This system is a research/educational decision-support prototype and is not intended for clinical diagnosis."
}
```

## 2. Architecture

```
                         +----------------------+
                         |     Web Frontend      |
                         | (static HTML/JS,      |
                         |  served by FastAPI)   |
                         +-----------+----------+
                                     |
                                     v
                         +----------------------+
                         |       FastAPI         |
                         | REST API + Swagger    |
                         +-----------+----------+
                                     |
                  +------------------+------------------+
                  v                  v                  v
           Image Validation    Preprocessing        Logging
                  |                  |
                  +---------+--------+
                            v
                   +-------------------+
                   |  PyTorch Model     |
                   | DenseNet121/ResNet |
                   +---------+---------+
                             |
               +-------------+--------------+
               v             v              v
           Prediction     Grad-CAM      Uncertainty
               |         (+faithfulness)  (MC Dropout +
               |             |             calibration)
               +-------------+--------------+
                             v
                      Result Service
                             |
                      +------+------+
                      v             v
                PostgreSQL     Prometheus/Grafana
               (predictions      (see "What's Next")
                 metadata)
```

## 3. Project Structure

```
medvision/
  app/
    main.py              FastAPI app: /health /model-info /predict /explain /prediction/{id}
    database.py            SQLAlchemy engine + PredictionRecord model
    schemas/                 Pydantic request/response models
    services/
      inference.py              validation, prediction, uncertainty, Grad-CAM orchestration
  src/
    data/
      synthetic_generator.py      sandbox stand-in dataset (see caveat above)
      quality_checks.py             corrupted/duplicate/blank/bad-dim detection
      prepare.py                      runs quality checks, writes cleaned labels CSV
      split.py                          PATIENT-LEVEL train/val/test split
      dataset.py                          PyTorch Dataset
    preprocessing/transforms.py           resize/normalize/augment
    models/factory.py                       ResNet50 / DenseNet121 / EfficientNet-B0 / ViT
    training/
      losses.py                              BCE / Weighted BCE / Focal Loss
      train.py                                 config-driven training loop + MLflow + checkpointing
    evaluation/
      metrics.py                                per-label ROC-AUC/PR-AUC/F1/sensitivity/specificity
      build_experiment_table.py                   aggregates run metrics into the comparison table
    explainability/
      gradcam.py                                    Grad-CAM (hooks-based, no third-party lib)
      faithfulness.py                                 perturbation-based explanation sanity check
      run_faithfulness_eval.py                          batch faithfulness evaluation script
    uncertainty/
      mc_dropout.py                                       Monte Carlo Dropout
      calibration.py                                        temperature scaling + ECE
      run_calibration_eval.py                                 batch calibration evaluation script
  configs/                   baseline.yaml, densenet.yaml, efficientnet.yaml, vit.yaml
  frontend/index.html          drag-and-drop single-page UI (served by FastAPI at "/")
  tests/                          23 pytest tests: API + pipeline
  reports/                          generated: data quality, experiment table, calibration, faithfulness
  models/                             trained checkpoints + per-run metrics JSON
  .github/workflows/ci.yml             lint -> test -> build Docker image
  Dockerfile
  docker-compose.yml
  requirements.txt
```

## 4. Setup & How to Run

### Quick start (no Docker, CPU)

> **Note:** this delivered copy already includes the generated demo
> dataset (`data/`), the three DenseNet121 checkpoints referenced in the
> experiment tables below (`models/*.pt`), and MLflow's metrics database
> (`mlflow.db`) so you can run the API immediately without retraining.
> The ResNet50/EfficientNet/ViT checkpoints and MLflow's artifact store
> (`mlruns/`) were trimmed from this delivered copy to keep it a
> reasonable size -- their full metrics are preserved in
> `models/*_metrics.json` and `reports/`. All of this is `.gitignore`d for
> actual repo use anyway (large binaries don't belong in git); `make demo`
> (or `make experiments` for all four architectures) regenerates
> everything from scratch in a few minutes.

```bash
pip install -r requirements.txt
pip install torch torchvision   # separately -- see requirements.txt comment for why

# 1. Generate the demo dataset + run quality checks (~10s)
python3 src/data/synthetic_generator.py
python3 -m src.data.prepare

# 2. Train the best-performing config from our experiments (~3 min on CPU)
python3 -m src.training.train --config configs/densenet.yaml

# 3. Serve the API (also serves the frontend at the same URL)
uvicorn app.main:app --reload
# open http://localhost:8000        (frontend)
# open http://localhost:8000/docs   (Swagger UI)
```

### With Docker

```bash
docker compose up --build
```

Starts Postgres + the API together. The API reads `DATABASE_URL` from the
compose environment automatically and mounts `./models` read-only, so
retraining a new checkpoint on the host doesn't require a rebuild.

### Running the tests

```bash
pytest -v
```

23 tests, all green: valid/corrupted/blank image handling, prediction
schema validation (probabilities in `[0,1]`, uncertainty tiers valid),
Grad-CAM generation (default + explicit label), prediction persistence and
retrieval, 404 handling, model loading, patient-level split leakage
checks, all-4-architectures forward pass, and loss function correctness
(focal loss assigns lower loss to easy examples, by construction).

## 5. Dataset

In this sandbox: a procedurally-generated synthetic stand-in (454 images,
226 patients, 8 multi-label findings: Atelectasis, Cardiomegaly, Effusion,
Infiltration, Mass, Nodule, Pneumonia, Pneumothorax). Not real medical
imaging data -- see `src/data/synthetic_generator.py` for exactly how it's
generated and why.

### Using the real dataset

To point this same codebase at NIH ChestX-ray14 or CheXpert:

1. Download NIH ChestX-ray14 (Box, requires free account) or CheXpert
   (Stanford ML Group, requires registration).
2. Produce a `labels.csv` with columns `image_id, patient_id,
   <one column per finding as 0/1>` -- the NIH release's own
   `Data_Entry_2017.csv` needs a small pivot (their labels are
   pipe-separated in one column; this repo expects one column per label,
   matching `src/data/synthetic_generator.LABELS`).
3. Put images in a flat directory, point `configs/*.yaml`'s
   `data.images_dir` / `data.labels_csv` at your paths.
4. Everything downstream -- quality checks, patient-level split, training,
   evaluation, Grad-CAM, calibration -- works unchanged.

## 6. Dataset Quality Report

Real output from `python3 -m src.data.prepare`, run against the (synthetic)
dataset in this repo -- including the corrupted files, duplicates, and
blank images the generator deliberately injects to prove the checker
catches them:

```
Dataset Quality Report
=======================
Total images:        454
Valid images:         430
Corrupted:               6
Duplicates:              12
Blank images:             3
Extreme brightness:       0
Bad dimensions:           0
Bad channel count:        3
Missing labels:           0
```

("Duplicates" uses perceptual hashing, not just byte-identical files, so it
catches near-duplicates too -- the count is a little higher than the 10
duplicates the generator explicitly injects, because our synthetic base
template also happens to produce a few genuinely near-identical images.)
The cleaned, filtered dataset (`data/processed/labels_clean.csv`) is what
training actually consumes.

## 7. Patient-Level Split

`src/data/split.py` splits by `patient_id`, not by image -- the #1 data
leakage mistake in medical imaging. If a patient has 3 X-rays, all 3 stay
in exactly one of train/val/test. Verified with a hard assertion (also a
pytest test):

```
Train: 311 images / 158 patients
Val:    71 images / 33 patients
Test:   72 images / 35 patients
Patient overlap train/val: 0, train/test: 0, val/test: 0
```

## 8. Experiment 1 -- Architecture Comparison

All four trained with plain BCE loss, `pretrained=False` (see the
transfer-learning caveat at the top), 6 epochs, early stopping patience 3,
on the same patient-level split:

| Model | Loss | ROC-AUC | PR-AUC | F1 | Sensitivity | Specificity | Latency (ms) |
|---|---|---|---|---|---|---|---|
| DenseNet121 | bce | 0.9380 | 0.8287 | 0.3687 | 0.3085 | 0.9972 | 30.2 |
| ResNet50 | bce | 0.8455 | 0.6561 | 0.2396 | 0.1954 | 0.9958 | 31.1 |
| ViT-B/32 | bce | 0.7946 | 0.4274 | 0.0938 | 0.1071 | 0.9933 | 107.2 |
| EfficientNet-B0 | bce | 0.7914 | 0.4876 | 0.1292 | 0.1032 | 1.0000 | 9.7 |

DenseNet121 wins on ROC-AUC and PR-AUC and was carried forward as the
architecture for the loss-function and calibration experiments below.
EfficientNet-B0 is by far the fastest (9.7ms) -- a reasonable choice if
latency mattered more than ceiling accuracy. ViT's weaker showing here is
expected: vision transformers are known to need either much more data or
pretraining to beat CNN inductive biases, and this run has neither (a few
hundred synthetic images, from-scratch init) -- not a claim that ViT is
categorically worse for chest X-rays.

## 9. Experiment 2 -- Loss Function Comparison (class imbalance)

Same architecture (DenseNet121), same split, only the loss changes:

| Loss | ROC-AUC | PR-AUC | F1 | Sensitivity | Specificity |
|---|---|---|---|---|---|
| BCE (baseline) | 0.9380 | 0.8287 | 0.3687 | 0.3085 | 0.9972 |
| Focal (alpha=0.25, gamma=2.0) | 0.9394 | 0.8086 | 0.2928 | 0.2103 | 0.9956 |
| Weighted BCE | 0.9354 | 0.7689 | 0.6656 | 0.9765 | 0.8080 |

This is the real, legitimate ML discussion the loss experiment is meant to
produce: all three land within ~0.004 ROC-AUC of each other (ranking
ability is barely affected), but F1 nearly doubles under Weighted BCE
(0.37 to 0.67), driven by sensitivity jumping from 0.31 to 0.98 at the cost
of specificity dropping to 0.81. Weighted BCE's `pos_weight` term directly
penalizes missed positives per-class, so at the default 0.5 threshold it
pushes the model to flag far more positives -- exactly the
sensitivity/specificity trade a screening tool (where missing a real
finding is costlier than a false alarm) would want to make deliberately,
via threshold tuning, not accept as a side effect of the loss function
alone. Focal loss's down-weighting of easy examples nudged ROC-AUC
slightly higher without touching the recall/precision balance much,
because it doesn't asymmetrically penalize the positive class the way
Weighted BCE's `pos_weight` does -- it makes all easy examples cheaper,
regardless of class.

Model deployed to the API: `densenet121` trained with `weighted_bce`,
chosen for the sensitivity profile appropriate to a screening-support
tool, not for the (marginally lower) ROC-AUC.

## 10. Explainability: Grad-CAM + a Faithfulness Sanity Check

Grad-CAM (`src/explainability/gradcam.py`) is implemented directly with
forward/backward hooks on the last convolutional block -- no third-party
library, fully inspectable. It's exposed via `POST /explain` and rendered
in the frontend as a heatmap overlay.

The important part isn't the heatmap -- it's checking whether the heatmap
is telling the truth. `src/explainability/faithfulness.py` implements a
perturbation-based test: mask the top-20%-activation region Grad-CAM says
is important, re-run inference, and check whether the prediction drops
more than masking a same-sized random region does. Run across 45 (image,
label) pairs on the deployed model:

```
Overall mean faithfulness margin: +0.1044
Overall % of cases where masking the Grad-CAM region dropped the
prediction MORE than masking a random region: 75.6%

Per-label breakdown:
  Atelectasis     margin=+0.0851  faithful=83.3%  (n=6)
  Cardiomegaly    margin=+0.0119  faithful=66.7%  (n=6)
  Effusion        margin=+0.0801  faithful=50.0%  (n=6)
  Infiltration    margin=+0.1893  faithful=100.0% (n=6)
  Mass            margin=+0.2733  faithful=100.0% (n=3)
  Nodule          margin=+0.0866  faithful=66.7%  (n=6)
  Pneumonia       margin=+0.1694  faithful=83.3%  (n=6)
  Pneumothorax    margin=+0.0243  faithful=66.7%  (n=6)
```

Honest reading: the explanation is meaningfully more informative than
chance in most cases (75.6% overall, and Infiltration/Mass are consistently
faithful), but it's not uniformly reliable -- Effusion is a coin flip
(50%), and individual (image, label) pairs do occasionally show negative
margins (masking the "important" region barely moved the prediction, while
a random mask moved it more). This is exactly the kind of thing a
plausible-looking heatmap can hide, and exactly why this test exists rather
than stopping at "the heatmap looks reasonable."

## 11. Uncertainty & Calibration

MC Dropout (`src/uncertainty/mc_dropout.py`): re-enables dropout at
inference and runs 15-20 stochastic forward passes; the standard deviation
across those passes is reported alongside every prediction as
`uncertainty` plus a Low/Medium/High tier. Real example from the API:

```
Pneumothorax    0.981 +/- 0.007  [Low]
Pneumonia       0.853 +/- 0.038  [Low]
Effusion        0.403 +/- 0.054  [Medium]
```

Temperature scaling + ECE (`src/uncertainty/calibration.py`): fits a
single scalar T on the validation set (LBFGS, doesn't touch model
weights), then measures Expected Calibration Error before/after on the
held-out test set. Real numbers from the deployed model:

```
Fitted temperature: 1.4336
ECE before calibration: 0.2394
ECE after calibration:  0.2735   (worse, not better)
```

Honest reading, not a fabricated success story: calibration got slightly
worse here, not better. With only ~33 patients in the validation split,
fitting a single temperature parameter is noisy enough that it can overfit
the small validation sample's particular miscalibration direction rather
than the population's. On a dataset the size of real ChestX-ray14 (tens of
thousands of validation images), temperature scaling reliably improves ECE
in the published literature -- this result is a genuine artifact of
demo-scale data, reported as such rather than smoothed over. `/predict`
still reports both raw and calibrated probabilities so this can be
re-evaluated the moment real data is used.

## 12. API

FastAPI + Pydantic, interactive Swagger docs auto-generated at `/docs`.

| Endpoint | Description |
|---|---|
| `GET /health` | Liveness probe |
| `GET /model-info` | Active model, labels, image size, test-set metrics |
| `POST /predict` | Upload an image, get predictions + uncertainty (no Grad-CAM) |
| `POST /explain?label=...` | Same as `/predict`, plus a Grad-CAM heatmap for the given (or top) label |
| `GET /prediction/{id}` | Retrieve a stored prediction record |

```bash
curl -X POST http://localhost:8000/predict -F "file=@chest_xray.png"
curl -X POST "http://localhost:8000/explain?label=Pneumonia" -F "file=@chest_xray.png"
curl http://localhost:8000/prediction/1
```

Engineering details baked into every request: input validation (decodable
image, supported mode, size 32px-4096px, aspect ratio under 3:1, not
blank), structured logging on every validation failure and every
prediction, model-version tagging, and per-request latency measurement
(returned in the response, not just logged).

## 13. Database

PostgreSQL (via docker-compose) stores prediction metadata only --
`image_hash` (SHA-256 of the uploaded bytes, not the image itself), model
version, top finding, probability, latency, timestamp, and the full
per-label findings JSON. No image data and no patient data are ever stored
-- this demo only ever processes synthetic or public-dataset images, and
the schema is deliberately built to make storing real patient data awkward
(no patient-identifying fields exist to populate).

## 14. Deploying

1. Push this repo to GitHub -- `.github/workflows/ci.yml` will lint, test,
   and build the Docker image automatically (and will work correctly on
   GitHub's runners, which have the internet access this sandbox lacks).
2. Backend: Render / Railway / Cloud Run, connect the repo, it'll pick up
   the `Dockerfile`. Add a managed Postgres, set `DATABASE_URL`.
3. Frontend: already served by the FastAPI app itself at `/`, so no
   separate deploy is required -- though splitting it out to Vercel as a
   proper Next.js app is a reasonable next step once there's a real
   product around this.
4. Put the live Swagger URL at the top of this README:
   `https://<your-app>.onrender.com/docs`.

## 15. Testing

```bash
pytest -v   # 23 tests, ~17s
```

Covers: valid/corrupted/blank/unsupported-format image handling, response
schema validation (probabilities in range, uncertainty tiers valid),
Grad-CAM generation (default + explicit label + invalid label rejection),
prediction persistence + retrieval + 404, `/health` and `/model-info`,
quality-check detection of injected data problems, patient-level split
leakage (zero overlap, reproducibility), all 4 architectures building and
forward-passing, and loss function correctness.

## 16. Monitoring

Every prediction and every validation failure is logged via Python's
`logging` module (`medvision.*` loggers) in a structured, greppable
format -- point log aggregation (CloudWatch, Datadog, whatever) at stdout
and you have request counts, error rates, and prediction-distribution data
immediately. A dedicated Prometheus/Grafana stack is real but genuinely
out of scope for what this project needed to prove -- see "What's Next."

## 17. Limitations

- Not real medical data -- see the caveat table at the top. Every metric
  here is about the model's ability to learn a synthetic proxy task, not
  about detecting real disease.
- Not trained with real transfer learning in this environment (random init
  instead of ImageNet weights) -- the code supports it, the sandbox
  network doesn't.
- Tiny dataset (454 images) -- calibration and per-label metrics for
  low-prevalence findings (e.g. Mass, n=3 test positives) are noisy; take
  per-label numbers as directional, not precise.
- Grad-CAM is CNN-only -- no explainability method is implemented for ViT
  in this repo (see `src/models/factory.get_target_layer_for_gradcam`);
  attention-rollout would be the ViT-appropriate technique.
- No reject option -- the API always returns a prediction rather than
  deferring low-confidence cases to human review (deliberately out of
  scope for this build).

## 18. What's Next

- Real transfer learning on real ChestX-ray14/CheXpert data, at scale (the
  #1 item -- everything else here is validated machinery waiting for real
  data).
- Alembic migrations instead of `create_all()` for the predictions table.
- Prometheus + Grafana for real request-rate/latency/error dashboards,
  beyond the structured logs already in place.
- Attention-based explainability for ViT (attention rollout), so the
  architecture comparison and the explainability story aren't CNN-only.
- A proper Next.js/React frontend in place of the current single-file
  HTML/JS prototype, once there's a real product around this rather than a
  demo.
- SHAP or Integrated Gradients as a second explanation method to
  cross-check Grad-CAM's faithfulness results against.
