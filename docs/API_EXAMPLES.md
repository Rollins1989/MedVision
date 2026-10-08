# MedVision API Examples

## Start the service

```bash
uvicorn app.main:app --reload
```

Interactive documentation is available at `/docs`; ReDoc is at `/redoc`.

## Health

```bash
curl http://127.0.0.1:8000/health
```

Example:

```json
{"status":"ok","model_loaded":true}
```

## Model information

```bash
curl http://127.0.0.1:8000/model-info
```

## Predict

### cURL

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "accept: application/json" \
  -F "file=@/path/to/chest_xray.png"
```

### Python

```python
import requests

with open("chest_xray.png", "rb") as image:
    response = requests.post(
        "http://127.0.0.1:8000/predict",
        files={"file": ("chest_xray.png", image, "image/png")},
        timeout=60,
    )

response.raise_for_status()
print(response.json())
```

## Explain

```bash
curl -X POST "http://127.0.0.1:8000/explain?label=Effusion" \
  -H "accept: application/json" \
  -F "file=@/path/to/chest_xray.png"
```

Omit `label` to explain the top prediction.

## Retrieve a prediction

```bash
curl http://127.0.0.1:8000/prediction/1
```

## Error contract

- `422`: invalid image input or unsupported explanation label.
- `404`: prediction ID does not exist.
- `503`: model checkpoint is unavailable.

## OpenAPI

FastAPI generates:

- Swagger UI: `/docs`
- ReDoc: `/redoc`
- Schema: `/openapi.json`

## Safety

This API is a research/educational interface, not a clinically validated diagnostic service. Medical images should be handled under appropriate privacy, consent, security, and retention controls.
