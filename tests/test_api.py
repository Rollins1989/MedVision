"""
tests/test_api.py
--------------------
API-level tests: health, model-info, valid/invalid image handling,
prediction schema, Grad-CAM generation, prediction retrieval.
"""


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_model_info_endpoint(client):
    resp = client.get("/model-info")
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_name"] == "densenet121"
    assert len(body["labels"]) == 8
    assert "disclaimer" in body
    assert "not intended for clinical diagnosis" in body["disclaimer"]


def test_predict_valid_image_accepted(client, sample_image_path):
    with open(sample_image_path, "rb") as f:
        resp = client.post("/predict", files={"file": ("xray.png", f, "image/png")})
    assert resp.status_code == 200
    body = resp.json()
    assert "prediction_id" in body
    assert "top_finding" in body
    assert len(body["findings"]) == 8


def test_predict_response_schema_valid(client, sample_image_path):
    with open(sample_image_path, "rb") as f:
        resp = client.post("/predict", files={"file": ("xray.png", f, "image/png")})
    body = resp.json()
    for finding in body["findings"]:
        assert 0.0 <= finding["probability"] <= 1.0
        assert 0.0 <= finding["calibrated_probability"] <= 1.0
        assert finding["uncertainty"] >= 0.0
        assert finding["uncertainty_tier"] in ("Low", "Medium", "High")
    assert 0.0 <= body["top_probability"] <= 1.0
    assert body["latency_ms"] > 0


def test_predict_corrupted_image_rejected(client):
    resp = client.post("/predict", files={"file": ("bad.txt", b"not an image", "text/plain")})
    assert resp.status_code == 422


def test_predict_blank_image_rejected(client, blank_image_path):
    with open(blank_image_path, "rb") as f:
        resp = client.post("/predict", files={"file": ("blank.png", f, "image/png")})
    assert resp.status_code == 422
    assert "blank" in resp.json()["detail"].lower()


def test_predict_unsupported_format_rejected(client):
    # A truncated/garbled byte stream with a plausible filename
    resp = client.post("/predict", files={"file": ("fake.png", b"\x89PNG\r\ngarbage", "image/png")})
    assert resp.status_code == 422


def test_explain_generates_gradcam(client, sample_image_path):
    with open(sample_image_path, "rb") as f:
        resp = client.post("/explain", files={"file": ("xray.png", f, "image/png")})
    assert resp.status_code == 200
    body = resp.json()
    assert body["explanation"]["available"] is True
    assert body["explanation"]["method"] == "Grad-CAM"
    assert body["explanation"]["heatmap_png_base64"] is not None
    assert len(body["explanation"]["heatmap_png_base64"]) > 100


def test_explain_with_specific_label(client, sample_image_path):
    with open(sample_image_path, "rb") as f:
        resp = client.post("/explain?label=Cardiomegaly", files={"file": ("xray.png", f, "image/png")})
    assert resp.status_code == 200
    assert resp.json()["explanation"]["target_label"] == "Cardiomegaly"


def test_explain_with_invalid_label_rejected(client, sample_image_path):
    with open(sample_image_path, "rb") as f:
        resp = client.post("/explain?label=NotARealFinding", files={"file": ("xray.png", f, "image/png")})
    assert resp.status_code == 422


def test_prediction_stored_and_retrievable(client, sample_image_path):
    with open(sample_image_path, "rb") as f:
        create_resp = client.post("/predict", files={"file": ("xray.png", f, "image/png")})
    pred_id = create_resp.json()["prediction_id"]

    get_resp = client.get(f"/prediction/{pred_id}")
    assert get_resp.status_code == 200
    body = get_resp.json()
    assert body["id"] == pred_id
    assert body["top_finding"] == create_resp.json()["top_finding"]


def test_prediction_not_found_404(client):
    resp = client.get("/prediction/999999")
    assert resp.status_code == 404


def test_predict_no_file_rejected(client):
    resp = client.post("/predict")
    assert resp.status_code == 422


def test_disclaimer_present_on_predict(client, sample_image_path):
    with open(sample_image_path, "rb") as f:
        resp = client.post("/predict", files={"file": ("xray.png", f, "image/png")})
    assert "not intended for clinical diagnosis" in resp.json()["disclaimer"]


def test_openapi_docs_available(client):
    resp = client.get("/docs")
    assert resp.status_code == 200
    resp2 = client.get("/openapi.json")
    assert resp2.status_code == 200
