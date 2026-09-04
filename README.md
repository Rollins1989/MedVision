# MedVision

**Explainable AI system for multi-label chest X-ray analysis.**

MedVision uses deep learning to analyze chest X-rays and identify multiple potential findings while providing **prediction confidence, uncertainty estimates, and visual explanations**.

### Problem

Medical imaging models can make predictions without showing **why** they made them, making their results difficult to interpret or trust.

### Solution

MedVision combines medical image classification with **uncertainty estimation and Grad-CAM explainability**, providing both predictions and visual evidence of the regions influencing the model.

### Features

* Multi-label chest X-ray classification
* CNN & Vision Transformer architectures
* Transfer-learning-ready models
* MC Dropout uncertainty estimation
* Temperature scaling & calibration
* Grad-CAM visual explanations
* Explanation faithfulness testing
* Patient-level data splitting to prevent leakage
* REST API with FastAPI
* PostgreSQL prediction storage
* MLflow experiment tracking
* Docker & Docker Compose
* Automated testing with Pytest
* GitHub Actions CI

### Tech Stack

**Python · PyTorch · FastAPI · PostgreSQL · SQLAlchemy · MLflow · Docker · Pytest · GitHub Actions**

### Pipeline

`X-ray → Quality Checks → Preprocessing → Deep Learning Model → Prediction + Uncertainty → Grad-CAM → API Response`

### Disclaimer

This is a **research/educational prototype** and is not intended for clinical diagnosis.

