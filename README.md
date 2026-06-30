# 🍷 End-to-End Machine Learning Project with MLOps

A production-style, modular machine learning pipeline that predicts **wine quality** from physicochemical properties. The project demonstrates a complete MLOps workflow — from data ingestion to a deployed, containerized Flask web app — using clean software engineering practices (config-driven design, component-based pipeline stages, logging, and exception handling).

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Tech Stack](#-tech-stack)
- [Project Architecture](#-project-architecture)
- [Project Structure](#-project-structure)
- [Workflow](#-workflow)
- [Pipeline Stages](#-pipeline-stages)
- [Dataset](#-dataset)
- [Installation & Setup](#-installation--setup)
- [Usage](#-usage)
- [MLflow Experiment Tracking](#-mlflow-experiment-tracking)
- [Configuration Files](#-configuration-files)
- [Docker](#-docker)
- [CI/CD Deployment (AWS)](#-cicd-deployment-aws)
- [API / Web App](#-api--web-app)
- [Logging](#-logging)
- [License](#-license)

---

## 🔍 Overview

This project implements a full **regression pipeline** that predicts wine quality (a score from 0–10) using the Wine Quality dataset. Instead of a single notebook, the solution is engineered as a reusable Python package (`mlProject`) with five distinct, independently runnable pipeline stages:

1. Data Ingestion
2. Data Validation
3. Data Transformation
4. Model Training
5. Model Evaluation

Each stage reads its configuration from YAML files, is tracked via logs, and is wrapped behind a single orchestrator (`main.py`). The trained model is served through a **Flask** web application with an HTML form for live predictions, and the project is fully **Dockerized** with a **GitHub Actions CI/CD pipeline** for deployment to **AWS (ECR + EC2)**.

---

## 🛠 Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.8 |
| ML / Data | scikit-learn (ElasticNet), pandas, numpy |
| Experiment Tracking | MLflow |
| Config Management | PyYAML, python-box, ensure |
| Web Framework | Flask, Flask-Cors |
| Packaging | setuptools (`mlProject` as an installable package) |
| Containerization | Docker |
| CI/CD | GitHub Actions |
| Cloud / Deployment | AWS ECR, AWS EC2 (self-hosted runner) |
| Logging | Python `logging` module |

---

## 🏗 Project Architecture

```
                ┌─────────────────┐
                │   config.yaml    │
                │   params.yaml    │──────┐
                │   schema.yaml    │      │
                └────────┬─────────┘      │
                         │                │  (configuration)
                         ▼                ▼
   ┌──────────────────────────────────────────────────┐
   │                ConfigurationManager                │
   └──────────────────────────────────────────────────┘
                         │
   ┌─────────────────────┼─────────────────────────────────┐
   ▼                     ▼                                 ▼
Data Ingestion  →  Data Validation  →  Data Transformation  →  Model Trainer  →  Model Evaluation
(download/unzip)   (schema check)      (train/test split)    (ElasticNet)        (RMSE/MAE/R², MLflow log)
                                                                                          │
                                                                                          ▼
                                                                              artifacts/model_trainer/model.joblib
                                                                                          │
                                                                                          ▼
                                                                    ┌──────────────────────────────┐
                                                                    │   Flask App (app.py)          │
                                                                    │   PredictionPipeline           │
                                                                    └──────────────────────────────┘
                                                                                          │
                                                                                          ▼
                                                                         Docker Image → AWS ECR → AWS EC2
```

---

## 📁 Project Structure

```
End-to-End-Machine-Learning-Project-with-MLOPS/
├── .github/workflows/
│   └── main.yaml                  # CI/CD pipeline (GitHub Actions → AWS ECR/EC2)
├── artifacts/                     # Generated at runtime (data, models, metrics)
│   ├── data_ingestion/
│   ├── data_validation/
│   ├── data_transformation/
│   ├── model_trainer/
│   └── model_evaluation/
├── config/
│   └── config.yaml                # File paths and stage-level configuration
├── research/                      # Jupyter notebooks used to prototype each stage
│   ├── 01_data_ingestion.ipynb
│   ├── 02_data_validation.ipynb
│   ├── 03_data_transformation.ipynb
│   ├── 04_model_trainer.ipynb
│   └── 05_model_evaluation.ipynb
├── src/mlProject/
│   ├── components/                # Core logic for each pipeline stage
│   │   ├── data_ingestion.py
│   │   ├── data_validation.py
│   │   ├── data_transformation.py
│   │   ├── model_trainer.py
│   │   └── model_evaluation.py
│   ├── config/
│   │   └── configuration.py       # Reads YAML and builds typed config objects
│   ├── constants/
│   │   └── __init__.py            # Paths to config/params/schema files
│   ├── entity/
│   │   └── config_entity.py       # @dataclass config schemas for each stage
│   ├── pipeline/
│   │   ├── stage_01_data_ingestion.py
│   │   ├── stage_02_data_validation.py
│   │   ├── stage_03_data_transformation.py
│   │   ├── stage_04_model_trainer.py
│   │   ├── stage_05_model_evaluation.py
│   │   └── prediction.py          # Inference pipeline used by the Flask app
│   ├── utils/
│   │   └── common.py              # YAML/JSON/dir helpers, logging decorators
│   └── __init__.py                # Logger configuration
├── static/                        # CSS/JS/images for the web UI
├── templates/
│   ├── index.html                 # Input form
│   └── results.html               # Prediction output page
├── app.py                         # Flask entry point (training + prediction routes)
├── main.py                        # Orchestrates all 5 pipeline stages end-to-end
├── Dockerfile
├── params.yaml                    # Model hyperparameters (ElasticNet)
├── schema.yaml                    # Expected dataset schema
├── requirements.txt
├── setup.py                       # Makes `mlProject` pip-installable
└── README.md
```

---

## 🔄 Workflow

The project follows a consistent, repeatable workflow whenever a new pipeline stage is added or modified:

1. Update **`config.yaml`** — add/update file paths and directories for the stage.
2. Update **`schema.yaml`** — if the data schema changes.
3. Update **`params.yaml`** — if a model hyperparameter changes.
4. Update the **entity** (`entity/config_entity.py`) — add a typed config dataclass.
5. Update the **`ConfigurationManager`** (`config/configuration.py`) — add a method that builds the new config object.
6. Update the **component** (`components/`) — implement the stage's core logic.
7. Update the **pipeline** (`pipeline/stage_xx_*.py`) — wrap the component in a runnable stage.
8. Update **`main.py`** — call the new stage in the orchestrator.
9. *(Optional)* Update **`app.py`** — expose new functionality via the web app.

This config → entity → component → pipeline pattern keeps every stage decoupled, testable, and easy to extend.

---

## 🧩 Pipeline Stages

### 1️⃣ Data Ingestion
Downloads the source dataset (`winequality-data.zip`) from a remote URL defined in `config.yaml`, then extracts it into `artifacts/data_ingestion/`.

### 2️⃣ Data Validation
Validates that all columns in the ingested CSV match the expected schema defined in `schema.yaml`. Writes the result (`True`/`False`) to `artifacts/data_validation/status.txt`.

### 3️⃣ Data Transformation
Performs preprocessing and a train/test split (using `sklearn.model_selection.train_test_split`), saving `train.csv` and `test.csv` to `artifacts/data_transformation/`.

### 4️⃣ Model Trainer
Trains an **ElasticNet regression** model using hyperparameters (`alpha`, `l1_ratio`) defined in `params.yaml`. The trained model is serialized with `joblib` to `artifacts/model_trainer/model.joblib`.

### 5️⃣ Model Evaluation
Loads the trained model, evaluates it on the test set using **RMSE**, **MAE**, and **R²**, saves the metrics to `artifacts/model_evaluation/metrics.json`, and logs parameters/metrics/the model itself to **MLflow**.

---

## 📊 Dataset

The project uses the **Wine Quality (Red Wine)** dataset, sourced from:
`https://github.com/entbappy/Branching-tutorial/raw/master/winequality-data.zip`

**Features (11 input columns):**

| Feature | Type |
|---|---|
| fixed acidity | float64 |
| volatile acidity | float64 |
| citric acid | float64 |
| residual sugar | float64 |
| chlorides | float64 |
| free sulfur dioxide | float64 |
| total sulfur dioxide | float64 |
| density | float64 |
| pH | float64 |
| sulphates | float64 |
| alcohol | float64 |

**Target column:** `quality` (int64) — wine quality score.

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.8+
- Git
- (Optional) Conda for environment management

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/End-to-End-Machine-Learning-Project-with-MLOPS.git
cd End-to-End-Machine-Learning-Project-with-MLOPS

# 2. Create and activate a virtual environment
conda create -n mlproj python=3.8 -y
conda activate mlproj

# 3. Install dependencies (also installs mlProject as an editable local package)
pip install -r requirements.txt
```

> `requirements.txt` includes `-e .`, which uses `setup.py` to install the `src/mlProject` package in editable mode — this is what makes `from mlProject... import ...` work throughout the codebase.

---

## ▶️ Usage

### Run the full training pipeline (all 5 stages)

```bash
python main.py
```

This sequentially executes Data Ingestion → Data Validation → Data Transformation → Model Training → Model Evaluation, logging progress for each stage to `logs/running_logs.log` and producing all artifacts under `artifacts/`.

### Run a single stage (for debugging/development)

```bash
python -m mlProject.pipeline.stage_01_data_ingestion
python -m mlProject.pipeline.stage_04_model_trainer
```

### Launch the web application

```bash
python app.py
```

Then open **`http://localhost:8080`** in your browser:
- `/` — Home page with the prediction form
- `/train` — Triggers `main.py` to retrain the model from the browser
- `/predict` — Accepts form input and returns the predicted wine quality

---

## 📈 MLflow Experiment Tracking

Model evaluation logs hyperparameters, metrics (RMSE, MAE, R²), and the trained model artifact to MLflow.

**Run a local MLflow UI:**

```bash
mlflow ui
```

Then visit `http://localhost:5000` to compare runs.

**To track against a remote MLflow server** (e.g., hosted on DagsHub), set the following environment variables before running the pipeline:

```bash
export MLFLOW_TRACKING_URI=<your-mlflow-tracking-uri>
export MLFLOW_TRACKING_USERNAME=<your-username>
export MLFLOW_TRACKING_PASSWORD=<your-token>
```

> ⚠️ Never commit credentials to source control — always use environment variables or a secrets manager.

---

## 📝 Configuration Files

| File | Purpose |
|---|---|
| `config/config.yaml` | Defines directories, file paths, and the dataset source URL for every stage |
| `params.yaml` | Model hyperparameters — currently `alpha` and `l1_ratio` for ElasticNet |
| `schema.yaml` | Expected column names, dtypes, and the target column |

Example — adjusting model hyperparameters in `params.yaml`:

```yaml
ElasticNet:
  alpha: 0.2
  l1_ratio: 0.1
```

---

## 🐳 Docker

**Build the image:**

```bash
docker build -t wine-quality-mlops .
```

**Run the container:**

```bash
docker run -p 8080:8080 wine-quality-mlops
```

The app will be accessible at `http://localhost:8080`.

---

## 🚀 CI/CD Deployment (AWS)

The repository ships with a GitHub Actions workflow (`.github/workflows/main.yaml`) that implements a 3-stage CI/CD pipeline:

1. **Continuous Integration** — checks out code, lints, and runs tests.
2. **Continuous Delivery** — builds the Docker image and pushes it to **Amazon ECR**.
3. **Continuous Deployment** — pulls the latest image on a **self-hosted EC2 runner** and runs it as a container, exposing it on port `8080`.

### Required GitHub Secrets

| Secret | Description |
|---|---|
| `AWS_ACCESS_KEY_ID` | IAM user access key |
| `AWS_SECRET_ACCESS_KEY` | IAM user secret key |
| `AWS_REGION` | e.g. `us-east-1` |
| `AWS_ECR_LOGIN_URI` | ECR registry URI |
| `ECR_REPOSITORY_NAME` | ECR repository name |

### High-level AWS setup

1. Create an **IAM user** with `AmazonEC2ContainerRegistryFullAccess` and `AmazonEC2FullAccess`.
2. Create an **ECR repository** to store the Docker image.
3. Launch an **EC2 instance** (Ubuntu), install Docker on it.
4. Register the EC2 instance as a **self-hosted GitHub Actions runner**.
5. Add the secrets above to the GitHub repository settings.
6. Push to the `main` branch — the workflow builds, pushes, and deploys automatically.

---

## 🌐 API / Web App

| Route | Method | Description |
|---|---|---|
| `/` | GET | Renders the home page with the input form |
| `/train` | GET | Triggers `main.py` to retrain the full pipeline |
| `/predict` | GET, POST | Accepts 11 wine feature values and returns the predicted quality |

The prediction route uses `PredictionPipeline` (`src/mlProject/pipeline/prediction.py`), which loads `artifacts/model_trainer/model.joblib` and returns a prediction for the submitted feature vector.

---

## 🧾 Logging

All pipeline stages log progress, warnings, and errors to both the console and `logs/running_logs.log`, configured centrally in `src/mlProject/__init__.py`. Each stage logs clear start/end markers (`>>>>>> stage X started <<<<<<` / `completed`), making it easy to trace failures back to a specific step.

---

## 📄 License

This project is released under the terms specified in the [LICENSE](./LICENSE) file.

---

## 🙌 Acknowledgements

Built as a learning project to demonstrate practical **MLOps principles**: modular pipeline design, experiment tracking with MLflow, containerization, and automated CI/CD deployment to the cloud.
