# FraudShield MLOps

An end-to-end Machine Learning project for detecting fraudulent credit card transactions using Scikit-learn, FastAPI, and Docker.

---

## Overview

FraudShield is a machine learning application that predicts whether a credit card transaction is fraudulent.

The project covers the complete ML workflow:

- Data exploration
- Data preprocessing
- Model training
- Model evaluation
- Prediction API
- Automated testing
- Docker deployment

---

## Features

- Exploratory Data Analysis (EDA)
- Data preprocessing
- Random Forest Classifier
- Model evaluation
- FastAPI REST API
- Automated tests with Pytest
- Docker support

---

## Technologies

- Python 3.12
- Pandas
- NumPy
- Scikit-learn
- FastAPI
- Uvicorn
- Joblib
- Pytest
- Docker
- Git
- GitHub

---

## Project Structure

```text
FraudShield-MLOps/
│
├── api/
│   └── main.py
│
├── src/
│   ├── data_preprocessing.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── predict.py
│
├── tests/
│   ├── test_api.py
│   └── test_model.py
│
├── notebooks/
│   └── exploration.ipynb
│
├── models/
│
├── Dockerfile
├── .dockerignore
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Model Performance

| Metric | Score |
|---------|------:|
| Accuracy | **99.95%** |
| Precision | **90.59%** |
| Recall | **78.57%** |
| F1-score | **84.15%** |
| ROC-AUC | **95.73%** |

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Ali9104/fraudshield-mlops.git
cd fraudshield-mlops
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it (Windows):

```bash
.venv\Scripts\activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

---

## Run the API

```bash
uvicorn api.main:app --reload
```

Open:

```
http://127.0.0.1:8000/docs
```

---

## Docker

Build the Docker image:

```bash
docker build -t fraudshield-api .
```

Run the container:

```bash
docker run -p 8000:8000 fraudshield-api
```

Open:

```
http://localhost:8000/docs
```

---

## Running Tests

Run all tests:

```bash
python -m pytest
```

---

## API Example

### Request

```json
{
  "Time": 0,
  "V1": -1.359807,
  "V2": -0.072781,
  "...": "...",
  "Amount": 149.62
}
```

### Response

```json
{
  "prediction": 0,
  "fraud_probability": 0.0012
}
```

---

## License

This project was developed for educational purposes.
