# MP185 – Helpline Call Length Prediction

**Group ID:** G181  
**Problem ID:** MP185  
**Project Type:** Machine Learning – Regression

## Project Overview

This project focuses on predicting the handling time of citizen helpline calls using historical call-related information.

The objective is to build a machine learning regression model that predicts call duration and evaluates its performance using Mean Absolute Error (MAE).

## Objectives

- Explore and understand the provided datasets.
- Identify data quality issues and inconsistencies.
- Clean and preprocess data.
- Train and evaluate regression models.
- Track experiments and manage model versions.
- Build and containerize a prediction API.
- Monitor model performance and data drift.

## Tech Stack

- **Programming Language:** Python
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Experiment Tracking:** MLflow
- **Data Version Control:** DVC
- **API Development:** FastAPI
- **Containerization:** Docker
- **Version Control:** Git and GitHub

## Repository Structure

```text
MP185-Helpline-Call-Length-G181/
│
├── data/
│   └── raw/
│
├── reports/
│   └── findings.yaml
│
├── src/
│   └── mp185/
│       ├── __init__.py
│       ├── inspect_data.py
│       └── data_checks.py
│
├── .dvc/
├── .dvcignore
├── params.yaml
├── requirements.txt
└── README.md
```

## Project Progress

| Stage | Work | Status |
|---|---|---|
| Stage 1 | Repository setup, DVC and data inspection | Completed |
| Stage 2 | Data cleaning, MLflow and testing | Upcoming |
| Stage 3 | FastAPI, Docker and Kubernetes | Upcoming |
| Stage 4 | Monitoring, retraining and predictions | Upcoming |

## Team Members

| Name | Student ID |
|---|---|
| Rishabh Pandey Kumar | S24CSEU2129 |
| Pragya Singh | S24CSEU2251 |
| Anya Yadav | S24CSEU2249 |

## Evaluation Metric

**Mean Absolute Error (MAE)** is used to evaluate the regression model's prediction accuracy.

## Repository

[MP185 – Helpline Call Length Prediction](https://github.com/pragyaa-singhh/MP185-Helpline-Call-Length-G181)

---

*Academic project – Group G181*
