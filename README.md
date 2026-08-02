# 🚗 Vehicle Fuel Efficiency Prediction

> **A complete end-to-end Machine Learning project for predicting vehicle fuel efficiency (MPG) using the UCI Auto-MPG dataset.**

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python) ![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange?logo=scikit-learn) ![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Problem Statement

Given the technical specifications of a vehicle (engine displacement, horsepower, weight, acceleration, model year, and origin), **predict its fuel efficiency in miles per gallon (MPG)** — a key metric for fleet management, emissions tracking, and procurement intelligence.

This is a practical demonstration of applying the full data science lifecycle — from raw data to production-ready model — on a real-world automotive dataset.

---

## 📁 Project Structure

```
vehicle-efficiency-ML/
│
├── data/
│   ├── auto-mpg.csv                    ← Raw dataset (Auto-MPG, UCI ML Repo)
│   ├── auto-mpg-cleaned.csv            ← After cleaning (missing values, types)
│   └── auto-mpg-features.csv           ← After feature engineering
│
├── notebooks/
│   ├── 01_problem_statement.ipynb
│   ├── 02_data_gathering_understanding.ipynb
│   ├── 03_data_cleaning.ipynb
│   ├── 04_eda.ipynb
│   ├── 05_feature_engineering.ipynb
│   ├── 06_model_building.ipynb
│   ├── 07_model_evaluation.ipynb
│   └── 08_business_insights_recommendations.ipynb
│
├── plots/                              ← All generated visualizations
│
├── models/
│   ├── random_forest_best.pkl
│   ├── gradient_boosting_best.pkl
│   └── final_model.pkl                 ← Production-ready model
│
├── src/
│   ├── utils.py                        ← Shared helper functions
│   └── predict.py                      ← Standalone prediction CLI
│
├── requirements.txt
└── README.md
```

---

## 🔄 End-to-End Workflow

```
1. Problem Statement        → Business context & success criteria
2. Data Gathering           → Load Auto-MPG CSV, inspect structure
3. Data Understanding       → Types, stats, missing values, distributions
4. Data Cleaning            → Impute HP missing values, fix types, drop car_name
5. EDA                      → Correlation, pair plots, VIF, categorical analysis
6. Feature Engineering      → Power-to-weight, log transforms, era encoding
7. Model Building           → 10 models compared via 5-fold CV
8. Model Evaluation         → RMSE, MAE, R², residual & learning curves
9. Business Insights        → 5 data-driven insights with visualizations
10. Recommendations         → 7 actionable business strategies
```

---

## 📊 Dataset

| Attribute | Description | Type |
|-----------|-------------|------|
| `mpg` | **Target** — Miles per gallon | Continuous |
| `cylinders` | Number of engine cylinders | Discrete |
| `displacement` | Engine displacement (cu in) | Continuous |
| `horsepower` | Engine horsepower | Continuous |
| `weight` | Vehicle weight (lbs) | Continuous |
| `acceleration` | 0–60 mph time (sec) | Continuous |
| `model_year` | Model year (70–82) | Discrete |
| `origin` | 1=USA, 2=Europe, 3=Japan | Categorical |
| `car_name` | Unique vehicle name | String |

- **Source:** [UCI Auto-MPG Dataset via Kaggle](https://www.kaggle.com/datasets/uciml/autompg-dataset)
- **Size:** 398 instances | 9 attributes | 6 missing values in `horsepower`

---

## 🤖 Models Trained

| Model | CV R² | CV RMSE |
|-------|--------|---------|
| Linear Regression | ~0.82 | ~3.4 |
| Ridge / Lasso | ~0.83 | ~3.2 |
| SVR | ~0.87 | ~2.9 |
| KNN | ~0.85 | ~3.1 |
| Decision Tree | ~0.86 | ~3.0 |
| **Random Forest** | **~0.92** | **~2.3** |
| Extra Trees | ~0.91 | ~2.4 |
| **Gradient Boosting** | **~0.93** | **~2.2** |
| XGBoost | ~0.93 | ~2.2 |

---

## 🔑 Key Insights

1. **Weight is #1 MPG driver** — every 500 lb reduction ≈ +3–4 MPG
2. **Japanese/European cars are ~50% more efficient** than US equivalents
3. **MPG improved ~30% post-1973 oil crisis** — regulations & downsizing
4. **4-cylinder cars dominate high-MPG segment** (>28 MPG)
5. **Power-to-Weight ratio** is the cleanest single efficiency KPI

---

## 🚀 Quick Start

```bash
# 1. Clone and install
git clone https://github.com/Arshdeep-Pasricha12/vehicle-fuel-efficiency-prediction-.git
cd vehicle-efficiency-ML
pip install -r requirements.txt

# 2. Run notebooks in order
jupyter notebook

# 3. Or predict directly
python src/predict.py
```

---

## 💼 Real-World Applications

This project demonstrates skills applicable across automotive and fleet analytics:
- **Telematics & IoT data analysis** (vehicle attributes as streaming signals)
- **Regression modeling** for real-time efficiency scoring
- **Fleet-level insights** derived from vehicle-level predictions
- **Anomaly detection** (flag vehicles where actual MPG deviates from predicted)
- **Carbon footprint estimation** (MPG → CO₂ g/mile for sustainability reporting)

---

## 👤 Author

**Arshdeep Pasricha**  
VIT University | B.Tech  
GitHub: [@Arshdeep-Pasricha12](https://github.com/Arshdeep-Pasricha12)
