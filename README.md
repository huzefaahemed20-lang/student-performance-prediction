# 🎓 Student Performance Prediction System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.20%2B-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.0%2B-F7931E.svg)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-1.5%2B-150458.svg)](https://pandas.pydata.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](#license)

An end-to-end Machine Learning web application designed to predict and analyze student academic performance (`Pass` / `Fail`) based on academic indicators, attendance, and study habits. Built with **Python**, **Scikit-Learn**, and **Streamlit**.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Project Architecture & Directory Structure](#-project-architecture--directory-structure)
- [Dataset Overview](#-dataset-overview)
- [Machine Learning Workflow](#-machine-learning-workflow)
- [Model Performance](#-model-performance)
- [Prerequisites](#-prerequisites)
- [Installation & Setup](#-installation--setup)
- [How to Run](#-how-to-run)
  - [1. Launch Streamlit Web Application](#1-launch-streamlit-web-application)
  - [2. Run Jupyter Notebook Analysis](#2-run-jupyter-notebook-analysis)
- [Streamlit App Features & Pages](#-streamlit-app-features--pages)
- [Technologies Used](#-technologies-used)
- [License](#-license)

---

## 📌 Overview

Early identification of students at academic risk allows educators and institutions to step in with timely interventions. The **Student Performance Prediction System** utilizes a trained machine learning pipeline to evaluate key academic metrics—such as class attendance, internal assessment scores, study hours, previous performance, and assignment grades—to accurately predict student outcomes and output personalized improvement advice.

---

## ✨ Key Features

- **🔮 Real-Time Outcome Prediction**: Instantly predicts whether a student will `Pass` or `Fail` with confidence probability percentages.
- **💡 Actionable Student Recommendations**: Evaluates input parameters against benchmark thresholds and delivers targeted warnings and study recommendations.
- **📊 Comprehensive Data Analysis**: Interactive exploration of dataset distribution, missing values, descriptive statistics, and feature histograms.
- **🤖 Model Performance & Transparency**: Detailed breakdown of accuracy metrics, confusion matrices, and feature importance rankings.
- **🔍 Individual Student Insights**: Interactive slider dashboard allowing educators to profile individual students and flag risk factors.
- **🎨 Modern Streamlit Dashboard**: Clean multi-page sidebar navigation with intuitive layouts and metrics cards.

---

## 📁 Project Architecture & Directory Structure

```text
student-performance-prediction/
│
├── data/
│   ├── raw/
│   │   └── student_performance.csv    # Raw dataset (5,000 student records)
│   └── processed/                      # Preprocessed data storage
│
├── models/
│   └── best_model.pkl                 # Serialized Scikit-Learn ML Pipeline model
│
├── notebooks/
│   └── 01_student_performance_analysis.ipynb  # EDA, preprocessing & model training notebook
│
├── reports/
│   └── model_results.csv              # Benchmark comparison results of trained models
│
├── src/                               # Modular Python source code directory
├── .gitignore                         # Files ignored by Git
├── app.py                             # Main Streamlit Web Application entry point
├── requirements.txt                   # Python library dependencies
└── README.md                          # Project documentation
```

---

## 📊 Dataset Overview

The model is trained on a dataset of **5,000 student records** containing key academic indicators:

| Feature Name | Type | Unit / Scale | Description |
| :--- | :--- | :--- | :--- |
| `student_id` | Categorical | ID (e.g. `S00001`) | Unique identifier for each student |
| `attendance` | Numerical | Percentage (`0.0 - 100.0%`) | Class attendance percentage |
| `internal_marks` | Numerical | Score (`0.0 - 100.0`) | Internal assessment / mid-term test score |
| `study_hours` | Numerical | Hours / Day (`0.0 - 10.0 hrs`) | Daily study time in hours |
| `previous_performance` | Numerical | Score (`0.0 - 100.0`) | Overall academic score in the previous term |
| `assignment_score` | Numerical | Score (`0.0 - 100.0`) | Average score across coursework assignments |
| `result` **(Target)** | Categorical | `Pass` / `Fail` | Final outcome (Target Variable) |

### Dataset Summary Statistics
- **Total Rows**: 5,000
- **Total Features**: 5 numeric predictor features + 1 target
- **Missing Values**: 0
- **Class Distribution**: 3,492 Pass (69.8%) | 1,508 Fail (30.2%)

---

## ⚙️ Machine Learning Workflow

1. **Data Ingestion & Cleaning**: Load `student_performance.csv`, verify data types, check for nulls, and deduplicate records.
2. **Exploratory Data Analysis (EDA)**: Statistical profiling and visualization of numeric feature distributions against student outcomes.
3. **Preprocessing Pipeline**:
   - `SimpleImputer(strategy='mean')`: Impute any potential missing values.
   - `StandardScaler()`: Standardize features by scaling to zero mean and unit variance.
4. **Model Training & Comparison**: Train multiple classifiers (Logistic Regression, Decision Trees).
5. **Model Serialization**: Export the top-performing pipeline using `joblib` into `models/best_model.pkl`.

---

## 🏆 Model Performance

Model evaluation was conducted on an independent test dataset split.

| Model Algorithm | Preprocessing Pipeline | Test Accuracy | Status |
| :--- | :--- | :--- | :--- |
| **Logistic Regression** | `Imputer + StandardScaler` | **79.6%** | **Selected (Best Model)** |
| **Decision Tree Classifier** | `Imputer + StandardScaler` | 75.9% | Evaluated |

---

## 🛠️ Prerequisites

Before getting started, ensure you have the following installed on your system:
- **Python 3.8** or higher ([Download Python](https://www.python.org/downloads/))
- **Git** ([Download Git](https://git-scm.com/))
- **Pip** (comes bundled with Python)

---

## 🚀 Installation & Setup

Follow these steps to set up the project locally on your machine.

### Step 1: Clone the Repository

```bash
git clone https://github.com/disha-birari/student-performance-prediction.git
cd student-performance-prediction
```

### Step 2: Create a Virtual Environment

- **On Windows (PowerShell / CMD):**
  ```powershell
  python -m venv .venv
  .venv\Scripts\activate
  ```

- **On macOS / Linux:**
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🖥️ How to Run

### 1. Launch Streamlit Web Application

Run the Streamlit dashboard server locally:

```bash
streamlit run app.py
```

Once executed, open your web browser and navigate to:
```text
http://localhost:8501
```

### 2. Run Jupyter Notebook Analysis

To inspect the exploratory data analysis, data preprocessing, and model training workflow:

```bash
jupyter notebook notebooks/01_student_performance_analysis.ipynb
```

---

## 📱 Streamlit App Features & Pages

The web application provides **6 interactive pages** accessible via the sidebar navigation:

1. **🏠 Dashboard**: High-level overview of total student records, feature count, model selected, and system workflow stages.
2. **🔮 Predict Result**: Input form where users enter attendance, internal marks, study hours, previous performance, and assignment scores to get real-time `Pass`/`Fail` predictions, probability scores, and targeted study advice.
3. **📊 Dataset Analysis**: Data preview, shape stats, missing value audit, dataset describe tables, and dynamic feature histogram visualizer.
4. **🤖 Model Performance**: Displays test accuracy, classification metrics table, and model feature coefficient importance charts.
5. **🔍 Student Insights**: Interactive sliders allowing teachers to quickly stress-test student parameters and flag potential risk areas.
6. **ℹ️ About Project**: Details on input features, end-to-end ML pipeline steps, technology stack list, and disclaimer notices.

---

## 💻 Technologies Used

- **Language**: Python 3.8+
- **Machine Learning**: Scikit-Learn
- **Web Framework**: Streamlit
- **Data Manipulation**: Pandas, NumPy
- **Data Visualization**: Matplotlib, Seaborn
- **Model Serialization**: Joblib
- **Development Environment**: Jupyter Notebook, VS Code

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

---

<p center>
  Made with ❤️ for Academic Excellence & Data Science
</p>
