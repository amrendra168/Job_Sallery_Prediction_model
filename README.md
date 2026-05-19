# Job Salary Prediction Model

This repository contains a Python-based machine learning pipeline designed to predict employee salaries using job attributes and demographic data. The project carries out everything from End-to-End Exploratory Data Analysis (EDA) and data preprocessing to predictive modeling using a high-performing tree-based ensemble algorithm.

## 🔗 Notebook Link
You can open, view, and run the complete interactive notebook directly on Google Colab or GitHub:
* **[Open in Google Colab / View Notebook](https://colab.research.google.com/drive/1MusvaM8R3_bB3rQ-9XVcjAFTg5mNOBfB?usp=sharing)**

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset Summary](#dataset-summary)
- [Key Features & Workflow](#key-features--workflow)
- [Installation & Requirements](#installation--requirements)
- [Model Performance](#model-performance)
- [License](#license)

## Project Overview
Predicting employee salaries based on job descriptions and qualifications can help HR departments streamline hiring processes, ensure equal pay equity, and give candidates a clearer expectation of compensation. This notebook builds an optimized predictive model using a dataset containing 250,000 job entries.

## Dataset Summary
The data is retrieved from `job_salary_prediction_dataset.csv`, which consists of **250,000 rows** and **10 features**.

### Data Fields:
- **Numerical Features (4):**
  - `experience_years`: Total years of professional experience.
  - `skills_count`: Number of individual technical/soft skills possessed.
  - `certifications`: Number of professional certifications.
  - `salary` *(Target)*: The annual compensation package.
- **Categorical Features (6):**
  - `job_title` (e.g., AI Engineer, Data Analyst, Product Manager)
  - `education_level` (e.g., Bachelor, PhD)
  - `industry` (e.g., Healthcare, Telecom, Retail)
  - `company_size` (Small, Medium, Large)
  - `location` (e.g., India, Australia, Singapore)
  - `remote_work` (Yes, No, Hybrid)

## Key Features & Workflow

### 1. Exploratory Data Analysis (EDA)
- Analyzed dataset dimension, column types, and foundational statistics (`.info()`, `.describe()`).
- Confirmed the integrity of the data with zero missing values across all features.
- Visualized the target column distribution (`salary`) using Seaborn and Matplotlib histograms to ensure normality and evaluate potential skewness.

### 2. Feature Engineering & Preprocessing
- **Categorical Encoding:** Applied One-Hot Encoding to categorical columns utilizing `pd.get_dummies()` with `drop_first=True` to mitigate multicollinearity risks (dummy variable trap). This expands the original feature set into 43 numeric columns optimized for training.
- **Feature Scaling:** Implemented `StandardScaler` to align numerical features onto a common scale, stabilizing gradient descent behavior and optimizing convergence speeds during training.

### 3. Machine Learning Modeling
- Segregated the dataset into an **80/20 train-test split** to ensure unbiased testing validation.
- Trained a robust **Random Forest Regressor** using `Scikit-Learn`. Tree-based ensembles are uniquely suited for this data structure due to their built-in ability to map non-linear connections between categories (like specific industries/job types) and numerical outcomes.

## Installation & Requirements

Ensure you have the following dependencies installed to run the notebook:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
