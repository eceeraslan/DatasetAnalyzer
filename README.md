# InsightForge 

**A Multi-Agent AI System for Automated Data Analysis and Machine Learning**

---

## Project Overview

InsightForge is a multi-agent AI-powered data science platform that automates the complete machine learning workflow.

Instead of focusing solely on model training, InsightForge performs:

* Dataset understanding
* Data quality assessment
* Data cleaning and preprocessing
* Machine learning model selection
* Model evaluation
* Visualization generation
* Automated reporting

The system combines Large Language Models (LLMs), CrewAI agents, and traditional machine learning algorithms to create an intelligent end-to-end data analysis pipeline.

---

## Motivation

Real-world machine learning projects require much more than simply training a model.

Data scientists must first answer several important questions:

* What type of problem is this?
* Is the dataset suitable for machine learning?
* Are there hidden missing-value indicators?
* Which features should be removed?
* Which algorithm should be selected?
* How should the final model be evaluated?

InsightForge automates these decisions through specialized AI agents.

---

# System Architecture

```text
Dataset Upload
       │
       ▼
Planner Agent
       │
       ▼
Data Preparation Agent
       │
       ▼
Modeling Agent
       │
       ▼
Evaluation Agent
       │
       ▼
Final Report + Visualizations
```

---

# Agents

## Planner Agent

The Planner Agent analyzes the dataset before any preprocessing or modeling occurs.

Responsibilities:

* Dataset inspection
* Data type analysis
* Correlation analysis
* Outlier detection
* Feature uniqueness analysis
* Dataset documentation analysis

The planner determines whether the problem should be treated as:

* Classification
* Regression
* Clustering

---

## Data Preparation Agent

The Data Preparation Agent improves dataset quality.

Features:

* Missing value handling
* Sentinel value detection (-200, -999, etc.)
* Identifier removal
* High-cardinality feature detection
* Dataset cleaning

---

## Modeling Agent

The Modeling Agent trains and compares multiple machine learning models.

### Supported Models

#### Classification

* Logistic Regression
* Random Forest Classifier
* Gradient Boosting Classifier

#### Regression

* Linear Regression
* Random Forest Regressor
* Gradient Boosting Regressor

#### Clustering

* K-Means Clustering

Cross-validation is used to compare candidate models and automatically select the best-performing model.

---

## Evaluation Agent

The Evaluation Agent reviews the complete workflow and produces a structured assessment.

Evaluation includes:

* Problem type correctness
* Data cleaning quality
* Model performance
* Visualization generation
* Improvement suggestions

---

# Features

✅ Automated dataset understanding

✅ Multi-agent workflow

✅ Automatic problem type detection

✅ Missing value detection

✅ Sentinel value detection

✅ Feature engineering support

✅ Cross-validation

✅ Automatic model comparison

✅ Visualization generation

✅ Trained model export (.pkl)

✅ Cleaned dataset export (.csv)

✅ Automated evaluation reports

---

# Technologies Used

### AI & Agents

* CrewAI
* Qwen LLM

### Data Science

* Scikit-Learn
* Pandas
* NumPy

### Visualization

* Matplotlib
* Seaborn

### Frontend

* Streamlit

---

# Installation

Clone the repository:

```bash
git clone https://github.com/your-repository/InsightForge.git
cd InsightForge
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file:

```env
DASHSCOPE_API_KEY=your_api_key
```

Run the application:

```bash
streamlit run app.py
```

---

# Usage

1. Upload a CSV dataset.
2. Optionally upload a dataset description document (PDF/TXT).
3. Specify a target column for classification or regression.
4. Click **Run Analysis**.
5. Review:

   * Planning Report
   * Data Preparation Report
   * Modeling Results
   * Evaluation Report
6. Download:

   * Cleaned Dataset
   * Trained Model
   * Generated Visualizations

---

# Example Outputs

The system automatically generates:

* Model Performance Metrics
* Feature Importance Charts
* Correlation Heatmaps
* Confusion Matrices
* Evaluation Reports

---

# License

This project was developed for academic and educational purposes.

---


## Conclusion

InsightForge demonstrates how multiple AI agents can collaborate to automate the complete machine learning workflow. By combining dataset understanding, preprocessing, modeling, and evaluation into a unified system, the project moves beyond traditional AutoML approaches and toward intelligent data science automation.
