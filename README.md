# Titanic Survival Prediction

**By:** Mohamad El Husseiny  
**Date:** 2025-06-08

A machine learning project to predict whether passengers survived the Titanic disaster using Logistic Regression, Random Forest, and XGBoost. This repository contains a clear, end-to-end Google Colab notebook, detailed data exploration, preprocessing steps, model training and evaluation, and final predictions ready for Kaggle submission.

---

## Table of Contents

1. [Project Overview](#project-overview)  
2. [Getting Started](#getting-started)  
3. [Repository Structure](#repository-structure)  
4. [Notebook Walkthrough](#notebook-walkthrough)  
5. [Installation Requirements](#installation-requirements)  
6. [How to Run](#how-to-run)  
7. [Results & Insights](#results--insights)  
8. [Submission](#submission)  
9. [Next Steps](#next-steps)  
10. [License](#license)

---

## Project Overview

The Titanic dataset is a classic binary classification problem: given passenger attributes (age, sex, class, etc.), predict who survived the sinking of the Titanic. In this project, you will:

- Perform exploratory data analysis (EDA) to understand feature distributions and missing values.  
- Preprocess data: impute missing values, engineer features (Titles from names), encode categorical variables, and scale numeric features.  
- Build baseline models: Logistic Regression, Random Forest, and XGBoost.  
- Tune hyperparameters using GridSearchCV for improved performance.  
- Generate final predictions for Kaggle submission.

## Getting Started

To get your own copy of the project up and running locally or in Google Colab, follow these steps.

### Repository Structure

```
README.md
titanic_notebook.ipynb   # Main Colab notebook
train.csv                # Training data (in the repository root)
test.csv                 # Test data (in the repository root)
LICENSE
```

## Notebook Walkthrough

All analysis is contained in `titanic_notebook.ipynb` (or open in Colab):

1. **EDA**  
   - `train.info()`, `train.describe()`  
   - Missing-value heatmap  
   - Survival rate by sex, class, and age groups  
2. **Preprocessing**  
   - Extract `Title` from `Name`  
   - Drop unused columns  
   - Pipelines for numeric (median imputation + scaling) and categorical (most-frequent fill + one-hot encoding)  
3. **Modeling**  
   - Baseline Logistic Regression  
   - Random Forest classifier  
   - XGBoost classifier  
4. **Hyperparameter Tuning**  
   - `GridSearchCV` on Random Forest hyperparameters  
   - Compare tuned vs. default performance  
5. **Submission**  
   - Generate `submission.csv` with `PassengerId` & predicted `Survived`

## Installation Requirements

The notebook uses the following Python libraries:

- `pandas`  
- `numpy`  
- `seaborn`  
- `matplotlib`  
- `scikit-learn`  
- `xgboost`

In Colab, you can install missing packages with:

```bash
!pip install seaborn xgboost
```

## How to Run

1. Open the notebook in Google Colab or locally with Jupyter.  
2. Ensure `train.csv` and `test.csv` are uploaded or mounted from your drive.  
3. Run all cells in order:  
   - EDA  
   - Preprocessing  
   - Train/validation split  
   - Model training & evaluation  
   - Hyperparameter tuning  
   - Submission file generation

## Results & Insights

- **Logistic Regression:** ~80% accuracy, precision ~0.74, recall ~0.77  
- **Random Forest:** ~82% accuracy after preprocessing  
- **XGBoost:** ~81% accuracy  
- **Tuned Random Forest (GridSearchCV):** improved accuracy ~83% on validation set

**Key observations:**  
- Female and first-class passengers had the highest survival rates.  
- Missing `Age` and `Cabin` values were common; median imputation and dropping `Cabin` worked well.  
- Title extraction (Mr/Miss/Mrs) added predictive power.

## Submission

1. Download `submission.csv` from the notebook output.  
2. Upload to the [Kaggle Titanic competition](https://www.kaggle.com/competitions/titanic/) under "Submit Predictions".  
3. View your public leaderboard score.

## Next Steps

- Tune XGBoost hyperparameters or use `RandomizedSearchCV` for faster search.  
- Add feature interactions (e.g., family size = `SibSp + Parch + 1`).  
- Explore ensemble of top models (stacking or voting classifier).  
- Test alternative imputation strategies (KNNImputer, predictive models).

## License

This project is released under the MIT License.
