# Why Customers are unhappy?

## Business Background

A customer satisfaction survey was conducted by a fast-growing logistics and delivery company. The company works with several partners to deliver on-demand orders. Customer happiness is a key business measure, particularly as the company pursues a strategic plan for global expansion.

## Goal

The main focus was to identify which factors affected most for the unhappy customers.

## Objectives

The objectives of this work were:

- A model was trained to improve identification of unhappy customers
- Target an accuracy benchmark of 73%
- Identify the most important survey questions
- Identify low-value questions can be removed from future surveys

---

## The Dataset

Customer feedback was collected from an **unbiased** cohort using a short survey of six questions. Each question was rated from 1 to 5, where higher values indicate stronger agreement. Customer happiness was recorded as a binary outcome.

**Source:** `data/ACME-HappinessSurvey2020.csv`

### Data Description
- **Y**: target label, where **0 = unhappy** and **1 = happy**
- **X1**: Order delivered on time
- **X2**: Contents of the order was as expected
- **X3**: Ordered everything that was wanted
- **X4**: Paid a good price for the order
- **X5**: Satisfied with courier
- **X6**: The app made ordering easy

---

### Dataset Summary

- **Rows:** 126
- **Features:** 6 survey questions
- **Target classes:** Nearly balanced

![Target distribution](figures/target_distribution.png)

---

## Method

### Pipeline Overview

The following steps were carried out:

1. Dataset was checked for shape, types, missing values, and class balance
2. Data was split into training and testing sets (80/20)
3. Exploratory plots were produced using training data only to prevent leakage
4. A wide set of classifiers was screened using LazyPredict to get a fast baseline view
5. Special attention was given to identifying unhappy customers using minority-class recall
6. Four candidate models were selected and compared using 5-fold cross-validation
7. Ensemble methods (Voting, Stacking) were evaluated alongside base models; LGBM, Random Forest, Extra Trees, and KNN
8. Hyperparameter tuning was performed on LightGBM using Hyperopt
8. Final model was evaluated on the held-out test set
9. Feature importance and model-based feature selection were applied

All experiments use a fixed random seed (**7964**) for reproducibility.

### Design Decisions

| Decision | Rationale |
|----------|-----------|
| **80/20 split, no stratification** | Standard ratio for small datasets; classes are nearly balanced so stratification is unnecessary |
| **EDA on training data only** | Prevents data leakage -- visualising test data could influence modelling choices |
| **Minority recall as primary metric** | Business objective is to catch unhappy customers |
| **LazyPredict for initial screening** | Provides a fast baseline across ~30 classifiers using a single split |
| **5-fold cross-validation for model comparison** | More robust than a single split; each fold contains ~20 samples with ~8-10 minority-class instances |
| **LightGBM as final model** | Best cross-validated minority recall; gradient boosting handles small tabular data well |
| **class_weight='balanced'** | Upweights the minority class during training, directly supporting the business goal |
| **Hyperopt TPE (50 evaluations)** | Bayesian optimisation is more sample-efficient than grid/random search |
| **Constrained hyperparameter ranges** | max_depth <= 8, min_child_samples >= 10 to prevent overfitting on ~100 training samples |
| **SelectFromModel for feature selection** | Model-based selection using the tuned LightGBM; drops features below mean importance threshold |

---

## Key Results

### Baseline Model Screening (LazyPredict)

LazyPredict was used as an initial screening step with a single train/test split.

| Model                | F1 Score | Minority Recall |
|----------------------|----------|-----------------|
| ExtraTreesClassifier | 0.68     | 0.85            |
| LGBMClassifier       | 0.73     | 0.77            |
| RandomForest         | 0.68     | 0.69            |
| KNeighborsClassifier | 0.61     | 0.69            |

Extra Trees achieved the highest minority recall (0.85) in this initial run, correctly identifying approximately 85% of unhappy customers in that particular split.

### Cross-Validated Comparison

Four base models plus two ensemble methods were compared using 5-fold cross-validation scored on minority recall.

Models compared: LightGBM, Random Forest, Extra Trees, KNN, Voting (soft), Stacking.

![Model comparison](figures/model_comparison.png)

**Note on ensemble performance:** The ensembles did not outperform the best individual model. This is likely because 3 of the 4 base models are tree-based, limiting architectural diversity. Ensembles benefit most when base learners make different types of errors. A mix of tree-based, distance-based, and linear models would improve generalisation through the ensemble.

### Tuned LightGBM Performance (Test Set)

The tuned LightGBM model was evaluated on the held-out test set.

|           | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
| **0 (unhappy)** | 0.62   | **0.77**   | 0.69     | 13      |
| 1 (happy)   | 0.70   | 0.54   | 0.61     | 13      |
| Accuracy |        |        | 0.65 | 26      |

- **Test accuracy:** 0.65
- **Unhappy class recall (Y=0):** 0.77

The model prioritises catching unhappy customers (77% recall for class 0) at the cost of lower overall accuracy. This trade-off aligns with the business goal. The 73% accuracy target was not met, which is expected given the small dataset size (126 rows) and 6-feature constraint. With more data or richer features, accuracy would likely improve.

### Feature Importance

The most influential survey questions were:

1. **X5:** Satisfied with courier
2. **X4:** Paid a good price for the order

Less influential questions related to app usability and ordering completeness.

### Survey Optimisation

Based on feature selection, the following questions can be removed with minimal impact on predictive value:

- X6: The app makes ordering easy
- X2: Contents of the order was as expected
- X1: Order delivered on time
- X3: Ordered everything wanted to order

---

## Business Recommendations

From a business perspective, the following recommendations can be made:

1. **Courier experience and pricing are the leading indicators of unhappiness.** Operational improvements in these areas are likely to have the largest impact on customer satisfaction. If you pay more attention to those areas, future surveys can include more questions related to courier and pricing while eliminating the other four questions.
2. **Replacement questions** around courier experience (e.g. timeliness, professionalism, communication) and pricing fairness (e.g. value for money, price transparency) would provide more actionable detail.
3. **Future surveys can be shortened** by removing the four low-value questions.
---

## Repository Structure

```
.
├── data/
│   └── ACME-HappinessSurvey2020.csv   # Source dataset
├── src/
│   ├── __init__.py                     # Package exports
│   └── model.py                        # Core ML functions (load, split, EDA, train, tune, select, save)
├── models/
│   └── lgbm_model.joblib               # Pre-trained LightGBM model (generated by main.py)
├── figures/                            # Saved plots (auto-generated)
├── main.py                             # End-to-end pipeline script; saves model to models/lgbm_model.joblib
├── streamlit_app.py                    # Streamlit app — entry point for Community Cloud
├── analysis.ipynb                      # Interactive notebook (synced via jupytext)
├── analysis.py                         # Notebook as plain Python (jupytext percent format)
├── requirements.txt                    # Python dependencies
└── README.md
```

## How to Run

**Full pipeline (EDA, model comparison, tuning):**
```bash
pip install -r requirements.txt
python main.py
```

Outputs are saved under `figures/` and printed to the console.

**Run the Streamlit app locally:**
```bash
streamlit run streamlit_app.py
```

The Jupyter notebook (`analysis.ipynb`) provides the same pipeline in an interactive format and is synced with `analysis.py` via jupytext.
