# Happy Customers

## Business Background
A customer satisfaction survey was provided by a fast-growing logistics and delivery company.  They work with several partners and make on-demand delivery to their customers. As a start-up witha strategic plan of global expansion, customer happiness is their key challenge and plus point.

## The Dataset

Customer feedback was collected from a select cohort which is assumed to be a non bias sample through a short survey.  Each question was rated on a scale from 1 to 5, where higher values indicated stronger agreement.  Customer happiness was recorded as a binary outcome: happy or unhappy.

### Data Description
Y = target attribute (Y) with values indicating 0 (unhappy) and 1 (happy) customers
X1 = Order delivered on time
X2 = Contents of the order was as expected
X3 = Ordered everything that wanted to order
X4 = Paid a good price for the order
X5 = Satisfied with courier
X6 = The app makes ordering easy

## Goal

To identify which factors affected most for the unhappiness of the customers.

## Objectives

The objectives of this work were:

- To train model that improve the prediction accuracy of the unhappy customers (Predict more unhappy customers)
- To reach or exceed the required accuracy benchmark of 73%  
- To identify the most important survey questions
- To determine whether any questions could be safely removed from future surveys  

## Approach Taken

The following steps were carried out:

- The dataset was explored to confirm data quality and balance  
- The data was split into training and testing sets using an 80/20 split  
- Exploratory analysis was performed only on training data to avoid leakage  
- More than 30 classification models were evaluated using an automated model comparison approach  
- Special attention was given to identifying unhappy customers using minority-class recall  
- Multiple ensemble models were compared, including Random Forest, Extra Trees, Voting, and Stacking classifiers  
- Hyperparameter tuning was performed using Bayesian optimisation  
- Feature importance and model-based feature selection were applied  

All experiments were run with a fixed random seed to ensure reproducibility.

## Key Results

- The dataset contained 126 survey responses with balanced target classes  
- Several models achieved accuracy close to or above the target benchmark during cross-validation  
- Extra Trees and LightGBM-based models showed strong performance for identifying unhappy customers  
- Minority-class recall reached as high as 0.85 during model comparison  
- The final tuned LightGBM model achieved balanced and interpretable results on unseen test data  

This confirmed that customer happiness could be predicted reliably even with a small dataset.

---
# Recommandation

Courier and paying good price for the order are the leading indicators for unhappiness. If you pay more attention for those future surveys can include more questions related to those two areas while eliminating the other four questions.
