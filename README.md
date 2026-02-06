# Happy Customers

---

## Business Background
A customer satisfaction survey was provided by a fast-growing logistics and delivery company.  They work with several partners to deliver on-demand orders to their customers. Customer happiness is one of their key business measures, as a start-up with a strategic plan of global expansion.

## Goal

To identify which factors affected most for the unhappiness of the customers.

## Objectives

The objectives of this work were:

- To train model that improve the prediction accuracy of the unhappy customers (Predict more unhappy customers)
- To reach or exceed the required accuracy benchmark of 73%  
- To identify the most important survey questions
- To determine whether any questions could be safely removed from future surveys

---

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

---

## Approach Taken

The following steps were carried out:

- The dataset was explored to confirm data quality and balance  
- The data was split into training and testing sets using an 80/20 split  
- Exploratory analysis was performed only on training data to avoid leakage  
- More than 30 classification models were evaluated using an automated model comparison approach  
- Special attention was given to identifying unhappy customers using minority-class recall  
- Multiple ensemble models were compared, including Random Forest, Extra Trees, Voting, and Stacking classifiers (these 4 were given by the lazyClassifier as the best minory recallvalues. Lazy classifier uses train and test set which is a dataleakge. later on all ther models with two ensemble methods stacking and voting were compared with cross vaodation approch. 
- The best model was selected for furthur Hyperparameter tuning was performed usinh hyperopt
- Feature importance and model-based feature selection were applied  

Tested several random seed and selected a constant, All experiments were run with a fixed random seed to ensure reproducibility.

## Key Results

- The dataset contained 126 survey responses with  nearly balanced target classes
Following is th eresult of the LazyClassifier.  
- Extra Trees Clssifeir gae th best minority-recall from the lazy classifier. Minority-class recall reached as high as 0.85 during model comparison 
- Later the ensempble results do not provide bette r accuracy they are highlt affected withe selected types of model, specifically 3 out 4 models select for the ensempbling are tree based models. If the approch was with several architectures such treebased, distance based, gradient boosting based then ensembling have a better generatlisation of different techniques when irt comes to modelin g the data.

- LightGBM-based models showed strong performance for identifying unhappy customers during cross-validation. 
- The final tuned LightGBM model achieved balanced and interpretable results on unseen test data  

### Feature Importance Findings

Feature importance analysis showed clear patterns in what influenced customer happiness:

Most influential questions were:

1. Satisfaction with the courier  
2. Perceived price fairness  
3. Order content accuracy  

Less influential questions were related to app usability and ordering completeness.

### Survey Optimisation Recommendation

Based on feature selection results, the following questions were identified as removable with minimal impact on prediction performance:

- The app makes ordering easy  
- Contents of the order was as expected  
- Order delivered on time  
- Ordered everything wanted to order  

This suggested that future surveys could be shortened significantly while preserving predictive value.

---

# Recommandation

From a business perspective, the following recommendations can be made:

Courier and paying good price for the order are the leading indicators for unhappiness. If you pay more attention for those future surveys can include more questions related to those two areas while eliminating the other four questions.
