# Credit Risk Classification

## Overview of the Analysis

This project focuses on developing a supervised machine learning model to assess the creditworthiness of borrowers using historical loan data. The objective is to help a peer-to-peer (P2P) lending services company identify borrowers who are at high risk of defaulting on their loans. To achieve this, a logistic regression model was trained on labeled data, where the label represents whether a loan is healthy (0) or high-risk (1). The analysis evaluates the model’s ability to accurately predict loan status and ultimately guide the company's lending decisions.

## Project Structure

- `credit_risk_classification.ipynb`: Jupyter notebook containing the code for training, evaluating, and interpreting the model.
- `lending_data.csv`: Dataset used to train and test the model.
- `README.md`: The credit risk analysis report and project summary.

## Machine Learning Process

1. **Data Preparation**:
   - Loaded the dataset using Pandas.
   - Defined the target variable (`loan_status`) as `y`.
   - Defined the feature set (`X`) using the remaining columns.
   - Split the data into training and testing sets using `train_test_split`.

2. **Model Training**:
   - Applied a logistic regression model to the training data.
   - Generated predictions on the test dataset.

3. **Model Evaluation**:
   - Evaluated performance using a confusion matrix and a classification report.

## Results

The logistic regression model produced the following results on the test set:

- **Confusion Matrix Summary**:
  - **True Negatives (healthy loans correctly predicted):** 18,759
  - **True Positives (high-risk loans correctly predicted):** 594
  - **False Positives (healthy loans misclassified as high-risk):** 32
  - **False Negatives (high-risk loans misclassified as healthy):** 86

- **Classification Report**:
  - **Healthy Loans (Label: 0)**:
    - Precision: **1.00**
    - Recall: **1.00**
    - F1-score: **1.00**
  - **High-Risk Loans (Label: 1)**:
    - Precision: **0.87**
    - Recall: **0.95**
    - F1-score: **0.91**
  - **Overall Accuracy**: **0.99**
  - **Macro Avg (Unweighted)**:
    - Precision: 0.94
    - Recall: 0.97
    - F1-score: 0.95
  - **Weighted Avg (Weighted by support)**:
    - Precision: 0.99
    - Recall: 0.99
    - F1-score: 0.99

## Analysis and Interpretation

The logistic regression model demonstrates strong overall performance, particularly in identifying healthy loans. Here are key takeaways:

- **Healthy Loan Prediction (Label 0)**:
  - With approximately 100% precision and recall, the model nearly perfectly identifies loans that are not likely to default. This is critical for maintaining lending activity with minimal unnecessary rejection of good applications.
  
- **High-Risk Loan Prediction (Label 1)**:
  - Although not as perfect as for healthy loans, the model performs well, with a **precision of 87%** and **recall of 95%**.
  - A high recall (95%) means that the model successfully identifies the majority of actual high-risk loans. This is crucial for minimizing potential defaults.
  - The precision of 87% indicates that while some loans are falsely labelled as high-risk (false positives), the model is still mostly accurate in predicting high-risk cases.

- **Confusion Matrix Insight**:
  - Out of 625 high-risk loans, only 32 were missed (false negatives).
  - The symmetric false positive count (32) confirms that misclassification is relatively balanced.
  - The error rate for high-risk loans is about 5%, which is acceptable given the smaller size of the high-risk class. The imbalance in data likely contributes to this skew.

## Recommendation

Based on the high overall accuracy (99%) and strong recall for both loan categories, especially the high-risk category, the logistic regression model is **recommended** for use by the company. It is effective at minimising lending to borrowers with a high probability of default while maximising approval for healthy loan applicants.

The model's small margin of error, particularly for the minority class (high-risk), is acceptable in the context of P2P lending, where the cost of lending to a defaulter is higher than missing out on a potential borrower. However, it is advisable that future iterations of the model consider addressing the class imbalance further—potentially with techniques such as SMOTE, class weighting, or ensemble models—to enhance high-risk prediction even more.

## Next Steps

- Explore advanced classification techniques like Random Forest or Gradient Boosting for further improvements.
- Investigate class balancing methods to mitigate potential bias due to class imbalance.
- Consider integrating borrower-level financial and behavioural data to enrich the feature set and improve predictive accuracy.

---


