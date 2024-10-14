# GSTN Hackathon - Machine Learning Model for GST Data Analysis

## Project Overview

This project aims to build a machine learning model for GST (Goods and Services Tax) data analysis to accurately classify and predict tax-related behavior. We tested multiple models and evaluated their performance based on accuracy, precision, recall, and F1-score. The XGBoost model provided the best performance and was selected for the final implementation.

## Dataset Description

The dataset used for model training and testing includes transactional data from the GST Network (GSTN). The following files were used:

- `X_Train_Data_Input.csv`: Training data with features for classification
- `Y_Train_Data_Target.csv`: Target labels for the training data
- `X_Test_Data_Input.csv`: Test data used for model evaluation
- `Y_Test_Data_Target.csv`: Target labels for the test data

## Evaluation Metrics

To evaluate the performance of different machine learning models, we used the following metrics:

- Accuracy: Measures how often the model is correct overall
- Precision: Measures the model's ability to return only relevant results
- Recall: Measures the model's ability to identify all relevant instances
- F1 Score: The harmonic mean of precision and recall, balancing the trade-offs between the two

## Model Prototypes Evaluated

We evaluated several machine learning models on the training and test datasets. Below are the results from each prototype tested, sorted by the F1 Score:

| Prototype | Accuracy | Precision | Recall | F1 Score |
|-----------|----------|-----------|--------|----------|
| XGBoost | 97.84 | 0.8496 | 0.9394 | 0.8923 |
| Artificial Neural Networks (ANN) | 97.54 | 0.8202 | 0.9492 | 0.8802 |
| AdaBoost | 97.53 | 0.8241 | 0.9413 | 0.8792 |
| Random Forest | 97.52 | 0.8437 | 0.9125 | 0.8774 |
| Logistic Regression | 96.86 | 0.803 | 0.882 | 0.8412 |
| Standard Scaler + Logistic | 96.88 | 0.8028 | 0.8805 | 0.8410 |
| Decision Tree | 96.73 | 0.8266 | 0.8310 | 0.8289 |
| Naive Bayes | 95.89 | 0.7305 | 0.8945 | 0.8041 |

## Why XGBoost?

After testing several models, we chose XGBoost as the final model for several reasons:

1. Performance: Consistently outperformed most other models in terms of accuracy, precision, recall, and F1-score.
2. Speed and Scalability: Optimized for speed and performance, especially with large datasets.
3. Handling Imbalanced Data: Built-in mechanisms to deal with imbalanced datasets through parameter tuning.
4. Feature Importance: Provides insights into feature importance for model interpretation and feature engineering.
5. Regularization: Includes L1 and L2 regularization to prevent overfitting and improve model robustness.

## XGBoost Model Performance

Detailed performance metrics for the XGBoost model:

- Accuracy: 97.84%
- Precision: 0.8496
- Recall: 0.9394
- F1 Score: 0.8923
- AUC-ROC: 0.9949

## Key Steps in Model Development

1. Data Preprocessing:
   - Handled missing values, normalized data, and removed unnecessary features
   - Applied label encoding to categorical variables

2. Model Training:
   - Trained multiple models using default and optimized hyperparameters
   - Performed hyperparameter tuning for XGBoost using techniques like grid search

3. Model Evaluation:
   - Evaluated each model on accuracy, precision, recall, and F1-score

4. Model Optimization:
   - Used Optuna for automated hyperparameter tuning
   - Optimized parameters like learning rate, max_depth, and n_estimators

5. Threshold Adjustment:
   - Tuned the classification threshold based on the precision-recall curve to maximize the F1-score

## How to Run the Model

To run the model, you'll need the following files:

1. `gst-model.pkl`: The trained XGBoost model saved as a pickle file.
2. `gstn-main.py`: Python script to load and run the model.
3. `gstn-main.ipynb`: Jupyter notebook version of the script.
4. Dataset files: `X_Train_Data_Input.csv`, `Y_Train_Data_Target.csv`, `X_Test_Data_Input.csv`, `Y_Test_Data_Target.csv`

### Steps to run the model:

1. Ensure you have Python installed on your system (preferably Python 3.7+).
2. Install required libraries:
   ```
   pip install pandas numpy scikit-learn xgboost
   ```
3. Place all the files (`gst-model.pkl`, `gstn-main.py`, `XGBoost_run.ipynb`, and the dataset files) in the same directory.

4. To run using the Python script:
   ```
   python gstn-main.py
   ```

5. To run using the Jupyter notebook:
   - Start Jupyter Notebook:
     ```
     jupyter notebook
     ```
   - Open `XGBoost_run.ipynb` in the Jupyter interface.
   - Run all cells in the notebook.


Note: Make sure you have sufficient RAM to load the datasets and the model. The exact requirements will depend on the size of your datasets.

## Conclusion

The XGBoost model demonstrated superior performance in classifying and predicting tax-related behavior based on GST data. With high accuracy, precision, recall, and F1-score, this model provides a robust solution for GST data analysis.

For more information or to contribute to this project, please contact the project maintainers.
