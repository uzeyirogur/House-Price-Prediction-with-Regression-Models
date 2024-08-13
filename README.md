Description:
This project predicts house prices using different regression models, including Linear Regression, Ridge Regression, Lasso Regression, and XGBoost. The dataset is preprocessed by handling outliers, normalizing data, encoding categorical variables, and managing missing values. Finally, predictions are made, and results are exported for submission.

Project Overview:
This project aims to predict house prices using various regression techniques. The data undergoes preprocessing steps such as outlier removal, normalization, and encoding of categorical variables. The models used include Linear Regression, Ridge Regression, Lasso Regression, and XGBoost.

Code Summary:
Data Loading: Train and test datasets are loaded and combined for preprocessing.
Outliers Handling: Extreme values are removed based on standard deviation.
Data Transformation: Categorical features are encoded, and numerical data is normalized using log transformation.

Model Training: Multiple regression models are trained, including Linear, Ridge, Lasso, and XGBoost.
Prediction: The trained models predict house prices on the test set, and the results are saved as a CSV file.

How to Run:
Load the datasets (train.csv, test.csv).
Run the script to preprocess the data.
Train the models and generate predictions.
Submit the results as required.