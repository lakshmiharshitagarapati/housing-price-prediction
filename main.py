# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error

# # Load data
# train = pd.read_csv('train.csv')
# print(train.head())
# print(train.info())

# # Check missing values
# print(train.isnull().sum().sort_values(ascending=False).head(20))

# # Distribution of target variable
# sns.histplot(train['SalePrice'], kde=True)
# plt.title("Distribution of Sale Price")
# plt.show()

# # Correlation matrix
# corr_matrix = train.corr(numeric_only=True)
# top_corr = corr_matrix['SalePrice'].sort_values(ascending=False).head(10)
# print(top_corr)
"""
House Price Prediction - Complete Pipeline
Author: Your Name
Date: April 5, 2025
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# -----------------------------
# 1. Load Data
# -----------------------------
print("🔄 Loading data...")

train_path = 'data/train.csv'
test_path = 'data/test.csv'

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

test_ids = test['Id'].copy()

# -----------------------------
# 2. Combine Train and Test for Consistent Processing
# -----------------------------
print("🧱 Combining train and test sets for preprocessing...")

all_data = pd.concat([train.drop('SalePrice', axis=1), test], axis=0)

# -----------------------------
# 3. Impute Missing Values
# -----------------------------
numeric_cols = all_data.select_dtypes(include=[np.number]).columns
all_data[numeric_cols] = all_data[numeric_cols].fillna(all_data[numeric_cols].median())

categorical_cols = all_data.select_dtypes(exclude=[np.number]).columns
all_data[categorical_cols] = all_data[categorical_cols].fillna(all_data[categorical_cols].mode().iloc[0])

# -----------------------------
# 4. One-Hot Encoding
# -----------------------------
print("🏷️ Performing one-hot encoding...")
all_data = pd.get_dummies(all_data)

# -----------------------------
# 5. Split Back into Train and Test
# -----------------------------
X_train = all_data.iloc[:len(train), :]
X_test = all_data.iloc[len(train):, :]

y_train = np.log(train['SalePrice'])  # Log transform target

# Ensure test has same columns as train
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# -----------------------------
# 6. Train Model
# -----------------------------
print("🧠 Training model...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# -----------------------------
# 7. Predict & Generate Submission
# -----------------------------
print("🔮 Making predictions...")
preds = model.predict(X_test)
final_preds = np.exp(preds)  # Reverse log transformation

submission = pd.DataFrame({
    'Id': test_ids,
    'SalePrice': final_preds
})

submission.to_csv('submission.csv', index=False)
print("✅ Submission file created successfully at 'submission.csv'")