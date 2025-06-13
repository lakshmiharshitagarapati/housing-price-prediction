# 🏠 Housing Price Prediction Project

> Predicting house prices using machine learning techniques on the [Kaggle House Prices dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques). 

![Python](https://img.shields.io/badge/Python-3.x-blue) 
![Model](https://img.shields.io/badge/Model-Random%20Forest-orange) 
![Kaggle Score](https://img.shields.io/badge/Kaggle%20Score-0.14740-green) 
## 📌 Overview

This project predicts housing prices using regression techniques based on features such as square footage, location, and year built. It includes exploratory data analysis (EDA), preprocessing, feature engineering, model training, and prediction.

The final score on Kaggle was **0.14740 (log RMSE)** using a Random Forest Regressor.
A machine learning project to predict house prices using the [Kaggle House Prices dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data). 
## 🥇 Kaggle Leaderboard Result

I ranked **#2785 out of 27,883 participants** on the Kaggle leaderboard!
### My Submission:
| Metric | Score |
|-------|--------|
| Log RMSE | 0.14740 |
| Rank | Top 10% |
### Leaderboard Screenshot:
![image](https://github.com/user-attachments/assets/8e1e7c36-048e-429c-a3f6-8258de9c5303)
## 📁 Folder Structure
housing-price-prediction/
├── data/
│   ├── train.csv
│   └── test.csv

├── images/
│   └── saleprice_hist.png  ← optional (example image)
│
├── notebooks/
│   └── eda.ipynb
│
├── main.py
├── submission.csv
├── README.md           ← updated version below
├── requirements.txt
├── LICENSE             ← added
└── .gitignore          ← optional
## 🧪 Features Used

- **Target Variable**: SalePrice
- **Numerical Features**: LotArea, GrLivArea, TotalBsmtSF
- **Categorical Features**: Neighborhood, YearBuilt, OverallQual
- **Custom Engineered Features**: TotalSqFt (TotalBsmtSF + GrLivArea), HouseAge (YrSold - YearBuilt)
## 🤖 Technologies Used

- **Python**
- Libraries:
  - `pandas`, `numpy` – data manipulation
  - `matplotlib`, `seaborn` – visualization
  - `scikit-learn`, `xgboost` – modeling
- Tools:
  - VS Code / Jupyter Notebook
  - Git & GitHub
## 📦 Requirements

Install dependencies:

```bash
pip install -r requirements.txt
MIT License – see LICENSE for details.
Contact
Feel free to reach out if you have any questions or suggestions!

Email : lakshmiharshita2003@gmail.com
LinkedIn :https://www.linkedin.com/in/lakshmi-harshita-garapati-8a0719259/?trk=opento_sprofile_details
GitHub : lakshmiharshitagarapati
Special thanks to:

Kaggle for providing the dataset
The open-source community for amazing tools like scikit-learn and pandas
