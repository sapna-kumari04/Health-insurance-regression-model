# ============================================
# STEP 1: IMPORT LIBRARIES
# ============================================

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer

# Regression models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Evaluation metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================
# STEP 2: LOAD DATASET
# ============================================

data = pd.read_csv(
    r"C:\\datatoolbox.py\INT234\\Indicators_of_Health_Insurance_Coverage_at_the_Time_of_Interview.csv"
)

# Drop rows where target is missing
data = data.dropna(subset=["Value"])

# Drop column with mostly missing values
data = data.drop(columns=["Suppression Flag"])

print(data.head())
print(data.info())
print(data.isnull().sum())


# ============================================
# STEP 3: DEFINE TARGET & FEATURES
# ============================================

target = "Value"

X = data.drop(columns=[target])
y = data[target]

# Identify column types from X only
numeric_cols = X.select_dtypes(include=np.number).columns
categorical_cols = X.select_dtypes(include="object").columns


# ============================================
# STEP 4: PREPROCESSING PIPELINES
# ============================================

numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, numeric_cols),
        ("cat", categorical_pipeline, categorical_cols)
    ]
)


# ============================================
# STEP 5: TRAIN TEST SPLIT
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ============================================
# STEP 6: EVALUATION FUNCTION
# ============================================

def evaluate_model(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    print(f"\n{name}")
    print("MAE :", mae)
    print("MSE :", mse)
    print("RMSE:", rmse)
    print("R2  :", r2)

    return mae, mse, rmse, r2


# ============================================
# STEP 7: LINEAR REGRESSION
# ============================================

linear_model = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", LinearRegression())
])

linear_model.fit(X_train, y_train)
pred_lr = linear_model.predict(X_test)

mae1, mse1, rmse1, r21 = evaluate_model(
    "LINEAR REGRESSION", y_test, pred_lr
)


# ============================================
# STEP 8: POLYNOMIAL REGRESSION (SAFE VERSION)
# ============================================

print("\n================ POLYNOMIAL REGRESSION ================")

poly_model = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("poly", PolynomialFeatures(degree=1, include_bias=False)),  # SAFE
    ("model", LinearRegression())
])

poly_model.fit(X_train, y_train)
pred_poly = poly_model.predict(X_test)

mae2, mse2, rmse2, r22 = evaluate_model(
    "POLYNOMIAL REGRESSION (DEGREE = 1)", y_test, pred_poly
)


# ============================================
# STEP 9: RANDOM FOREST REGRESSION (SAFE VERSION)
# ============================================

print("\n================ RANDOM FOREST REGRESSION ================")

rf_model = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", RandomForestRegressor(
        n_estimators=50,     # SAFE
        max_depth=10,        # SAFE
        random_state=42,
        n_jobs=-1
    ))
])

rf_model.fit(X_train, y_train)
pred_rf = rf_model.predict(X_test)

mae3, mse3, rmse3, r23 = evaluate_model(
    "RANDOM FOREST REGRESSION", y_test, pred_rf
)


# ============================================
# STEP 10: FINAL MODEL COMPARISON (PRINT ONLY)
# ============================================

print("\n================ FINAL MODEL COMPARISON ================\n")

print("{:<22} {:<10} {:<10} {:<10} {:<10}".format(
    "MODEL", "MAE", "MSE", "RMSE", "R2"
))
print("-" * 65)

print("{:<22} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f}".format(
    "Linear Regression", mae1, mse1, rmse1, r21
))

print("{:<22} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f}".format(
    "Polynomial Regression", mae2, mse2, rmse2, r22
))

print("{:<22} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f}".format(
    "Random Forest", mae3, mse3, rmse3, r23
))

print("\n================ END =================\n")



# ============================================
# STEP 11: CORRELATION HEATMAP
# ============================================

selected_cols = [
    "Value", "Low CI", "High CI", "Time Period", "Quartile Number"
]

plt.figure(figsize=(12, 7))
sns.heatmap(
    data[selected_cols].corr(),
    annot=True,
    cmap="Spectral",
    linewidths=0.5
)
plt.title("Correlation Heatmap - Health Insurance Indicators", fontsize=16)
plt.show()


# ============================================
# STEP 12: SCATTER PLOT
# ============================================

plt.figure(figsize=(8, 5))
plt.scatter(data["Time Period"], data["Value"], alpha=0.6)
plt.xlabel("Time Period")
plt.ylabel("Insurance Coverage Value")
plt.title("Time Period vs Insurance Coverage")
plt.grid(True)
plt.show()


# ============================================
# STEP 13: HISTOGRAMS
# ============================================

data[numeric_cols].hist(
    figsize=(12, 8),
    edgecolor="black"
)

plt.suptitle("Distribution of Numeric Features", fontsize=16)
plt.tight_layout()
plt.show()


# ======================================================== 
# Step 14: Trend Line (Insurance Coverage Across Time Period)
# ==========================================================
plt.figure(figsize=(12, 4))
plt.plot(data['Time Period'], data['Value'], linewidth=2)
plt.title("Trend of Health Insurance Coverage Across Time Period", fontsize=20)
plt.xlabel("Time Period", fontsize=16)
plt.ylabel("Insurance Coverage Value", fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True)
plt.show()



#=======================================================
# Step 15: Violin Plot - Insurance Coverage by Quartile
# ======================================================

plt.figure(figsize=(8, 5))
sns.violinplot(
    x=data['Quartile Number'],
    y=data['Value']
)
plt.title("Distribution of Insurance Coverage by Quartile", fontsize=16)
plt.xlabel("Quartile Number", fontsize=14)
plt.ylabel("Insurance Coverage Value", fontsize=14)
plt.show()


# ===================================
# Pairplot for Numeric Features
# ===================================

numeric_features = [
    'Value', 'Low CI', 'High CI', 'Time Period', 'Quartile Number'
]

sns.pairplot(data[numeric_features])
plt.suptitle(
    "Pairwise Relationships Between Health Insurance Indicators",
    y=1.02,
    fontsize=16
)
plt.show()


# ============================================
# Boxplot - Insurance Coverage Value
# ============================================
plt.figure(figsize=(6, 4))
sns.boxplot(y=data['Value'])
plt.title("Boxplot of Health Insurance Coverage Value", fontsize=16)
plt.ylabel("Insurance Coverage Value", fontsize=14)
plt.show()

# =========================================
# Count Plot - Quartile Distribution
# =========================================

plt.figure(figsize=(6, 4))
sns.countplot(x=data['Quartile Number'])
plt.title("Count of Records by Quartile Number", fontsize=16)
plt.xlabel("Quartile Number", fontsize=14)
plt.ylabel("Count", fontsize=14)
plt.show()




