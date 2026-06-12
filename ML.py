import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import classification_report
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
from datetime import datetime

df = pd.read_csv(r"C:\Users\Rig1\Documents\csv\Marvel_Comics.csv")

# print(df.columns.tolist())
#
# print(df.head(3))
#
# print(df.dtypes)
#
# print(df.info())
#
#
# df["publish_date"] = pd.to_datetime(df["publish_date"], errors="coerce")
# df["publish_year"] = df["publish_date"].dt.year
# df["publish_month"] = df["publish_date"].dt.month
#
# df_num = df.select_dtypes(include=["number"]).dropna()
#
# threshold = df_num["APPEARANCES"].quantile(0.75)
# #df_num["HIGH_VALUE_PROXY"] = (df_num["APPEARANCES"] >= threshold).astype(int)
#
# #df_num["HIGH_VALUE_PROXY"] = ...
#
#
# X = df_num.drop(columns=["APPEARANCES", "HIGH_VALUE_PROXY"])
# y = df_num["HIGH_VALUE_PROXY"]
#
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.25, random_state=42
# )
#
# model = LogisticRegression(max_iter=1000)
# model.fit(X_train, y_train)
#
# preds = model.predict(X_test)
# print(classification_report(y_test, preds))

print("Rows:", len(df))
# print("Price missing:", df["Price"].isna().sum())
# print("Price non-missing:", df["Price"].notna().sum())
# print("Columns:", df.columns.tolist())

df.columns = df.columns.str.strip()
print(df.head(3))
#print(df.dtypes)

df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
print("Price nulls after cleaning:", df["Price"].isna().sum())
print(df[["Price"]].head(10))

df["Price_raw"] = df["Price"]

s = df["Price"].astype(str).str.strip()

df["Price"] = (
    s.str.replace(",", "", regex=False)
     .str.extract(r"(\d+(?:\.\d+)?)", expand=False)
)

df["Price"] = pd.to_numeric(df["Price"], errors="coerce")

print("Non-null Price count:", df["Price"].notna().sum())
print(df[["Price_raw", "Price"]].head(20))

df["publish_date"] = pd.to_datetime(df["publish_date"], errors="coerce")
df["publish_year"] = df["publish_date"].dt.year
df["publish_month"] = df["publish_date"].dt.month


target = "Price"
features = ["publish_year", "publish_month", "Imprint", "Format", "Rating"]

#df_model = df[features + [target]].dropna()

df_model = df[features + [target]].copy()
df_model = df_model[df_model["Price"].notna()]

for col in ["Imprint", "Format", "Rating"]:
    df_model[col] = df_model[col].fillna("Unknown")

df_model["publish_year"] = df_model["publish_year"].fillna(df_model["publish_year"].median())
df_model["publish_month"] = df_model["publish_month"].fillna(df_model["publish_month"].median())

print("Rows available for modeling:", len(df_model))

X = df_model[features]
y = df_model[target]

print("Total rows in df:", len(df))
print("Rows after cleaning:", len(df_model))
print(df_model.head())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

cat_cols = ["Imprint", "Format", "Rating"]
num_cols = ["publish_year", "publish_month"]

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols),
    ]
)

model = Pipeline(steps=[
    ("prep", preprocess),
    ("lr", LinearRegression())
])

model.fit(X_train, y_train)
preds = model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, preds))
print("R²:", r2_score(y_test, preds))











