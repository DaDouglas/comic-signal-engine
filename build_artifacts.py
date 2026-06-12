import os, sys
print("RUNNING:", __file__)
print("CWD:", os.getcwd())
print("PYTHON:", sys.executable)


import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib

DATA_PATH = r"C:\Users\Rig1\Documents\csv\Marvel_Comics.csv"  # change if needed
N_CLUSTERS = 5

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()

    # dates
    df["publish_date"] = pd.to_datetime(df["publish_date"], errors="coerce")
    current_year = datetime.now().year
    df["publish_year"] = df["publish_date"].dt.year
    df["comic_age_years"] = current_year - df["publish_year"]

    # creators
    for col in ["writer", "penciler", "cover_artist"]:
        df[col] = df[col].fillna("Unknown")

    df["writer_popularity"] = df["writer"].map(df["writer"].value_counts())
    df["penciler_popularity"] = df["penciler"].map(df["penciler"].value_counts())
    df["cover_artist_popularity"] = df["cover_artist"].map(df["cover_artist"].value_counts())

    # text structure
    df["title_length"] = df["issue_title"].astype(str).str.len()
    df["description_length"] = df["issue_description"].astype(str).str.len()
    df["word_count"] = df["issue_description"].astype(str).str.split().str.len()

    feature_cols = [
        "comic_age_years",
        "writer_popularity",
        "penciler_popularity",
        "cover_artist_popularity",
        "title_length",
        "description_length",
        "word_count",
    ]
    return df, df[feature_cols].copy(), feature_cols

def main():
    df = pd.read_csv(DATA_PATH)

    df, feature_df, feature_cols = build_features(df)

    # keep only rows with complete features
    valid_idx = feature_df.dropna().index
    X = feature_df.loc[valid_idx]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)

    df_out = df.loc[valid_idx].copy()
    df_out["cluster"] = clusters

    # Save human-readable artifact
    df_out.to_csv("comics_with_clusters_v1.csv", index=False)

    # Save machine artifact (for future online scoring)
    bundle = {
        "scaler": scaler,
        "kmeans": kmeans,
        "feature_cols": feature_cols,
        "n_clusters": N_CLUSTERS
    }
    joblib.dump(bundle, "model_bundle.joblib")

    print("Saved artifacts:")
    print(" - comics_with_clusters_v1.csv")
    print(" - model_bundle.joblib")
    print("Rows saved:", len(df_out))

if __name__ == "__main__":
    main()