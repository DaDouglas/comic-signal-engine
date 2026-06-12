import pandas as pd
from datetime import datetime

df = pd.read_csv(r"C:\Users\Rig1\Documents\csv\Marvel_Comics.csv")
df.columns = df.columns.str.strip()

print("Rows:", len(df))
print(df.head(3))

df["publish_date"] = pd.to_datetime(df["publish_date"], errors="coerce")

current_year = datetime.now().year
df["publish_year"] = df["publish_date"].dt.year
df["comic_age_years"] = current_year - df["publish_year"]


for col in ["writer", "penciler", "cover_artist"]:
    df[col] = df[col].fillna("Unknown")

writer_counts = df["writer"].value_counts()
penciler_counts = df["penciler"].value_counts()
cover_counts = df["cover_artist"].value_counts()

df["writer_popularity"] = df["writer"].map(writer_counts)
df["penciler_popularity"] = df["penciler"].map(penciler_counts)
df["cover_artist_popularity"] = df["cover_artist"].map(cover_counts)

df["Imprint"] = df["Imprint"].fillna("Unknown")
df["Format"] = df["Format"].fillna("Unknown")
df["Rating"] = df["Rating"].fillna("Unknown")

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
    "word_count"
]

feature_df = df[feature_cols].copy()
feature_df = feature_df.dropna()

print("Feature table shape:", feature_df.shape)
print(feature_df.head(10))

feature_df.to_csv("comic_feature_table_v1.csv", index=False)
print("Saved comic_feature_table_v1.csv")

from sklearn.preprocessing import StandardScaler

X = feature_df.copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

feature_df["cluster"] = clusters
print(feature_df.head(10))

cluster_summary = feature_df.groupby("cluster").mean()
print(cluster_summary)

df_with_clusters = df.loc[feature_df.index].copy()
df_with_clusters["cluster"] = feature_df["cluster"]

print(
    df_with_clusters[
        ["comic_name", "issue_title", "publish_year", "cluster"]
    ].head(10)
)

df_with_clusters.to_csv("comics_with_clusters_v1.csv", index=False)
print("Saved comics_with_clusters_v1.csv")

for c in [2, 0]:
    print(f"\n--- Cluster {c} summary ---")
    print(feature_df[feature_df["cluster"] == c].mean())
    print("Count:", (feature_df["cluster"] == c).sum())


