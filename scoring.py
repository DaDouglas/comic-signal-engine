import pandas as pd

# load artifact once
df = pd.read_csv("comics_with_clusters_v1.csv")
df.columns = df.columns.str.strip()

# Optional: label clusters (you can refine these once you inspect summaries)
CLUSTER_NAMES = {
    0: "Mass-market / disposable-ish",
    2: "Legacy-dense / collectible-ish",
}

def label_from_cluster(cluster_id: int) -> str:
    # v1 proxy output (not a real market probability yet)
    if cluster_id == 2:
        return "High (proxy)"
    if cluster_id == 0:
        return "Low (proxy)"
    return "Medium (proxy)"

def score_comic(query: str):
    query = (query or "").strip()
    if not query:
        return None

    # search comic_name first
    hits = df[df["comic_name"].astype(str).str.contains(query, case=False, na=False)]

    # fallback to issue_title
    if len(hits) == 0:
        hits = df[df["issue_title"].astype(str).str.contains(query, case=False, na=False)]

    if len(hits) == 0:
        return None

    row = hits.iloc[0].to_dict()
    cluster_id = int(row.get("cluster", -1))

    return {
        "comic_name": row.get("comic_name", ""),
        "issue_title": row.get("issue_title", ""),
        "publish_date": row.get("publish_date", ""),
        "cluster": cluster_id,
        "cluster_name": CLUSTER_NAMES.get(cluster_id, "Unlabeled cluster"),
        "label": label_from_cluster(cluster_id),
    }