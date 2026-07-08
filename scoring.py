import pandas as pd

# Load artifact once
df = pd.read_csv("comics_with_clusters_v1.csv")
df.columns = df.columns.str.strip()

CLUSTER_NAMES = {
    0: "Mass-market / disposable-ish",
    2: "Legacy-dense / collectible-ish",
}


def normalize_query(query: str) -> str:
    return (query or "").strip()


def label_from_cluster(cluster_id: int) -> str:
    if cluster_id == 2:
        return "High (proxy)"
    if cluster_id == 0:
        return "Low (proxy)"
    return "Medium (proxy)"


def find_matches(query: str):
    hits = df[df["comic_name"].astype(str).str.contains(query, case=False, na=False)]

    if len(hits) == 0:
        hits = df[df["issue_title"].astype(str).str.contains(query, case=False, na=False)]

    return hits


def choose_best_match(hits):
    if len(hits) == 0:
        return None

    return hits.iloc[0].to_dict()


def build_result(row: dict) -> dict:
    cluster_id = int(row.get("cluster", -1))

    return {
        "comic_name": row.get("comic_name", ""),
        "issue_title": row.get("issue_title", ""),
        "publish_date": row.get("publish_date", ""),
        "cluster": cluster_id,
        "cluster_name": CLUSTER_NAMES.get(cluster_id, "Unlabeled cluster"),
        "label": label_from_cluster(cluster_id),
    }


def score_comic(query: str):
    query = normalize_query(query)

    if not query:
        return None

    hits = find_matches(query)
    row = choose_best_match(hits)

    if row is None:
        return None

    return build_result(row)