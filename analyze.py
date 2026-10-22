import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

ROOT = Path(__file__).parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", help="Find the closest sample review")
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Missing OPENAI_API_KEY; add it to .env.")

    reviews = pd.read_csv(ROOT / "data" / "reviews.csv")
    client = OpenAI()
    texts = reviews["review"].tolist()
    vectors = client.embeddings.create(
        model="text-embedding-3-small", input=texts
    ).data
    embeddings = np.asarray([item.embedding for item in vectors], dtype=np.float32)

    max_k = min(6, len(reviews) - 1)
    candidates = range(2, max_k + 1)
    scored = []
    for k in candidates:
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(embeddings)
        scored.append((silhouette_score(embeddings, labels), k, labels))
    score, best_k, labels = max(scored, key=lambda result: result[0])
    reviews["topic"] = labels

    output = ROOT / "outputs"
    output.mkdir(exist_ok=True)
    reviews.to_csv(output / "reviews_with_topics.csv", index=False)
    projected = PCA(n_components=2, random_state=42).fit_transform(embeddings)
    plt.figure(figsize=(8, 5))
    points = plt.scatter(projected[:, 0], projected[:, 1], c=labels, cmap="tab10")
    plt.colorbar(points, label="K-means topic")
    plt.title(f"Review embeddings projected with PCA (k={best_k})")
    plt.xlabel("Principal component 1")
    plt.ylabel("Principal component 2")
    plt.tight_layout()
    plt.savefig(output / "topics.png", dpi=160)
    plt.close()

    print(f"Reviews analyzed: {len(reviews)} (synthetic sample)")
    print(f"Selected clusters: {best_k}; silhouette score: {score:.3f}")
    print("\nRepresentative review per cluster:")
    for topic, group in reviews.groupby("topic"):
        print(f"- Topic {topic}: {group.iloc[0]['review']}")
    print(f"\nWrote {output / 'reviews_with_topics.csv'} and {output / 'topics.png'}")

    if args.query:
        query_vector = client.embeddings.create(
            model="text-embedding-3-small", input=[args.query]
        ).data[0].embedding
        similarities = embeddings @ np.asarray(query_vector, dtype=np.float32)
        best = int(np.argmax(similarities))
        print(f"\nClosest review (cosine similarity {similarities[best]:.3f}):")
        print(reviews.iloc[best]["review"])


if __name__ == "__main__":
    main()
