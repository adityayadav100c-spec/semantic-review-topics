import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)

ROOT = Path(__file__).parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", help="Find the closest sample review")
    parser.add_argument("--max-clusters", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    if args.max_clusters < 2 or args.top_k < 1:
        raise ValueError("max-clusters must be at least 2 and top-k must be positive.")
    load_dotenv(ROOT / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Missing OPENAI_API_KEY; add it to .env.")

    generated = ROOT / "data" / "generated_reviews.csv"
    source = generated if generated.exists() else ROOT / "data" / "reviews.csv"
    reviews = pd.read_csv(source)
    if reviews.empty or reviews["review"].isna().any():
        raise ValueError(f"Review data is empty or has missing review text: {source}")
    client = OpenAI()
    texts = reviews["review"].tolist()
    vectors = client.embeddings.create(
        model="text-embedding-3-small", input=texts
    ).data
    embeddings = np.asarray([item.embedding for item in vectors], dtype=np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.maximum(norms, 1e-12)

    max_k = min(args.max_clusters, len(reviews) - 1)
    if max_k < 2:
        raise ValueError("At least three reviews are required to compare clusters.")
    candidates = range(2, max_k + 1)
    scored = []
    for k in candidates:
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(embeddings)
        scored.append((silhouette_score(embeddings, labels), k, labels))
    score, best_k, labels = max(scored, key=lambda result: result[0])
    reviews["topic"] = labels
    vectorizer = CountVectorizer(
        stop_words="english", ngram_range=(1, 2), max_features=5000
    )
    word_matrix = vectorizer.fit_transform(texts)
    words = np.asarray(vectorizer.get_feature_names_out())

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
    silhouette_values = [
        {"k": k, "score": float(candidate_score)}
        for candidate_score, k, _ in scored
    ]
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot(
        [item["k"] for item in silhouette_values],
        [item["score"] for item in silhouette_values],
        marker="o",
    )
    axis.axvline(best_k, linestyle="--", color="gray", label=f"selected k={best_k}")
    axis.set_xlabel("Number of clusters (k)")
    axis.set_ylabel("Silhouette score")
    axis.set_title("Cluster-count comparison")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output / "silhouette_scores.png", dpi=160)
    plt.close(figure)

    if "theme" in reviews:
        categories = sorted(reviews["theme"].dropna().unique())
        colors = plt.get_cmap("tab10", max(len(categories), 1))
        figure, axis = plt.subplots(figsize=(8, 5))
        for color_index, theme in enumerate(categories):
            mask = reviews["theme"].to_numpy() == theme
            axis.scatter(
                projected[mask, 0],
                projected[mask, 1],
                label=theme,
                color=colors(color_index),
                alpha=0.75,
                s=20,
            )
        axis.set_title("PCA projection colored by generated source theme")
        axis.set_xlabel("Principal component 1")
        axis.set_ylabel("Principal component 2")
        axis.legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
        figure.tight_layout()
        figure.savefig(output / "themes.png", dpi=160, bbox_inches="tight")
        plt.close(figure)

    topic_details = []
    print(f"Reviews analyzed: {len(reviews)} from {source.name}")
    print(f"Selected clusters: {best_k}; silhouette score: {score:.3f}")
    print("\nTopic profiles and representative examples:")
    for topic, group in reviews.groupby("topic"):
        indices = group.index.to_numpy()
        mean_counts = np.asarray(word_matrix[indices].mean(axis=0)).ravel()
        top_words = words[np.argsort(mean_counts)[-6:][::-1]].tolist()
        detail = {
            "topic": int(topic),
            "review_count": len(group),
            "keywords": top_words,
            "examples": group["review"].head(3).tolist(),
        }
        topic_details.append(detail)
        print(
            f"- Topic {topic} ({len(group)} reviews), "
            f"keywords={', '.join(top_words)}"
        )
        for example in detail["examples"]:
            print(f"    {example}")

    evaluation = {
        "source_file": source.name,
        "review_count": len(reviews),
        "embedding_model": "text-embedding-3-small",
        "clustering": "KMeans with silhouette-based k selection",
        "selected_k": best_k,
        "silhouette_score": float(score),
        "candidate_scores": [
            {"k": k, "silhouette_score": float(candidate_score)}
            for candidate_score, k, _ in scored
        ],
        "topics": topic_details,
        "synthetic_theme_alignment": (
            {
                "adjusted_rand_index": float(
                    adjusted_rand_score(reviews["theme"], labels)
                ),
                "normalized_mutual_information": float(
                    normalized_mutual_info_score(reviews["theme"], labels)
                ),
                "interpretation": (
                    "Agreement with generated source labels only; it does not "
                    "measure the usefulness of themes in real customer feedback."
                ),
            }
            if "theme" in reviews.columns
            else None
        ),
        "limitations": [
            "Silhouette score measures geometric separation, not business usefulness.",
            "Topic keywords and representative examples require human interpretation.",
            "Generated review data is synthetic and cannot support customer claims.",
        ],
    }
    (output / "analysis.json").write_text(
        json.dumps(evaluation, indent=2), encoding="utf-8"
    )
    print(f"\nWrote topic CSV, PCA chart, and analysis JSON under {output}")

    if args.query:
        query_vector = client.embeddings.create(
            model="text-embedding-3-small", input=[args.query]
        ).data[0].embedding
        query_vector = np.asarray(query_vector, dtype=np.float32)
        query_vector /= max(float(np.linalg.norm(query_vector)), 1e-12)
        similarities = embeddings @ query_vector
        best_matches = np.argsort(similarities)[-args.top_k:][::-1]
        print("\nNearest reviews (cosine similarity):")
        for index in best_matches:
            print(f"- {similarities[index]:.3f}: {reviews.iloc[index]['review']}")


if __name__ == "__main__":
    main()
