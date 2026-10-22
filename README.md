# Semantic Review Search & Topic Analysis

An embedding workflow for exploring fictional e-commerce feedback. It compares
candidate K-means cluster counts, ranks semantically similar reviews, extracts
interpretable topic keywords, and writes an embedding projection and analysis
artifacts. OpenAI embeddings are the semantic representation; K-means topics
remain exploratory and require human review.

## Run

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
# Add your OpenAI API key to .env
python analyze.py
python analyze.py --query "Is the battery life good?"
```

Embedding calls require an OpenAI API key and may incur charges.

To create a larger reproducible dataset with 100 fictional examples per theme,
then analyze it:

```powershell
python generate_reviews.py --per-topic 100
python analyze.py --max-clusters 10 --top-k 5
```

The generated dataset is explicitly synthetic. K-means topic labels are not
validated business conclusions; review the cluster examples before assigning
business meaning.

The checked-in generated sample has 600 rows, six synthetic themes, and a
`is_synthetic` marker on every row. When those generated labels are present,
the report also includes Adjusted Rand Index and Normalized Mutual Information
for comparison with the clusters. See [`docs/METHODOLOGY.md`](./docs/METHODOLOGY.md)
for the data card and metric interpretation.

Outputs include `topics.png`, a generated-theme PCA view when labels are
available, a silhouette-by-cluster-count chart, topic assignments, and a JSON
report with cluster examples and limitations.

Run deterministic, dependency-light generator checks with
`python -m unittest discover -s tests`.

GitHub Actions also regenerates a small fixture and runs compilation/tests.
