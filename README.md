# Semantic Review Search & Topic Analysis

Creates OpenAI text embeddings for a small fictional e-commerce review dataset, supports nearest-review search, and groups the vectors with K-means. A PCA chart and review-to-topic CSV are written to `outputs/`.

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

Embedding calls require an OpenAI API key and may incur charges. The bundled synthetic reviews are for demonstration only. K-means topics are exploratory labels, not validated business conclusions. The script compares candidate cluster counts using silhouette score and reports the selected score.
