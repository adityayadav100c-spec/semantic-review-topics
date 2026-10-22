# Methodology and Data Card

## Inputs

- `data/reviews.csv` is the small hand-written fictional example set.
- `data/generated_reviews.csv` contains 600 deterministic generated review
  strings across six seeded themes. Every row includes `is_synthetic=true`.
- `generate_reviews.py` recreates or scales the generated sample. Its seed is
  configurable; records are not real customer feedback.

## Analysis flow

1. Validate review text and create OpenAI `text-embedding-3-small` vectors.
2. L2-normalize vectors so dot product is cosine similarity.
3. Fit K-means for each requested candidate `k` and select the highest
   silhouette score.
4. Show candidate scores, per-topic size, representative examples, and
   unigram/bigram keywords for human interpretation.
5. Project vectors to two PCA dimensions for a qualitative plot.
6. If generated source labels exist, compare clusters to those labels with
   Adjusted Rand Index and Normalized Mutual Information.
7. Optionally rank nearest reviews for a free-text query.

## Interpreting the metrics

Silhouette score measures geometric separation in the chosen embedding space;
it does not establish that the selected grouping is meaningful to a business.
Synthetic-label alignment only checks recovery of the generator's six
categories. PCA is a lossy projection and the displayed distance is not the
full high-dimensional semantic distance.

Do not use these generated examples to describe customers, estimate sentiment
prevalence, or make product decisions. For real deployment, verify consent and
retention requirements, remove identifiers, sample and label representative
reviews, and assess cluster stability across embedding/model revisions.
