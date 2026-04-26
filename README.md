# Human vs. AI Text Stylometrics
### A Corpus Analysis of the HC3 Dataset

**DS 5001 Text as Data — Final Project**

---

## Overview

This project applies a full text analytic pipeline to the [HC3 (Human ChatGPT Comparison Corpus)](https://huggingface.co/datasets/Hello-SimpleAI/HC3) — a dataset of 24,322 questions paired with both human-written and ChatGPT-generated answers across five domains: Reddit ELI5, open QA, medicine, finance, and Wikipedia CS/AI.

The goal is to characterize the stylometric, topical, and affective differences between human and AI writing using unsupervised methods alone — without any labeled training signal.

---

## Key Findings



---

## Repo Structure

```
human-ai-text-stylometrics/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   ├── 01_data_loading.ipynb     # Load HC3, EDA, OHCO structure, sampling
│   ├── 02_corpus_vocab.ipynb     # Build LIB, CORPUS, VOCAB tables
│   ├── 03_bow_tfidf.ipynb        # Build BOW, DTM, TFIDF, TFIDF_L2
│   ├── 04_pca.ipynb              # PCA components, DCM, loadings, visualizations
│   ├── 05_lda.ipynb              # LDA topic model, THETA, PHI, LDA+PCA plot
│   ├── 06_sentiment.ipynb        # Sentiment analysis: VOCAB_SENT, BOW_SENT, DOC_SENT
│   ├── 07_word2vec.ipynb         # Word2Vec embeddings, VOCAB_W2V, t-SNE plot
│   └── 08_riffs.ipynb            # Combined visualizations (riffs)
│
├── images/                       # All PNG figures for FinalProject report
│   ├── pca_docs_pc0_pc1.png
│   ├── pca_loadings_pc0_pc1.png
│   ├── pca_docs_pc2_pc3.png
│   ├── pca_loadings_pc2_pc3.png
│   ├── lda_pca.png
│   ├── sentiment_plot.png
│   ├── w2v_tsne.png
│   ├── riff1_sentiment_heatmap.png
│   ├── riff2_length_boxplot.png
│   └── riff3_topic_heatmap.png
│
└── data/
    └── README.md                 # Points to UVA Box for large data files
```

---

## How to Run

Run notebooks in order (01 → 08). Each notebook reads from `data/` and writes back to `data/` and `images/`.

```bash
pip install -r requirements.txt
jupyter notebook
```

---

## Data

All data files (CSV) are stored on UVA Box due to file size constraints. See `data/README.md` for links.

Raw data is loaded directly from HuggingFace in notebook 01 — no manual download needed.

---

## Dependencies

See `requirements.txt`. Key libraries: `pandas`, `numpy`, `nltk`, `scikit-learn`, `gensim`, `matplotlib`, `seaborn`, `requests`.
