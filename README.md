# Human vs. AI Text Stylometrics
### A Corpus Analysis of the HC3 Dataset

**DS 5001 Text as Data — Final Project**

---

## Overview

This project applies a full text analytic pipeline to the [HC3 (Human ChatGPT Comparison Corpus)](https://huggingface.co/datasets/Hello-SimpleAI/HC3) — a dataset of 24,322 questions paired with both human-written and ChatGPT-generated answers across five domains: Reddit ELI5, open QA, medicine, finance, and Wikipedia CS/AI.

The goal is to characterize the stylometric, topical, and affective differences between human and AI writing using unsupervised methods alone — without any labeled training signal.

---

## Key Findings

This project ran a full unsupervised text-analytic pipeline on 10,258 documents from the HC3 corpus, covering paired human and ChatGPT answers to the same questions across five domains: finance, medicine, open_qa, reddit_eli5, and wiki_csai. The methods included BOW/TFIDF vectorization, PCA, LDA topic modeling, NRC sentiment analysis, and Word2Vec embeddings. Across all of them, consistent patterns showed up that separate human from AI-generated writing.

Length stood out as the clearest signal. ChatGPT answers ran longer than human answers in every domain, with the gap widest in medicine (medians around 190 vs 70 words) and open_qa (around 115 vs 35 words). The difference goes beyond word count, ChatGPT responses tend to hit multiple angles and maintain a structured format regardless of what the question actually needs. Human Reddit answers were the most variable, with a low median but a long tail of very detailed posts, reflecting how much individual effort and context shape human responses.

PCA put domain front and center as the main axis of variation. PC0 split the corpus along a finance-versus-science line, with terms like stock, money, and company pulling left and water, air, and body pulling right. PC1 then separated documents within the science end, with ChatGPT answers sitting below zero toward physical and biological vocabulary while human answers spread upward. Subject matter explained more of the variance than author type, but the author signal still came through without any labeled data.

ChatGPT scored higher on sentiment in four of the five domains. Overall means came out at 0.116 for ChatGPT and 0.065 for human. Medicine was the exception, where human answers scored 0.058 against ChatGPT's 0.043, likely because human medical writing carries more personal and supportive language. Wiki_csai had the highest scores for both groups, which fits the generally positive framing of writing about technology.

LDA showed that human and ChatGPT answers emphasize different topics. Human documents loaded heavily on T05 (General Discussion, mean 0.238), a topic built around everyday conversational words like get, people, and time, while ChatGPT led on T06 (Natural Science, mean 0.208), centered on words like water, body, and air. ChatGPT spread its weight more evenly across all ten topics while human answers concentrated more sharply on the general discussion topic, pointing to human writers defaulting to a broader conversational register while ChatGPT gravitates toward structured domain content.

Word2Vec embeddings reflected real semantic structure. The t-SNE plot showed verbs like know, understand, and see grouping together in the upper-center region, pulled close by their shared role in explanatory sentence frames. Nouns spread across the right side with some domain clustering visible. POS groups showed enough local separation to confirm the embeddings picked up genuine grammatical and semantic patterns.

Taken together the results show that human and AI writing differ across multiple dimensions at once rather than along any single line. Length, sentiment, topic focus, and vocabulary all shift in consistent directions across all five domains, which points to these being features of how the model generates text rather than quirks of any one subject area. Picking up on these differences reliably likely requires looking at several signals together rather than relying on any one of them alone.

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
