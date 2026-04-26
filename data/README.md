# Data Files

All data files are stored on UVA Box due to GitHub file size limits.

**UVA Box folder:** UVA BOX FOLDER URL

## Files

| File | Description | Size (approx) |
|---|---|---|
| `docs_sampled.csv` | Sampled document list (LIB source)
| `hc3_LIB.csv` | LIB table — one row per document
| `hc3_CORPUS.csv` | CORPUS table — one row per token
| `hc3_VOCAB.csv` | VOCAB table — one row per unique term
| `hc3_BOW.csv` | Bag-of-words with TFIDF
| `hc3_DTM.csv` | Document-term count matrix
| `hc3_TFIDF.csv` | TF-IDF matrix
| `hc3_TFIDF_L2.csv` | Reduced and L2-normalized TF-IDF
| `hc3_PCA_DCM.csv` | PCA document-component matrix
| `hc3_PCA_LOADINGS.csv` | PCA component-term loadings
| `hc3_LDA_THETA.csv` | LDA document-topic matrix
| `hc3_LDA_PHI.csv` | LDA topic-term matrix
| `hc3_VOCAB_SENT.csv` | VOCAB with sentiment scores
| `hc3_BOW_SENT.csv` | BOW with sentiment scores
| `hc3_DOC_SENT.csv` | Document-level sentiment
| `hc3_VOCAB_W2V.csv` | Word2Vec embeddings per term

## Raw Data

The raw HC3 data is loaded directly from HuggingFace in `notebooks/01_data_loading.ipynb` — no manual download needed.

**Source:** https://huggingface.co/datasets/Hello-SimpleAI/HC3
