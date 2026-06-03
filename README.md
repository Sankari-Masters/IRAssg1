# Information Retrieval System - Assignment 1

## Overview
This Streamlit application implements the end-to-end Information Retrieval workflow required for Assignment 1. The complete workflow runs from the front end: users can load or upload documents, inspect preprocessing output, build indexes, run queries, compare phrase-query methods, compare BST and B-Tree dictionary search, test tolerant retrieval, and collect inferences for the report.

## Features

### 1. Document Upload
- Upload multiple TXT documents or CSV datasets.
- Load the included `dataset/` document collection.
- View all loaded documents in the Streamlit UI.

### 2. Text Preprocessing
- Tokenization.
- Lowercasing.
- Stop-word removal.
- Hyphen handling.
- Stemming with Porter stemmer.
- Lemmatization with WordNet when available, with a fallback so the app still runs.
- Inverted index creation from processed tokens.
- Stemming vs lemmatization comparison using cosine similarity over query and document term vectors.

### 3. Inverted Index
- Build an inverted index from the selected preprocessing configuration.
- Search terms from the UI.
- Display document frequency, term frequency, postings lists, and ranked result rows.

### 4. Phrase Query Processing
- Build a biword index.
- Build a positional index.
- Search phrase queries using both indexes.
- Show biword-only false positives for longer phrases.
- Explain why the positional index is more accurate.

### 5. Dictionary Search
- Build a balanced Binary Search Tree over the vocabulary.
- Build a B-Tree over the same vocabulary.
- Compare query search time, postings retrieval time, and total time.
- Generate an experimental results table over multiple queries.

### 6. Tolerant Retrieval
- Wildcard queries using a k-gram index.
- Spelling correction using edit distance.
- Phonetic correction using Soundex.
- Display matching terms and retrieved documents.

### 7. Inference and Discussion
- Dynamic answers for the compulsory inference questions after experiments are run.
- Static limitations and possible improvements for report writing.

## Installation

### Prerequisites
- Python 3.8 or higher.
- pip.

### Steps

1. Navigate to the project directory:
```bash
cd "/Users/venkateshbhaskara/Documents/** BITS PILANI **/Education/Sem2/IR Assignment1/IRAssg1-main"
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the app:
```bash
streamlit run app.py
```

Streamlit opens the application in a browser, usually at `http://localhost:8501`.

## Usage

### Recommended Assignment Demo Flow
1. Open the app and go to `Document Upload`.
2. Click `Load sample dataset`.
3. Go to `Text Preprocessing`, apply preprocessing, and run the stemming-vs-lemmatization comparison.
4. Go to `Inverted Index`, build the index, and search for a term such as `retrieval`.
5. Go to `Phrase Query Processing`, build both indexes, and try `machine learning algorithms` to inspect the biword vs positional comparison.
6. Go to `Dictionary Search`, build dictionary trees, search a term, and run the experimental table.
7. Go to `Tolerant Retrieval`, build tolerant indexes, and test `retriev*`, `retrival`, and `lern`.
8. Go to `Inference & Discussion` and use the generated observations for the report.

## Project Structure

```text
IRAssg1-main/
├── app.py
├── requirements.txt
├── README.md
└── dataset/
    ├── doc1.txt
    ├── doc2.txt
    ├── doc3.txt
    ├── doc4.txt
    ├── doc5.txt
    ├── doc6.txt
    └── test.txt
```

`test.txt` is ignored by the app when empty. The added `doc6.txt` is an edge-case document that helps demonstrate a longer-phrase biword false positive.

## Dependencies

- streamlit
- pandas
- numpy
- nltk

## Assignment Coverage

| Requirement | Implemented In |
| --- | --- |
| Streamlit end-to-end workflow | All modules run from the Streamlit UI |
| Text preprocessing | `Text Preprocessing` page |
| Stemming vs lemmatization | Cosine-similarity comparison in `Text Preprocessing` |
| Phrase query using biword and positional indexes | `Phrase Query Processing` page |
| Binary Tree and B-Tree comparison | `Dictionary Search (BST & B-Tree)` page |
| Tolerant retrieval | `Tolerant Retrieval` page |
| Experimental evidence and inference | Dynamic tables plus `Inference & Discussion` page |

## Submission Notes

Submit:
- `app.py`.
- `requirements.txt`.
- `README.md`.
- The `dataset/` folder.
- Report with screenshots from each module.
- Experimental tables and inferences generated from your own app run.
- Demo evidence such as screenshots or a short screen recording.

## NLTK Data
The app does not require a manual NLTK download to start. If the local machine already has NLTK stopwords and WordNet data, the app uses them. If they are missing, it falls back to a compact built-in stop-word list and simple rule-based lemmatization so the assignment workflow remains executable.
