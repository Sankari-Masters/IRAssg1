# Information Retrieval System - Assignment 1

## Overview
This is a comprehensive Information Retrieval system built with Streamlit that demonstrates various IR techniques including text preprocessing, indexing, phrase query processing, dictionary search, and tolerant retrieval.

## Features

### 1. Document Upload
- Upload multiple text documents (TXT, CSV)
- View uploaded documents
- Built-in sample dataset for testing

### 2. Text Preprocessing
- Tokenization
- Lowercasing
- Stop word removal
- Hyphen handling
- Stemming (Porter Stemmer)
- Lemmatization (WordNet Lemmatizer)
- Comparison between stemming and lemmatization

### 3. Inverted Index
- Build and search inverted index
- View index structure
- Term frequency and document frequency

### 4. Phrase Query Processing
- Biword Index for phrase queries
- Positional Index for phrase queries
- Comparison of both approaches
- Analysis of false positives

### 5. Dictionary Search
- Binary Search Tree (BST) implementation
- B-Tree implementation
- Performance comparison with multiple queries
- Experimental results table

### 6. Tolerant Retrieval
- Wildcard queries using k-gram index
- Spelling correction using edit distance
- Phonetic correction using Soundex algorithm
- Analysis of tolerance to imperfect queries

### 7. Inference & Discussion
- Comprehensive analysis of all techniques
- Conclusions based on experimental results
- Limitations and future improvements

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Steps

1. Navigate to the project directory:
```bash
cd ir_assignment
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Run the Streamlit application:
```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

## Usage

### Using Sample Dataset
1. Open the application
2. Navigate to "Document Upload" page
3. Click "Load Sample Dataset" to load 5 sample documents about AI/ML topics
4. Proceed to other modules to explore different features

### Using Custom Documents
1. Navigate to "Document Upload" page
2. Upload your text documents using the file uploader
3. View uploaded documents
4. Use other modules to process and search your documents

### Running Experiments

#### Text Preprocessing
1. Go to "Text Preprocessing" page
2. Select preprocessing options
3. Click "Apply Preprocessing"
4. View results and stemming vs lemmatization comparison

#### Inverted Index
1. Go to "Inverted Index" page
2. Click "Build Inverted Index"
3. Search for specific terms
4. View index structure

#### Phrase Query Processing
1. Go to "Phrase Query Processing" page
2. Build both Biword and Positional indices
3. Enter phrase queries (e.g., "machine learning")
4. Compare results from both approaches

#### Dictionary Search
1. Go to "Dictionary Search" page
2. Click "Build Dictionary from Documents"
3. Search for individual terms
4. Run "100 Random Queries" for performance comparison
5. View experimental results table

#### Tolerant Retrieval
1. Go to "Tolerant Retrieval" page
2. Click "Build K-Gram Index"
3. Try wildcard queries (e.g., "mach*")
4. Test spelling correction with misspelled words
5. Test phonetic correction

## Assignment Requirements Coverage

### Streamlit End-to-End Workflow (1 Mark)
✅ Complete workflow through Streamlit interface
✅ Document upload and viewing
✅ Query input from frontend
✅ Preprocessing and retrieval options selection
✅ Display of intermediate and final outputs

### Text Preprocessing (1.5 Marks)
✅ Tokenization and inverse index creation
✅ Lowercasing
✅ Stop word removal
✅ Hyphen handling
✅ Stemming and Lemmatization
✅ Comparison with semantic similarity measure

### Stemming vs Lemmatization (1 Mark)
✅ Experimental comparison
✅ Vocabulary reduction analysis
✅ Retrieval quality comparison
✅ Justified conclusion

### Phrase Query Processing (1.5 Marks)
✅ Biword index implementation
✅ Positional index implementation
✅ Query results using both indices
✅ False positive analysis
✅ Accuracy comparison

### Binary Tree and B-Tree Comparison (1.5 Marks)
✅ BST implementation
✅ B-Tree implementation
✅ Performance comparison (query search time, retrieval time)
✅ Multiple query experiments
✅ Results table with inferences

### Tolerant Retrieval (1.5 Marks)
✅ Wildcard queries (k-gram index)
✅ Spelling correction (edit distance)
✅ Phonetic correction (Soundex)
✅ Experimental demonstration

### Experimental Evidence and Inference (1 Mark)
✅ Comprehensive inference section
✅ Answers to all 7 questions
✅ Experimental results summary
✅ Limitations and improvements

## Project Structure

```
ir_assignment/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── dataset/              # Optional: place custom documents here
```

## Dependencies

- streamlit==1.28.0
- pandas==2.1.0
- numpy==1.24.3
- nltk==3.8.1

## Notes for Submission

### Required Submission Components
1. ✅ Streamlit application code (app.py)
2. ✅ Dataset (use built-in sample or upload your own)
3. 📝 Report (create separately with screenshots and experimental results)
4. 📹 Demo evidence (take screenshots or screen recording)
5. ✅ README file (this file)

### Report Guidelines
Include screenshots of:
- Document upload page with sample dataset
- Text preprocessing results
- Stemming vs Lemmatization comparison
- Inverted index structure
- Phrase query comparison (Biword vs Positional)
- BST vs B-Tree performance table
- Tolerant retrieval results
- Inference and discussion section

### Demo Evidence
Take screenshots or record a short video showing:
- Uploading documents
- Running preprocessing
- Building indices
- Performing searches
- Comparing different techniques

## Troubleshooting

### NLTK Data Download
If you encounter NLTK data errors, the application automatically downloads required data. If issues persist:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
```

### Port Already in Use
If port 8501 is already in use, Streamlit will automatically try the next available port (8502, 8503, etc.)

### Memory Issues
For large datasets, consider:
- Processing fewer documents at a time
- Using a machine with more RAM
- Implementing disk-based indexing

## Academic Integrity
This project is submitted as part of the Information Retrieval course assignment. Ensure you:
- Run your own experiments
- Document your specific findings
- Take your own screenshots
- Write your own inferences based on your results

## Contact
For assignment-related queries, use the Taxila Discussion Forum after checking existing queries.

---
**Course:** Information Retrieval (Merged - AIMLCZG537/DSECLZG537)  
**Assignment:** 1  
**Work Integrated Learning Programmes Division**  
**BITS Pilani**
