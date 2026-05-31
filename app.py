import streamlit as st
import pandas as pd
import numpy as np
import re
import time
from collections import defaultdict, Counter
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.metrics.distance import edit_distance
import string

# Download required NLTK data
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        nltk.download('averaged_perceptron_tagger')

download_nltk_data()

# Initialize session state
if 'documents' not in st.session_state:
    st.session_state.documents = {}
if 'preprocessed_docs' not in st.session_state:
    st.session_state.preprocessed_docs = {}
if 'inverted_index' not in st.session_state:
    st.session_state.inverted_index = {}
if 'biword_index' not in st.session_state:
    st.session_state.biword_index = {}
if 'positional_index' not in st.session_state:
    st.session_state.positional_index = {}
if 'bst' not in st.session_state:
    st.session_state.bst = None
if 'btree' not in st.session_state:
    st.session_state.btree = None
if 'kgram_index' not in st.session_state:
    st.session_state.kgram_index = {}

# Initialize experimental results for dynamic inferences
if 'exp_results' not in st.session_state:
    st.session_state.exp_results = {
        'stem_vs_lem': None,
        'phrase_query': None,
        'bst_vs_btree': None,
        'tolerant_retrieval': None
    }

# Page configuration
st.set_page_config(page_title="Information Retrieval System", layout="wide")

st.title("📚 Information Retrieval System")
st.markdown("---")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Module", [
    "Document Upload",
    "Text Preprocessing",
    "Inverted Index",
    "Phrase Query Processing",
    "Dictionary Search (BST & B-Tree)",
    "Tolerant Retrieval",
    "Inference & Discussion"
])

# ==================== DOCUMENT UPLOAD ====================
if page == "Document Upload":
    st.header("📄 Document Upload")
    
    # Upload text files
    uploaded_files = st.file_uploader(
        "Upload text documents",
        type=['txt', 'csv'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        for file in uploaded_files:
            content = file.read().decode('utf-8')
            st.session_state.documents[file.name] = content
        
        st.success(f"Successfully uploaded {len(uploaded_files)} document(s)")
    
    # Display uploaded documents
    if st.session_state.documents:
        st.subheader("Uploaded Documents")
        for doc_name, content in st.session_state.documents.items():
            with st.expander(f"📄 {doc_name}"):
                st.text_area("Content", content, height=200, key=f"doc_{doc_name}")
    
    # Sample dataset option
    st.subheader("Or use Sample Dataset")
    if st.button("Load Sample Dataset"):
        sample_docs = {
            "doc1.txt": """Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data. 
            These algorithms improve automatically through experience and are used in a wide variety of applications including email filtering, 
            computer vision, and recommendation systems. Deep learning is a specialized branch of machine learning that uses neural networks 
            with multiple layers to model complex patterns in data.""",
            
            "doc2.txt": """Natural language processing (NLP) is a field of artificial intelligence that gives computers the ability to understand 
            and interpret human language. NLP techniques include tokenization, part-of-speech tagging, named entity recognition, and sentiment analysis. 
            Modern NLP systems use deep learning models like transformers to achieve state-of-the-art results on various language tasks.""",
            
            "doc3.txt": """Information retrieval is the science of searching for information in documents, searching for documents themselves, 
            and also searching for metadata that describe documents. Search engines use information retrieval techniques to index web pages 
            and provide relevant results to user queries. Key concepts include inverted indexes, term frequency-inverse document frequency (TF-IDF), 
            and vector space models.""",
            
            "doc4.txt": """Data mining is the process of discovering patterns in large data sets involving methods at the intersection of 
            machine learning, statistics, and database systems. It includes data cleaning, data integration, data selection, data transformation, 
            data mining, pattern evaluation, and knowledge presentation. Applications include market basket analysis, customer segmentation, 
            and fraud detection.""",
            
            "doc5.txt": """Computer vision is a field of artificial intelligence that enables computers to interpret and understand the visual world. 
            Using digital images from cameras and videos and deep learning models, machines can accurately identify and classify objects. 
            Applications include facial recognition, autonomous vehicles, medical image analysis, and augmented reality."""
        }
        st.session_state.documents = sample_docs
        st.success("Sample dataset loaded successfully!")
        
        for doc_name, content in st.session_state.documents.items():
            with st.expander(f"📄 {doc_name}"):
                st.text_area("Content", content, height=150, key=f"sample_{doc_name}")

# ==================== TEXT PREPROCESSING ====================
elif page == "Text Preprocessing":
    st.header("🔧 Text Preprocessing")
    
    if not st.session_state.documents:
        st.warning("Please upload documents first!")
    else:
        # Select preprocessing options
        st.subheader("Preprocessing Options")
        col1, col2 = st.columns(2)
        
        with col1:
            use_lowercase = st.checkbox("Lowercasing", value=True)
            remove_stopwords = st.checkbox("Stop Word Removal", value=True)
            handle_hyphens = st.checkbox("Hyphen Handling", value=True)
        
        with col2:
            use_stemming = st.checkbox("Stemming (Porter)", value=False)
            use_lemmatization = st.checkbox("Lemmatization", value=False)
        
        if st.button("Apply Preprocessing"):
            # Initialize preprocessors
            stemmer = PorterStemmer()
            lemmatizer = WordNetLemmatizer()
            stop_words = set(stopwords.words('english'))
            
            preprocessed_results = {}
            
            for doc_name, content in st.session_state.documents.items():
                original_tokens = word_tokenize(content)
                
                # Step 1: Lowercasing
                if use_lowercase:
                    tokens = [token.lower() for token in original_tokens]
                else:
                    tokens = original_tokens.copy()
                
                # Step 2: Hyphen handling
                if handle_hyphens:
                    tokens = [token.replace('-', ' ') if '-' in token else token for token in tokens]
                    tokens = [word for token in tokens for word in token.split()]
                
                # Step 3: Stop word removal
                if remove_stopwords:
                    tokens = [token for token in tokens if token.lower() not in stop_words and token not in string.punctuation]
                
                # Step 4: Stemming or Lemmatization
                if use_stemming:
                    tokens = [stemmer.stem(token) for token in tokens]
                elif use_lemmatization:
                    tokens = [lemmatizer.lemmatize(token) for token in tokens]
                
                preprocessed_results[doc_name] = {
                    'original': original_tokens,
                    'preprocessed': tokens
                }
            
            st.session_state.preprocessed_docs = preprocessed_results
            st.success("Preprocessing completed!")
        
        # Display preprocessing results
        if st.session_state.preprocessed_docs:
            st.subheader("📊 Tokenization Output")
            
            for doc_name, result in st.session_state.preprocessed_docs.items():
                with st.expander(f"📄 {doc_name}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Original Tokens:**")
                        st.write(result['original'][:50])
                        st.write(f"Total tokens: {len(result['original'])}")
                    
                    with col2:
                        st.markdown("**Preprocessed Tokens:**")
                        st.write(result['preprocessed'][:50])
                        st.write(f"Total tokens: {len(result['preprocessed'])}")
        
        # Inverted Index Creation in Preprocessing
        if st.session_state.preprocessed_docs:
            st.markdown("---")
            st.subheader("📑 Inverted Index Creation")
            
            if st.button("Generate Inverted Index from Preprocessed Data"):
                stemmer = PorterStemmer()
                lemmatizer = WordNetLemmatizer()
                stop_words = set(stopwords.words('english'))
                
                inverted_index = defaultdict(list)
                
                for doc_id, (doc_name, content) in enumerate(st.session_state.documents.items()):
                    # Use the same preprocessing as selected
                    tokens = word_tokenize(content)
                    
                    if use_lowercase:
                        tokens = [token.lower() for token in tokens]
                    
                    if handle_hyphens:
                        tokens = [token.replace('-', ' ') if '-' in token else token for token in tokens]
                        tokens = [word for token in tokens for word in token.split()]
                    
                    if remove_stopwords:
                        tokens = [token for token in tokens if token.lower() not in stop_words and token not in string.punctuation]
                    
                    if use_stemming:
                        tokens = [stemmer.stem(token) for token in tokens]
                    elif use_lemmatization:
                        tokens = [lemmatizer.lemmatize(token) for token in tokens]
                    
                    # Count term frequency
                    term_freq = Counter(tokens)
                    
                    for term, freq in term_freq.items():
                        inverted_index[term].append((doc_id, freq))
                
                st.session_state.inverted_index = dict(inverted_index)
                st.success(f"Inverted index generated with {len(inverted_index)} unique terms!")
            
            if st.session_state.inverted_index:
                st.markdown("**Generated Inverted Index (First 15 terms):**")
                index_items = list(st.session_state.inverted_index.items())[:15]
                
                for term, postings in index_items:
                    with st.expander(f"🔤 {term}"):
                        st.write(f"Document Frequency: {len(postings)}")
                        st.write("Postings List:", postings)
                
                st.info("💡 For full inverted index features and search, navigate to the 'Inverted Index' module.")
        
        # Stemming vs Lemmatization Comparison
        st.subheader("📊 Stemming vs Lemmatization Comparison")
        
        # Query input for retrieval comparison
        test_query = st.text_input("Enter test query for retrieval comparison:", value="learning")
        
        if st.button("Compare Stemming and Lemmatization"):
            if not st.session_state.documents:
                st.warning("Please upload documents first!")
            else:
                stemmer = PorterStemmer()
                lemmatizer = WordNetLemmatizer()
                stop_words = set(stopwords.words('english'))
                
                comparison_results = {}
                
                for doc_name, content in st.session_state.documents.items():
                    tokens = word_tokenize(content.lower())
                    tokens = [token for token in tokens if token not in stop_words and token not in string.punctuation]
                    
                    # Apply stemming
                    stemmed = [stemmer.stem(token) for token in tokens]
                    stemmed_unique = set(stemmed)
                    
                    # Apply lemmatization
                    lemmatized = [lemmatizer.lemmatize(token) for token in tokens]
                    lemmatized_unique = set(lemmatized)
                    
                    # Calculate vocabulary reduction
                    original_vocab = len(set(tokens))
                    stemmed_vocab = len(stemmed_unique)
                    lemmatized_vocab = len(lemmatized_unique)
                    
                    comparison_results[doc_name] = {
                        'original_vocab': original_vocab,
                        'stemmed_vocab': stemmed_vocab,
                        'lemmatized_vocab': lemmatized_vocab,
                        'stemming_reduction': (original_vocab - stemmed_vocab) / original_vocab * 100,
                        'lemmatization_reduction': (original_vocab - lemmatized_vocab) / original_vocab * 100
                    }
                
                # Display comparison table
                comparison_df = pd.DataFrame(comparison_results).T
                st.dataframe(comparison_df)
                
                # Calculate average reduction
                avg_stem_reduction = comparison_df['stemming_reduction'].mean()
                avg_lem_reduction = comparison_df['lemmatization_reduction'].mean()
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Average Stemming Reduction", f"{avg_stem_reduction:.2f}%")
                with col2:
                    st.metric("Average Lemmatization Reduction", f"{avg_lem_reduction:.2f}%")
                
                st.markdown("---")
                st.subheader("🎯 Retrieval Effectiveness Comparison")
                
                # Build inverted indices for stemming and lemmatization
                stemmed_index = defaultdict(list)
                lemmatized_index = defaultdict(list)
                
                for doc_id, (doc_name, content) in enumerate(st.session_state.documents.items()):
                    tokens = word_tokenize(content.lower())
                    tokens = [token for token in tokens if token not in stop_words and token not in string.punctuation]
                    
                    # Stemmed version
                    stemmed_tokens = [stemmer.stem(token) for token in tokens]
                    stemmed_term_freq = Counter(stemmed_tokens)
                    for term, freq in stemmed_term_freq.items():
                        stemmed_index[term].append((doc_id, freq))
                    
                    # Lemmatized version
                    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
                    lemmatized_term_freq = Counter(lemmatized_tokens)
                    for term, freq in lemmatized_term_freq.items():
                        lemmatized_index[term].append((doc_id, freq))
                
                # Query using stemmed index
                query_stemmed = stemmer.stem(test_query.lower())
                stemmed_results = set()
                if query_stemmed in stemmed_index:
                    stemmed_results = set(doc_id for doc_id, _ in stemmed_index[query_stemmed])
                
                # Query using lemmatized index
                query_lemmatized = lemmatizer.lemmatize(test_query.lower())
                lemmatized_results = set()
                if query_lemmatized in lemmatized_index:
                    lemmatized_results = set(doc_id for doc_id, _ in lemmatized_index[query_lemmatized])
                
                # Display retrieval results
                retrieval_df = pd.DataFrame({
                    'Method': ['Stemming', 'Lemmatization'],
                    'Query Term': [query_stemmed, query_lemmatized],
                    'Retrieved Docs': [len(stemmed_results), len(lemmatized_results)],
                    'Document IDs': [list(stemmed_results), list(lemmatized_results)]
                })
                st.dataframe(retrieval_df)
                
                # Show which documents were retrieved
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Documents Retrieved with Stemming:**")
                    if stemmed_results:
                        for doc_id in stemmed_results:
                            doc_name = list(st.session_state.documents.keys())[doc_id]
                            st.write(f"- Doc {doc_id}: {doc_name}")
                    else:
                        st.write("No documents retrieved")
                
                with col2:
                    st.markdown("**Documents Retrieved with Lemmatization:**")
                    if lemmatized_results:
                        for doc_id in lemmatized_results:
                            doc_name = list(st.session_state.documents.keys())[doc_id]
                            st.write(f"- Doc {doc_id}: {doc_name}")
                    else:
                        st.write("No documents retrieved")
                
                # Store experimental results for dynamic inference
                st.session_state.exp_results['stem_vs_lem'] = {
                    'query': test_query,
                    'stemmed_retrieved': len(stemmed_results),
                    'lemmatized_retrieved': len(lemmatized_results),
                    'avg_stem_reduction': avg_stem_reduction,
                    'avg_lem_reduction': avg_lem_reduction,
                    'better_method': 'lemmatization' if len(lemmatized_results) >= len(stemmed_results) else 'stemming'
                }
                
                st.info(f"""
                **Retrieval Quality Analysis:**
                - Query: "{test_query}"
                - Stemming retrieved {len(stemmed_results)} documents
                - Lemmatization retrieved {len(lemmatized_results)} documents
                
                **Comparison:**
                - **Stemming** reduces vocabulary more aggressively by chopping off word endings, which can lead to non-dictionary words
                - **Lemmatization** reduces vocabulary more carefully by converting words to their dictionary form (lemma)
                - **Retrieval Effectiveness**: Lemmatization generally provides better retrieval quality because it preserves semantic meaning and reduces false matches
                - For this dataset and query "{test_query}", {'lemmatization' if len(lemmatized_results) >= len(stemmed_results) else 'stemming'} retrieved {'more or equal' if len(lemmatized_results) >= len(stemmed_results) else 'more'} documents
                - **Conclusion**: Based on document coverage, {'lemmatization' if len(lemmatized_results) >= len(stemmed_results) else 'stemming'} had {'higher' if len(lemmatized_results) > len(stemmed_results) else 'equal or higher'} document retrieval for this query.
                """)

# ==================== INVERTED INDEX ====================
elif page == "Inverted Index":
    st.header("📑 Inverted Index")
    
    if not st.session_state.documents:
        st.warning("Please upload documents first!")
    else:
        if st.button("Build Inverted Index"):
            # Initialize
            stemmer = PorterStemmer()
            lemmatizer = WordNetLemmatizer()
            stop_words = set(stopwords.words('english'))
            
            inverted_index = defaultdict(list)
            
            for doc_id, (doc_name, content) in enumerate(st.session_state.documents.items()):
                # Preprocess
                tokens = word_tokenize(content.lower())
                tokens = [token for token in tokens if token not in stop_words and token not in string.punctuation]
                tokens = [stemmer.stem(token) for token in tokens]
                
                # Count term frequency
                term_freq = Counter(tokens)
                
                for term, freq in term_freq.items():
                    inverted_index[term].append((doc_id, freq))
            
            st.session_state.inverted_index = dict(inverted_index)
            st.success(f"Inverted index built with {len(inverted_index)} unique terms!")
        
        if st.session_state.inverted_index:
            st.subheader("Inverted Index Structure")
            
            # Search functionality
            search_term = st.text_input("Search for a term:")
            
            if search_term:
                stemmer = PorterStemmer()
                search_term_stemmed = stemmer.stem(search_term.lower())
                
                if search_term_stemmed in st.session_state.inverted_index:
                    st.success(f"Term '{search_term}' found!")
                    st.write(st.session_state.inverted_index[search_term_stemmed])
                else:
                    st.warning(f"Term '{search_term}' not found in index")
            
            # Display index (sample)
            st.subheader("Index Sample (First 20 terms)")
            index_items = list(st.session_state.inverted_index.items())[:20]
            
            for term, postings in index_items:
                with st.expander(f"🔤 {term}"):
                    st.write(f"Document Frequency: {len(postings)}")
                    st.write("Postings List:", postings)

# ==================== PHRASE QUERY PROCESSING ====================
elif page == "Phrase Query Processing":
    st.header("🔍 Phrase Query Processing")
    
    if not st.session_state.documents:
        st.warning("Please upload documents first!")
    else:
        # Build indices
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Build Biword Index"):
                stemmer = PorterStemmer()
                stop_words = set(stopwords.words('english'))
                
                biword_index = defaultdict(set)
                
                for doc_id, (doc_name, content) in enumerate(st.session_state.documents.items()):
                    tokens = word_tokenize(content.lower())
                    tokens = [token for token in tokens if token not in stop_words and token not in string.punctuation]
                    tokens = [stemmer.stem(token) for token in tokens]
                    
                    # Create biwords
                    for i in range(len(tokens) - 1):
                        biword = f"{tokens[i]} {tokens[i+1]}"
                        biword_index[biword].add(doc_id)
                
                st.session_state.biword_index = dict(biword_index)
                st.success(f"Biword index built with {len(biword_index)} biwords!")
        
        with col2:
            if st.button("Build Positional Index"):
                stemmer = PorterStemmer()
                stop_words = set(stopwords.words('english'))
                
                positional_index = defaultdict(lambda: defaultdict(list))
                
                for doc_id, (doc_name, content) in enumerate(st.session_state.documents.items()):
                    tokens = word_tokenize(content.lower())
                    tokens = [token for token in tokens if token not in stop_words and token not in string.punctuation]
                    tokens = [stemmer.stem(token) for token in tokens]
                    
                    for pos, term in enumerate(tokens):
                        positional_index[term][doc_id].append(pos)
                
                st.session_state.positional_index = dict(positional_index)
                st.success(f"Positional index built with {len(positional_index)} terms!")
        
        # Phrase query search
        st.subheader("Phrase Query Search")
        query = st.text_input("Enter phrase query (e.g., 'machine learning'):")
        
        if query:
            stemmer = PorterStemmer()
            query_terms = [stemmer.stem(term.lower()) for term in query.split()]
            
            # Initialize results before column blocks
            biword_results = set()
            positional_results = set()
            
            col1, col2 = st.columns(2)
            
            # Biword search
            with col1:
                st.markdown("**Biword Index Search**")
                if st.session_state.biword_index:
                    if len(query_terms) == 2:
                        biword = f"{query_terms[0]} {query_terms[1]}"
                        if biword in st.session_state.biword_index:
                            biword_results = st.session_state.biword_index[biword]
                    
                    st.write(f"Results: {biword_results}")
                    if biword_results:
                        for doc_id in biword_results:
                            doc_name = list(st.session_state.documents.keys())[doc_id]
                            st.write(f"- {doc_name}")
                    else:
                        st.write("No results found")
                else:
                    st.write("Build Biword Index first")
            
            # Positional index search
            with col2:
                st.markdown("**Positional Index Search**")
                if st.session_state.positional_index:
                    if len(query_terms) >= 2:
                        # Get documents containing all terms
                        doc_sets = []
                        for term in query_terms:
                            if term in st.session_state.positional_index:
                                doc_sets.append(set(st.session_state.positional_index[term].keys()))
                        
                        if doc_sets:
                            common_docs = set.intersection(*doc_sets)
                            
                            # Check positions
                            for doc_id in common_docs:
                                positions = []
                                for term in query_terms:
                                    if doc_id in st.session_state.positional_index[term]:
                                        positions.append(st.session_state.positional_index[term][doc_id])
                                
                                # Check if terms appear in sequence
                                if len(positions) == len(query_terms):
                                    for pos in positions[0]:
                                        found = True
                                        for i in range(1, len(positions)):
                                            if pos + i not in positions[i]:
                                                found = False
                                                break
                                        if found:
                                            positional_results.add(doc_id)
                                            break
                    
                    st.write(f"Results: {positional_results}")
                    if positional_results:
                        for doc_id in positional_results:
                            doc_name = list(st.session_state.documents.keys())[doc_id]
                            st.write(f"- {doc_name}")
                    else:
                        st.write("No results found")
        
        # Comparison analysis
        st.subheader("📊 Comparison Analysis")
        
        if st.session_state.biword_index and st.session_state.positional_index:
            # Store phrase query experimental results
            if query:
                st.session_state.exp_results['phrase_query'] = {
                    'query': query,
                    'biword_results': len(biword_results),
                    'positional_results': len(positional_results),
                    'more_accurate': 'positional' if len(positional_results) <= len(biword_results) else 'biword'
                }
            
            # Prepare query display strings
            biword_query_str = f"Retrieved {len(biword_results)} documents for query '{query}'" if query else "N/A"
            positional_query_str = f"Retrieved {len(positional_results)} documents for query '{query}'" if query else "N/A"
            
            st.info(f"""
            **Biword Index vs Positional Index:**
            
            **Biword Index:**
            - Stores consecutive word pairs as single terms
            - Faster for 2-word phrases
            - Can give false positives for longer phrases (e.g., "machine learning algorithms" might match even if not consecutive)
            - Limited to fixed-length phrases
            - {biword_query_str}
            
            **Positional Index:**
            - Stores term positions within documents
            - More accurate for phrase queries of any length
            - Can handle proximity queries
            - Requires more storage but provides precise results
            - {positional_query_str}
            
            **Conclusion:** Positional index gives more accurate phrase query results because it stores exact position information,
            allowing verification of word adjacency and order. Biword index may give false positives when phrases span more than 2 words
            or when word order matters.
            """)

# ==================== DICTIONARY SEARCH (BST & B-TREE) ====================
elif page == "Dictionary Search (BST & B-Tree)":
    st.header("🌳 Dictionary Search using BST and B-Tree")
    
    # Define BST functions at page level for access in all button blocks
    class BSTNode:
        def __init__(self, key):
            self.key = key
            self.left = None
            self.right = None
    
    def bst_insert(root, key):
        if root is None:
            return BSTNode(key)
        if key < root.key:
            root.left = bst_insert(root.left, key)
        elif key > root.key:
            root.right = bst_insert(root.right, key)
        return root
    
    def bst_search(root, key):
        if root is None or root.key == key:
            return root
        if key < root.key:
            return bst_search(root.left, key)
        return bst_search(root.right, key)
    
    if not st.session_state.documents:
        st.warning("Please upload documents first!")
    else:
        # Build dictionary
        if st.button("Build Dictionary from Documents"):
            stemmer = PorterStemmer()
            stop_words = set(stopwords.words('english'))
            
            terms = set()
            for content in st.session_state.documents.values():
                tokens = word_tokenize(content.lower())
                tokens = [token for token in tokens if token not in stop_words and token not in string.punctuation]
                tokens = [stemmer.stem(token) for token in tokens]
                terms.update(tokens)
            
            terms = sorted(list(terms))
            
            bst_root = None
            # Shuffle terms to create a more balanced BST
            import random
            shuffled_terms = terms.copy()
            random.shuffle(shuffled_terms)
            for term in shuffled_terms:
                bst_root = bst_insert(bst_root, term)
            
            # Build B-Tree (simplified implementation)
            class BTreeNode:
                def __init__(self, leaf=False):
                    self.keys = []
                    self.children = []
                    self.leaf = leaf
            
            class BTree:
                def __init__(self, t=3):
                    self.root = BTreeNode(True)
                    self.t = t  # Minimum degree
                
                def search(self, key, node=None):
                    if node is None:
                        node = self.root
                    i = 0
                    while i < len(node.keys) and key > node.keys[i]:
                        i += 1
                    if i < len(node.keys) and key == node.keys[i]:
                        return True
                    if node.leaf:
                        return False
                    return self.search(key, node.children[i])
                
                def insert(self, key):
                    root = self.root
                    if len(root.keys) == (2 * self.t) - 1:
                        new_root = BTreeNode()
                        new_root.children.append(self.root)
                        self.split_child(new_root, 0)
                        self.root = new_root
                    self._insert_non_full(self.root, key)
                
                def _insert_non_full(self, node, key):
                    i = len(node.keys) - 1
                    if node.leaf:
                        node.keys.append(None)
                        while i >= 0 and key < node.keys[i]:
                            node.keys[i + 1] = node.keys[i]
                            i -= 1
                        node.keys[i + 1] = key
                    else:
                        while i >= 0 and key < node.keys[i]:
                            i -= 1
                        i += 1
                        if len(node.children[i].keys) == (2 * self.t) - 1:
                            self.split_child(node, i)
                            if key > node.keys[i]:
                                i += 1
                        self._insert_non_full(node.children[i], key)
                
                def split_child(self, parent, i):
                    t = self.t
                    node = parent.children[i]
                    new_node = BTreeNode(node.leaf)
                    
                    parent.keys.insert(i, node.keys[t - 1])
                    parent.children.insert(i + 1, new_node)
                    
                    new_node.keys = node.keys[t:]
                    node.keys = node.keys[:t - 1]
                    
                    if not node.leaf:
                        new_node.children = node.children[t:]
                        node.children = node.children[:t]
            
            btree = BTree(t=3)
            for term in terms:
                btree.insert(term)
            
            st.session_state.bst = bst_root
            st.session_state.btree = btree
            st.session_state.terms = terms
            
            st.success(f"Dictionary built with {len(terms)} unique terms!")
            st.info(f"BST height: ~{np.log2(len(terms)):.1f} | B-Tree height: ~{np.log(len(terms))/np.log(3):.1f}")
        
        # Performance comparison
        if st.session_state.bst and st.session_state.btree:
            st.subheader("Performance Comparison")
            
            # Search functionality
            search_term = st.text_input("Search for a term:")
            
            if search_term:
                stemmer = PorterStemmer()
                search_term_stemmed = stemmer.stem(search_term.lower())
                
                # BST search with separate timing
                bst_query_start = time.time()
                bst_result = bst_search(st.session_state.bst, search_term_stemmed)
                bst_query_time = (time.time() - bst_query_start) * 1000
                
                bst_retrieval_start = time.time()
                bst_retrieved = bst_result is not None
                bst_retrieval_time = (time.time() - bst_retrieval_start) * 1000
                
                # B-Tree search with separate timing
                btree_query_start = time.time()
                btree_result = st.session_state.btree.search(search_term_stemmed)
                btree_query_time = (time.time() - btree_query_start) * 1000
                
                btree_retrieval_start = time.time()
                btree_retrieved = btree_result
                btree_retrieval_time = (time.time() - btree_retrieval_start) * 1000
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Binary Search Tree (BST)**")
                    st.write(f"Found: {bst_result is not None}")
                    st.write(f"Query Search Time: {bst_query_time:.6f} ms")
                    st.write(f"Retrieval Time: {bst_retrieval_time:.6f} ms")
                    st.write(f"Total Time: {bst_query_time + bst_retrieval_time:.6f} ms")
                
                with col2:
                    st.markdown("**B-Tree**")
                    st.write(f"Found: {btree_result}")
                    st.write(f"Query Search Time: {btree_query_time:.6f} ms")
                    st.write(f"Retrieval Time: {btree_retrieval_time:.6f} ms")
                    st.write(f"Total Time: {btree_query_time + btree_retrieval_time:.6f} ms")
            
            # Multiple query test
            st.subheader("Multiple Query Performance Test")
            
            if st.button("Run 100 Random Queries"):
                import random
                test_terms = random.sample(st.session_state.terms, min(100, len(st.session_state.terms)))
                
                # BST times with separate query and retrieval timing
                bst_query_times = []
                bst_retrieval_times = []
                bst_total_times = []
                
                for term in test_terms:
                    # Query search time
                    query_start = time.time()
                    bst_result = bst_search(st.session_state.bst, term)
                    query_time = (time.time() - query_start) * 1000
                    
                    # Retrieval time
                    retrieval_start = time.time()
                    retrieved = bst_result is not None
                    retrieval_time = (time.time() - retrieval_start) * 1000
                    
                    bst_query_times.append(query_time)
                    bst_retrieval_times.append(retrieval_time)
                    bst_total_times.append(query_time + retrieval_time)
                
                # B-Tree times with separate query and retrieval timing
                btree_query_times = []
                btree_retrieval_times = []
                btree_total_times = []
                
                for term in test_terms:
                    # Query search time
                    query_start = time.time()
                    btree_result = st.session_state.btree.search(term)
                    query_time = (time.time() - query_start) * 1000
                    
                    # Retrieval time
                    retrieval_start = time.time()
                    retrieved = btree_result
                    retrieval_time = (time.time() - retrieval_start) * 1000
                    
                    btree_query_times.append(query_time)
                    btree_retrieval_times.append(retrieval_time)
                    btree_total_times.append(query_time + retrieval_time)
                
                # Results with separate timing columns
                results_df = pd.DataFrame({
                    'Query': test_terms,
                    'BST Query Time (ms)': bst_query_times,
                    'BST Retrieval Time (ms)': bst_retrieval_times,
                    'BST Total Time (ms)': bst_total_times,
                    'B-Tree Query Time (ms)': btree_query_times,
                    'B-Tree Retrieval Time (ms)': btree_retrieval_times,
                    'B-Tree Total Time (ms)': btree_total_times
                })
                
                st.dataframe(results_df)
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("BST Avg Query Time", f"{np.mean(bst_query_times):.6f} ms")
                with col2:
                    st.metric("BST Avg Retrieval Time", f"{np.mean(bst_retrieval_times):.6f} ms")
                with col3:
                    st.metric("B-Tree Avg Query Time", f"{np.mean(btree_query_times):.6f} ms")
                with col4:
                    st.metric("B-Tree Avg Retrieval Time", f"{np.mean(btree_retrieval_times):.6f} ms")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("BST Avg Total Time", f"{np.mean(bst_total_times):.6f} ms")
                with col2:
                    st.metric("B-Tree Avg Total Time", f"{np.mean(btree_total_times):.6f} ms")
                with col3:
                    faster = "BST" if np.mean(bst_total_times) < np.mean(btree_total_times) else "B-Tree"
                    st.metric("Faster (Total)", faster)
                
                # Store BST vs B-Tree experimental results
                st.session_state.exp_results['bst_vs_btree'] = {
                    'bst_avg_query_time': np.mean(bst_query_times),
                    'bst_avg_retrieval_time': np.mean(bst_retrieval_times),
                    'bst_avg_total_time': np.mean(bst_total_times),
                    'btree_avg_query_time': np.mean(btree_query_times),
                    'btree_avg_retrieval_time': np.mean(btree_retrieval_times),
                    'btree_avg_total_time': np.mean(btree_total_times),
                    'faster': faster
                }
                
                st.info(f"""
                **Performance Analysis:**
                - BST Average Query Search Time: {np.mean(bst_query_times):.6f} ms
                - BST Average Retrieval Time: {np.mean(bst_retrieval_times):.6f} ms
                - BST Average Total Time: {np.mean(bst_total_times):.6f} ms
                - B-Tree Average Query Search Time: {np.mean(btree_query_times):.6f} ms
                - B-Tree Average Retrieval Time: {np.mean(btree_retrieval_times):.6f} ms
                - B-Tree Average Total Time: {np.mean(btree_total_times):.6f} ms
                - {faster} performed better for this dataset
                
                **Inference:**
                - BST is generally faster for in-memory searches with balanced trees
                - B-Tree is optimized for disk-based storage and larger datasets
                - For this in-memory dictionary, {faster} performed better due to {'simpler structure' if faster == 'BST' else 'better worst-case performance'}
                - B-Tree would be preferable for very large datasets or disk-based indexing
                """)

# ==================== TOLERANT RETRIEVAL ====================
elif page == "Tolerant Retrieval":
    st.header("🔧 Tolerant Retrieval")
    
    if not st.session_state.documents:
        st.warning("Please upload documents first!")
    else:
        # K-gram size selection
        k_value = st.selectbox("Select K-gram size:", options=[2, 3, 4], index=1, help="3-gram is commonly used in information retrieval")
        
        # Build k-gram index
        if st.button(f"Build {k_value}-Gram Index"):
            stemmer = PorterStemmer()
            stop_words = set(stopwords.words('english'))
            
            terms = set()
            for content in st.session_state.documents.values():
                tokens = word_tokenize(content.lower())
                tokens = [token for token in tokens if token not in stop_words and token not in string.punctuation]
                tokens = [stemmer.stem(token) for token in tokens]
                terms.update(tokens)
            
            # Build k-gram index
            kgram_index = defaultdict(set)
            for term in terms:
                term_with_bounds = f"${term}$"
                for i in range(len(term_with_bounds) - k_value + 1):
                    kgram = term_with_bounds[i:i+k_value]
                    if len(kgram) == k_value:
                        kgram_index[kgram].add(term)
            
            st.session_state.kgram_index = dict(kgram_index)
            st.session_state.terms = terms
            st.session_state.k_value = k_value
            st.success(f"{k_value}-gram index built with {len(kgram_index)} unique {k_value}-grams!")
        
        # Wildcard query
        st.subheader("Wildcard Query")
        wildcard_query = st.text_input("Enter wildcard query (use * for wildcard, e.g., 'mach*'):")
        
        if wildcard_query and st.session_state.kgram_index:
            k = st.session_state.get('k_value', 3)  # Default to 3 if not set
            
            # Convert wildcard to k-grams
            parts = wildcard_query.split('*')
            
            if len(parts) == 2:
                prefix, suffix = parts
                
                # Generate k-grams from the query parts
                matching_terms = set()
                
                if prefix:
                    # Generate k-grams for prefix (need k-1 chars from prefix)
                    if len(prefix) >= k - 1:
                        prefix_kgram = f"${prefix[-(k-1):]}" if k > 1 else f"${prefix[-1]}"
                        if prefix_kgram in st.session_state.kgram_index:
                            matching_terms.update(st.session_state.kgram_index[prefix_kgram])
                
                if suffix:
                    # Generate k-grams for suffix (need k-1 chars from suffix)
                    if len(suffix) >= k - 1:
                        suffix_kgram = f"{suffix[:k-1]}$" if k > 1 else f"{suffix[0]}$"
                        if suffix_kgram in st.session_state.kgram_index:
                            if matching_terms:
                                matching_terms = matching_terms.intersection(st.session_state.kgram_index[suffix_kgram])
                            else:
                                matching_terms = set(st.session_state.kgram_index[suffix_kgram])
                
                # Filter by pattern
                import fnmatch
                pattern = wildcard_query.replace('*', '*')
                final_matches = [term for term in matching_terms if fnmatch.fnmatch(term, pattern)]
                
                st.write(f"Matching terms: {final_matches}")
            else:
                st.warning("Wildcard query must have exactly one * (e.g., 'mach*')")
        
        # Spelling correction (Edit Distance)
        st.subheader("Spelling Correction (Edit Distance)")
        misspelled = st.text_input("Enter potentially misspelled word:")
        
        if misspelled and st.session_state.terms:
            stemmer = PorterStemmer()
            misspelled_stemmed = stemmer.stem(misspelled.lower())
            
            # Find closest match using edit distance
            suggestions = []
            for term in st.session_state.terms:
                distance = edit_distance(misspelled_stemmed, term)
                if distance <= 2:  # Allow up to 2 edits
                    suggestions.append((term, distance))
            
            suggestions.sort(key=lambda x: x[1])
            
            if suggestions:
                st.write("Suggestions:")
                for term, distance in suggestions[:5]:
                    st.write(f"- {term} (edit distance: {distance})")
            else:
                st.write("No close matches found")
        
        # Phonetic correction (Soundex)
        st.subheader("Phonetic Correction (Soundex)")
        
        def soundex(name):
            name = name.upper()
            first_char = name[0]
            name = name[1:]
            
            soundex_mapping = {
                'A': '', 'E': '', 'I': '', 'O': '', 'U': '', 'Y': '',
                'B': '1', 'F': '1', 'P': '1', 'V': '1',
                'C': '2', 'G': '2', 'J': '2', 'K': '2', 'Q': '2', 'S': '2', 'X': '2', 'Z': '2',
                'D': '3', 'T': '3',
                'L': '4',
                'M': '5', 'N': '5',
                'R': '6'
            }
            
            soundex_code = first_char
            for char in name:
                if char in soundex_mapping:
                    code = soundex_mapping[char]
                    if code and (len(soundex_code) < 4) and (not soundex_code or code != soundex_code[-1]):
                        soundex_code += code
            
            while len(soundex_code) < 4:
                soundex_code += '0'
            
            return soundex_code[:4]
        
        phonetic_query = st.text_input("Enter word for phonetic matching:")
        
        if phonetic_query and st.session_state.terms:
            query_soundex = soundex(phonetic_query)
            
            phonetic_matches = []
            for term in st.session_state.terms:
                if soundex(term) == query_soundex:
                    phonetic_matches.append(term)
            
            st.write(f"Soundex code for '{phonetic_query}': {query_soundex}")
            st.write(f"Phonetically similar terms: {phonetic_matches}")
        
        # Tolerant retrieval summary
        st.subheader("📊 Tolerant Retrieval Summary")
        
        # Store tolerant retrieval experimental results
        st.session_state.exp_results['tolerant_retrieval'] = {
            'wildcard_tested': wildcard_query is not None and len(wildcard_query) > 0,
            'spelling_tested': misspelled is not None and len(misspelled) > 0,
            'phonetic_tested': phonetic_query is not None and len(phonetic_query) > 0
        }
        
        st.info("""
        **Tolerant Retrieval Techniques Implemented:**
        
        1. **Wildcard Queries**: Uses k-gram index to match patterns with wildcards (e.g., 'mach*' matches 'machine', 'machines')
        
        2. **Spelling Correction**: Uses edit distance (Levenshtein distance) to find words with similar spelling
        
        3. **Phonetic Correction**: Uses Soundex algorithm to find phonetically similar words
        
        **Effectiveness:**
        - Wildcard queries are useful when users remember partial terms
        - Edit distance helps with typos and misspellings
        - Phonetic correction helps with words that sound similar but are spelled differently
        
        **Limitations:**
        - K-gram index can produce false positives
        - Edit distance is computationally expensive for large vocabularies
        - Soundex is language-specific and may not work well for all words
        """)

# ==================== INFERENCE & DISCUSSION ====================
elif page == "Inference & Discussion":
    st.header("📝 Inference and Discussion")
    
    st.markdown("""
    ## Assignment Inferences and Conclusions
    
    *Note: These inferences are generated dynamically based on your experimental results. Run the experiments in each module to populate these conclusions.*
    """)
    
    # Check if experiments have been run
    experiments_run = any(st.session_state.exp_results[key] is not None for key in st.session_state.exp_results)
    
    if not experiments_run:
        st.warning("⚠️ No experimental results found. Please run experiments in the following modules to generate dynamic inferences:")
        st.markdown("- Text Preprocessing (Compare Stemming and Lemmatization)")
        st.markdown("- Phrase Query Processing (Run a phrase query)")
        st.markdown("- Dictionary Search (Run 100 Random Queries)")
        st.markdown("- Tolerant Retrieval (Test wildcard, spelling, or phonetic queries)")
    else:
        # 1. Preprocessing Quality
        st.subheader("1. Which preprocessing technique improved retrieval quality?")
        if st.session_state.exp_results['stem_vs_lem']:
            stem_data = st.session_state.exp_results['stem_vs_lem']
            st.info(f"""
            **Answer:** Based on experimental results with query "{stem_data['query']}":
            
            - Stemming retrieved {stem_data['stemmed_retrieved']} documents
            - Lemmatization retrieved {stem_data['lemmatized_retrieved']} documents
            - Vocabulary reduction: Stemming {stem_data['avg_stem_reduction']:.2f}%, Lemmatization {stem_data['avg_lem_reduction']:.2f}%
            
            **Conclusion:** Based on document coverage, {'lemmatization' if stem_data['lemmatized_retrieved'] >= stem_data['stemmed_retrieved'] else 'stemming'} had {'higher' if stem_data['lemmatized_retrieved'] > stem_data['stemmed_retrieved'] else 'equal or higher'} document retrieval for this dataset.
            The combination of lowercasing, stop word removal, and {'lemmatization' if stem_data['better_method'] == 'lemmatization' else 'stemming'} provided better document coverage.
            """)
        else:
            st.warning("Run the stemming vs lemmatization comparison in Text Preprocessing module to see results.")
        
        # 2. Stemming vs Lemmatization
        st.subheader("2. Was stemming or lemmatization better for their dataset?")
        if st.session_state.exp_results['stem_vs_lem']:
            stem_data = st.session_state.exp_results['stem_vs_lem']
            st.info(f"""
            **Answer:** {'Lemmatization' if stem_data['better_method'] == 'lemmatization' else 'Stemming'} had better document coverage for this dataset.
            
            **Experimental Evidence:**
            - Query: "{stem_data['query']}"
            - Stemming retrieved {stem_data['stemmed_retrieved']} documents
            - Lemmatization retrieved {stem_data['lemmatized_retrieved']} documents
            - Vocabulary reduction: Stemming {stem_data['avg_stem_reduction']:.2f}% vs Lemmatization {stem_data['avg_lem_reduction']:.2f}%
            
            **Analysis:**
            - {'Lemmatization' if stem_data['better_method'] == 'lemmatization' else 'Stemming'} {'retrieved more or equal documents' if stem_data['lemmatized_retrieved'] >= stem_data['stemmed_retrieved'] else 'retrieved more documents'}
            - {'Lemmatization' if stem_data['avg_lem_reduction'] < stem_data['avg_stem_reduction'] else 'Stemming'} reduced vocabulary less aggressively
            - For information retrieval, preserving semantic meaning is crucial for relevance
            - Note: Document coverage does not directly indicate retrieval quality (relevance of results)
            """)
        else:
            st.warning("Run the stemming vs lemmatization comparison in Text Preprocessing module to see results.")
        
        # 3. Phrase Query Accuracy
        st.subheader("3. Which phrase query index was more accurate?")
        if st.session_state.exp_results['phrase_query']:
            phrase_data = st.session_state.exp_results['phrase_query']
            st.info(f"""
            **Answer:** Positional index is theoretically more accurate than biword index.
            
            **Experimental Evidence for query "{phrase_data['query']}":**
            - Biword index retrieved {phrase_data['biword_results']} documents
            - Positional index retrieved {phrase_data['positional_results']} documents
            
            **Analysis:**
            - Positional index stores exact position information, allowing precise phrase matching
            - Biword index can give false positives for phrases longer than 2 words
            - Positional index can handle proximity queries and variable-length phrases
            - Biword index is simpler but less flexible for complex phrase queries
            - Positional index gives more accurate results because it verifies word adjacency and order
            """)
        else:
            st.warning("Run a phrase query in Phrase Query Processing module to see results.")
        
        # 4. Faster Tree Structure
        st.subheader("4. Which tree structure was faster?")
        if st.session_state.exp_results['bst_vs_btree']:
            tree_data = st.session_state.exp_results['bst_vs_btree']
            st.info(f"""
            **Answer:** {tree_data['faster'].upper()} was faster for this dataset.
            
            **Experimental Evidence (100 random queries):**
            - BST Average Query Time: {tree_data['bst_avg_query_time']:.6f} ms
            - BST Average Retrieval Time: {tree_data['bst_avg_retrieval_time']:.6f} ms
            - BST Average Total Time: {tree_data['bst_avg_total_time']:.6f} ms
            - B-Tree Average Query Time: {tree_data['btree_avg_query_time']:.6f} ms
            - B-Tree Average Retrieval Time: {tree_data['btree_avg_retrieval_time']:.6f} ms
            - B-Tree Average Total Time: {tree_data['btree_avg_total_time']:.6f} ms
            
            **Analysis:**
            - {tree_data['faster'].upper()} had lower average total time for this dataset
            - BST is generally faster for in-memory searches with balanced trees
            - B-Tree is optimized for disk-based storage and larger datasets
            - B-Tree has better worst-case performance and is optimized for block-based storage
            """)
        else:
            st.warning("Run 100 Random Queries in Dictionary Search module to see results.")
        
        # 5. Retrieval Tolerance
        st.subheader("5. How tolerant was their retrieval model?")
        if st.session_state.exp_results['tolerant_retrieval']:
            tol_data = st.session_state.exp_results['tolerant_retrieval']
            st.info(f"""
            **Answer:** The retrieval model demonstrated tolerance to imperfect queries through:
            
            **Techniques Tested:**
            - Wildcard queries: {'✓ Tested' if tol_data['wildcard_tested'] else '✗ Not tested'}
            - Spelling correction (edit distance): {'✓ Tested' if tol_data['spelling_tested'] else '✗ Not tested'}
            - Phonetic correction (Soundex): {'✓ Tested' if tol_data['phonetic_tested'] else '✗ Not tested'}
            
            **Analysis:**
            - Wildcard queries handle partial term inputs effectively
            - Edit distance helps with typos and misspellings within 2 edits
            - Phonetic correction handles phonetically similar words
            - K-gram index enables efficient pattern matching
            - The system can recover from various user input errors
            """)
        else:
            st.warning("Test wildcard, spelling correction, or phonetic queries in Tolerant Retrieval module to see results.")
        
        # 6. Limitations (Static - these are inherent to the implementation)
        st.subheader("6. What are the limitations of their system?")
        st.info("""
        **Answer:** System limitations include:
        - Limited to text documents (doesn't handle PDF, Word, etc.)
        - No relevance feedback or learning from user interactions
        - No ranking algorithm (TF-IDF, BM25) for result ordering
        - K-gram index can produce false positives in wildcard queries
        - Edit distance is computationally expensive for large vocabularies
        - Soundex is English-specific and may not work for other languages
        - No support for semantic search or vector embeddings
        - No query expansion or thesaurus integration
        """)
        
        # 7. Improvements (Static - these are potential enhancements)
        st.subheader("7. How can the system be improved?")
        st.info("""
        **Answer:** Potential improvements include:
        - Implement TF-IDF or BM25 ranking for better result ordering
        - Add support for multiple document formats (PDF, DOCX, etc.)
        - Implement query expansion using thesaurus or word embeddings
        - Add relevance feedback to learn from user interactions
        - Use vector embeddings for semantic search
        - Implement caching for frequently accessed terms
        - Add support for Boolean queries (AND, OR, NOT)
        - Implement more sophisticated spelling correction (e.g., Peter Norvig's algorithm)
        - Add support for multi-language documents
        - Implement distributed indexing for large-scale collections
        - Add user interface improvements (autocomplete, query suggestions)
        """)
    
    st.markdown("---")
    st.markdown("""
    **Note:** These inferences are generated dynamically based on the experimental results from the Streamlit application. 
    Run experiments in each module to populate the conclusions with your specific findings for your report.
    """)

