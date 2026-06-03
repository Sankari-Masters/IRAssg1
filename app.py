import fnmatch
import io
import math
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import nltk
import numpy as np
import pandas as pd
import streamlit as st
from nltk.metrics.distance import edit_distance
from nltk.stem import PorterStemmer, WordNetLemmatizer


st.set_page_config(page_title="Information Retrieval Assignment 1", layout="wide")


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*")

FALLBACK_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "he", "in", "is", "it", "its", "of", "on", "that", "the",
    "to", "was", "were", "will", "with", "this", "these", "those",
    "into", "than", "then", "through", "or", "but", "not", "can",
    "also", "their", "they", "them", "we", "you", "your", "our",
}

DEFAULT_EXPERIMENTS = {
    "stem_vs_lemma": None,
    "phrase_query": None,
    "tree": None,
    "tolerant": None,
}


@st.cache_resource
def language_tools():
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    try:
        from nltk.corpus import stopwords

        stop_words = set(stopwords.words("english"))
    except LookupError:
        stop_words = FALLBACK_STOPWORDS
    return stemmer, lemmatizer, stop_words


def init_state():
    defaults = {
        "documents": {},
        "preprocessed_docs": {},
        "inverted_index": {},
        "doc_tokens": {},
        "biword_index": {},
        "positional_index": {},
        "phrase_tokens": {},
        "term_postings": {},
        "terms": [],
        "bst": None,
        "btree": None,
        "kgram_index": {},
        "k_value": 3,
        "exp_results": DEFAULT_EXPERIMENTS.copy(),
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_derived_state():
    mapping_keys = {
        "preprocessed_docs",
        "inverted_index",
        "doc_tokens",
        "biword_index",
        "positional_index",
        "phrase_tokens",
        "term_postings",
        "kgram_index",
    }
    for key in mapping_keys:
        st.session_state[key] = {}
    st.session_state.terms = []
    st.session_state.bst = None
    st.session_state.btree = None
    st.session_state.exp_results = DEFAULT_EXPERIMENTS.copy()


def decode_bytes(raw_bytes):
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return raw_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw_bytes.decode("utf-8", errors="ignore")


def csv_to_documents(file_name, text):
    try:
        df = pd.read_csv(io.StringIO(text))
    except Exception:
        return {file_name: text}

    documents = {}
    for row_number, row in df.iterrows():
        row_text = " ".join(str(value) for value in row.dropna().tolist()).strip()
        if row_text:
            documents[f"{file_name}#row-{row_number + 1}"] = row_text
    return documents or {file_name: text}


def load_uploaded_file(uploaded_file):
    text = decode_bytes(uploaded_file.read())
    if uploaded_file.name.lower().endswith(".csv"):
        return csv_to_documents(uploaded_file.name, text)
    return {uploaded_file.name: text}


def load_dataset_directory():
    dataset_dir = Path("dataset")
    documents = {}
    if not dataset_dir.exists():
        return documents

    for path in sorted(dataset_dir.glob("*.txt")):
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        if text:
            documents[path.name] = text
    return documents


def raw_tokens(text):
    return TOKEN_PATTERN.findall(text)


def simple_lemma(token):
    if len(token) > 4 and token.endswith("ies"):
        return f"{token[:-3]}y"
    if len(token) > 5 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 4 and token.endswith("ed"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s"):
        return token[:-1]
    return token


def apply_normalizer(token, method):
    stemmer, lemmatizer, _ = language_tools()
    if method == "stemming":
        return stemmer.stem(token)
    if method == "lemmatization":
        try:
            return lemmatizer.lemmatize(token)
        except LookupError:
            return simple_lemma(token)
    return token


def preprocess_text(
    text,
    lowercase=True,
    handle_hyphens=True,
    remove_stopwords=True,
    method="none",
):
    _, _, stop_words = language_tools()

    steps = {}
    tokens = raw_tokens(text)
    steps["Tokenization"] = tokens

    if lowercase:
        tokens = [token.lower() for token in tokens]
    steps["Lowercasing"] = tokens.copy()

    if handle_hyphens:
        expanded = []
        for token in tokens:
            expanded.extend(part for part in token.replace("-", " ").split() if part)
        tokens = expanded
    steps["Hyphen handling"] = tokens.copy()

    if remove_stopwords:
        tokens = [token for token in tokens if token.lower() not in stop_words]
    steps["Stop word removal"] = tokens.copy()

    if method != "none":
        tokens = [apply_normalizer(token, method) for token in tokens]
    steps["Stemming/Lemmatization"] = tokens.copy()
    steps["Final"] = tokens.copy()
    return steps


def build_inverted_index(documents, options):
    index = defaultdict(dict)
    doc_tokens = {}
    for doc_id, (_, text) in enumerate(documents.items()):
        tokens = preprocess_text(text, **options)["Final"]
        doc_tokens[doc_id] = tokens
        for term, freq in Counter(tokens).items():
            index[term][doc_id] = freq
    return {term: dict(postings) for term, postings in sorted(index.items())}, doc_tokens


def postings_rows(postings, documents):
    names = list(documents.keys())
    rows = []
    for doc_id, freq in sorted(postings.items()):
        rows.append({"Doc ID": doc_id, "Document": names[doc_id], "Term Frequency": freq})
    return rows


def build_term_postings(documents, options):
    index, _ = build_inverted_index(documents, options)
    return index, sorted(index.keys())


def preprocess_query(query, options):
    return preprocess_text(query, **options)["Final"]


def cosine_scores(query_tokens, doc_tokens):
    query_counts = Counter(query_tokens)
    query_norm = math.sqrt(sum(value * value for value in query_counts.values()))
    scores = {}

    if not query_counts or query_norm == 0:
        return scores

    for doc_id, tokens in doc_tokens.items():
        doc_counts = Counter(tokens)
        doc_norm = math.sqrt(sum(value * value for value in doc_counts.values()))
        if doc_norm == 0:
            scores[doc_id] = 0.0
            continue
        dot = sum(query_counts[term] * doc_counts.get(term, 0) for term in query_counts)
        scores[doc_id] = dot / (query_norm * doc_norm)
    return scores


def ranked_rows(scores, documents, postings=None):
    names = list(documents.keys())
    rows = []
    for doc_id, score in sorted(scores.items(), key=lambda item: (-item[1], item[0])):
        if score <= 0:
            continue
        rows.append(
            {
                "Doc ID": doc_id,
                "Document": names[doc_id],
                "Score": round(score, 4),
                "Matched Terms": len(postings.get(doc_id, [])) if postings else "",
            }
        )
    return rows


def compare_stem_and_lemma(documents, query):
    base_options = {
        "lowercase": True,
        "handle_hyphens": True,
        "remove_stopwords": True,
    }
    comparisons = []
    method_payload = {}

    for method in ("stemming", "lemmatization"):
        options = {**base_options, "method": method}
        index, doc_tokens = build_inverted_index(documents, options)
        query_tokens = preprocess_query(query, options)
        scores = cosine_scores(query_tokens, doc_tokens)
        retrieved_docs = {doc_id for doc_id, score in scores.items() if score > 0}

        original_vocab = []
        method_vocab = []
        for text in documents.values():
            original_vocab.extend(
                preprocess_text(text, **{**base_options, "method": "none"})["Final"]
            )
            method_vocab.extend(preprocess_text(text, **options)["Final"])

        original_size = len(set(original_vocab)) or 1
        method_size = len(set(method_vocab))
        reduction = (original_size - method_size) / original_size * 100
        top_scores = sorted([score for score in scores.values() if score > 0], reverse=True)[:3]
        avg_top_score = float(np.mean(top_scores)) if top_scores else 0.0

        comparisons.append(
            {
                "Method": method.title(),
                "Vocabulary Size": method_size,
                "Vocabulary Reduction %": round(reduction, 2),
                "Retrieved Documents": len(retrieved_docs),
                "Average Top-3 Cosine": round(avg_top_score, 4),
                "Query Tokens": ", ".join(query_tokens),
            }
        )
        method_payload[method] = {
            "index": index,
            "doc_tokens": doc_tokens,
            "scores": scores,
            "query_tokens": query_tokens,
            "retrieved_docs": retrieved_docs,
            "reduction": reduction,
            "avg_top_score": avg_top_score,
        }

    stem_score = method_payload["stemming"]["avg_top_score"]
    lemma_score = method_payload["lemmatization"]["avg_top_score"]
    better = "lemmatization" if lemma_score >= stem_score else "stemming"
    return comparisons, method_payload, better


def build_phrase_indices(documents, options):
    biword_index = defaultdict(set)
    positional_index = defaultdict(lambda: defaultdict(list))
    phrase_tokens = {}

    for doc_id, (_, text) in enumerate(documents.items()):
        tokens = preprocess_text(text, **options)["Final"]
        phrase_tokens[doc_id] = tokens

        for pos, term in enumerate(tokens):
            positional_index[term][doc_id].append(pos)

        for first, second in zip(tokens, tokens[1:]):
            biword_index[f"{first} {second}"].add(doc_id)

    positional = {
        term: {doc_id: positions for doc_id, positions in postings.items()}
        for term, postings in sorted(positional_index.items())
    }
    biword = {term: set(postings) for term, postings in sorted(biword_index.items())}
    return biword, positional, phrase_tokens


def search_biword_phrase(query_tokens, biword_index, positional_index):
    if not query_tokens:
        return set()
    if len(query_tokens) == 1:
        return set(positional_index.get(query_tokens[0], {}).keys())

    pairs = [f"{first} {second}" for first, second in zip(query_tokens, query_tokens[1:])]
    posting_sets = [set(biword_index.get(pair, set())) for pair in pairs]
    if not posting_sets:
        return set()
    return set.intersection(*posting_sets)


def search_positional_phrase(query_tokens, positional_index):
    if not query_tokens:
        return set()

    posting_sets = []
    for term in query_tokens:
        postings = positional_index.get(term)
        if not postings:
            return set()
        posting_sets.append(set(postings.keys()))

    result_docs = set()
    for doc_id in set.intersection(*posting_sets):
        starts = set(positional_index[query_tokens[0]][doc_id])
        for offset, term in enumerate(query_tokens[1:], start=1):
            next_positions = set(positional_index[term][doc_id])
            starts = {start for start in starts if start + offset in next_positions}
            if not starts:
                break
        if starts:
            result_docs.add(doc_id)
    return result_docs


def doc_list(doc_ids, documents):
    names = list(documents.keys())
    return [names[doc_id] for doc_id in sorted(doc_ids)]


@dataclass
class BSTNode:
    key: str
    value: dict
    left: object = None
    right: object = None


def build_balanced_bst(items):
    if not items:
        return None
    mid = len(items) // 2
    key, value = items[mid]
    return BSTNode(
        key=key,
        value=value,
        left=build_balanced_bst(items[:mid]),
        right=build_balanced_bst(items[mid + 1 :]),
    )


def bst_search(root, key):
    current = root
    while current is not None:
        if key == current.key:
            return current.value
        if key < current.key:
            current = current.left
        else:
            current = current.right
    return None


class BTreeNode:
    def __init__(self, leaf=True):
        self.leaf = leaf
        self.keys = []
        self.values = []
        self.children = []


class BTree:
    def __init__(self, degree=8):
        self.root = BTreeNode(True)
        self.degree = degree

    def search(self, key, node=None):
        node = node or self.root
        i = 0
        while i < len(node.keys) and key > node.keys[i]:
            i += 1
        if i < len(node.keys) and key == node.keys[i]:
            return node.values[i]
        if node.leaf:
            return None
        return self.search(key, node.children[i])

    def insert(self, key, value):
        if self.search(key) is not None:
            return

        root = self.root
        if len(root.keys) == (2 * self.degree) - 1:
            new_root = BTreeNode(False)
            new_root.children.append(root)
            self.root = new_root
            self.split_child(new_root, 0)
        self.insert_non_full(self.root, key, value)

    def insert_non_full(self, node, key, value):
        i = len(node.keys) - 1
        if node.leaf:
            node.keys.append("")
            node.values.append({})
            while i >= 0 and key < node.keys[i]:
                node.keys[i + 1] = node.keys[i]
                node.values[i + 1] = node.values[i]
                i -= 1
            node.keys[i + 1] = key
            node.values[i + 1] = value
            return

        while i >= 0 and key < node.keys[i]:
            i -= 1
        i += 1
        if len(node.children[i].keys) == (2 * self.degree) - 1:
            self.split_child(node, i)
            if key > node.keys[i]:
                i += 1
        self.insert_non_full(node.children[i], key, value)

    def split_child(self, parent, index):
        degree = self.degree
        child = parent.children[index]
        new_child = BTreeNode(child.leaf)

        median_key = child.keys[degree - 1]
        median_value = child.values[degree - 1]

        new_child.keys = child.keys[degree:]
        new_child.values = child.values[degree:]
        child.keys = child.keys[: degree - 1]
        child.values = child.values[: degree - 1]

        if not child.leaf:
            new_child.children = child.children[degree:]
            child.children = child.children[:degree]

        parent.keys.insert(index, median_key)
        parent.values.insert(index, median_value)
        parent.children.insert(index + 1, new_child)


def btree_height(node):
    if node is None:
        return 0
    if node.leaf:
        return 1
    return 1 + btree_height(node.children[0])


def bst_height(node):
    if node is None:
        return 0
    return 1 + max(bst_height(node.left), bst_height(node.right))


def timed_search(search_fn, key, repeat=5):
    timings = []
    value = None
    for _ in range(repeat):
        start = time.perf_counter_ns()
        value = search_fn(key)
        timings.append((time.perf_counter_ns() - start) / 1_000_000)
    return value, float(np.mean(timings))


def retrieval_time(value, repeat=5):
    timings = []
    docs = {}
    for _ in range(repeat):
        start = time.perf_counter_ns()
        docs = dict(value or {})
        timings.append((time.perf_counter_ns() - start) / 1_000_000)
    return docs, float(np.mean(timings))


def build_kgram_index(terms, k):
    index = defaultdict(set)
    for term in terms:
        bounded = f"${term}$"
        for i in range(max(len(bounded) - k + 1, 0)):
            index[bounded[i : i + k]].add(term)
    return {gram: set(values) for gram, values in sorted(index.items())}


def text_kgrams(text, k):
    return [text[i : i + k] for i in range(max(len(text) - k + 1, 0))]


def wildcard_search(pattern, terms, kgram_index, k):
    pattern = pattern.lower().strip()
    if not pattern:
        return []

    fragments = pattern.split("*")
    grams = []
    for idx, fragment in enumerate(fragments):
        if not fragment:
            continue
        source = fragment
        if idx == 0 and not pattern.startswith("*"):
            source = f"${source}"
        if idx == len(fragments) - 1 and not pattern.endswith("*"):
            source = f"{source}$"
        grams.extend(text_kgrams(source, k))

    candidate_sets = [kgram_index[gram] for gram in grams if gram in kgram_index]
    if grams and len(candidate_sets) != len(grams):
        candidates = set()
    elif candidate_sets:
        candidates = set.intersection(*candidate_sets)
    else:
        candidates = set(terms)

    matches = [term for term in candidates if fnmatch.fnmatch(term, pattern)]
    return sorted(matches)


def soundex(term):
    cleaned = re.sub(r"[^A-Za-z]", "", term).upper()
    if not cleaned:
        return ""

    mapping = {
        "B": "1",
        "F": "1",
        "P": "1",
        "V": "1",
        "C": "2",
        "G": "2",
        "J": "2",
        "K": "2",
        "Q": "2",
        "S": "2",
        "X": "2",
        "Z": "2",
        "D": "3",
        "T": "3",
        "L": "4",
        "M": "5",
        "N": "5",
        "R": "6",
    }

    first = cleaned[0]
    encoded = []
    previous = mapping.get(first, "")
    for char in cleaned[1:]:
        code = mapping.get(char, "")
        if code and code != previous:
            encoded.append(code)
        previous = code
    return (first + "".join(encoded) + "000")[:4]


def normalizer_controls(prefix, defaults=None):
    defaults = defaults or {}
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        lowercase = st.checkbox("Lowercase", value=defaults.get("lowercase", True), key=f"{prefix}_lower")
    with col2:
        handle_hyphens = st.checkbox(
            "Split hyphenated terms",
            value=defaults.get("handle_hyphens", True),
            key=f"{prefix}_hyphen",
        )
    with col3:
        remove_stopwords = st.checkbox(
            "Remove stop words",
            value=defaults.get("remove_stopwords", True),
            key=f"{prefix}_stop",
        )
    with col4:
        method = st.selectbox(
            "Normalizer",
            ["none", "stemming", "lemmatization"],
            index=["none", "stemming", "lemmatization"].index(defaults.get("method", "none")),
            key=f"{prefix}_method",
        )
    return {
        "lowercase": lowercase,
        "handle_hyphens": handle_hyphens,
        "remove_stopwords": remove_stopwords,
        "method": method,
    }


def require_documents():
    if st.session_state.documents:
        return True
    st.warning("Load or upload documents first from the Document Upload page.")
    return False


def document_metric_bar():
    documents = st.session_state.documents
    total_chars = sum(len(text) for text in documents.values())
    col1, col2, col3 = st.columns(3)
    col1.metric("Documents", len(documents))
    col2.metric("Characters", f"{total_chars:,}")
    col3.metric("Dataset Folder Files", len(load_dataset_directory()))


init_state()

st.title("Information Retrieval System")
st.caption("Streamlit implementation for Assignment 1: preprocessing, indexing, phrase queries, tree dictionaries, tolerant retrieval, and inferences.")

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select module",
    [
        "Document Upload",
        "Text Preprocessing",
        "Inverted Index",
        "Phrase Query Processing",
        "Dictionary Search (BST & B-Tree)",
        "Tolerant Retrieval",
        "Inference & Discussion",
    ],
)
st.sidebar.write(f"Loaded documents: {len(st.session_state.documents)}")


if page == "Document Upload":
    st.header("Document Upload")
    st.write("Upload TXT/CSV files or load the included document collection from the `dataset` folder.")

    uploaded_files = st.file_uploader(
        "Upload text documents or CSV datasets",
        type=["txt", "csv"],
        accept_multiple_files=True,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Add uploaded files", disabled=not uploaded_files):
            added = {}
            for uploaded_file in uploaded_files:
                added.update(load_uploaded_file(uploaded_file))
            st.session_state.documents.update(added)
            reset_derived_state()
            st.success(f"Added {len(added)} document(s).")

    with col2:
        if st.button("Load sample dataset"):
            documents = load_dataset_directory()
            if documents:
                st.session_state.documents = documents
                reset_derived_state()
                st.success(f"Loaded {len(documents)} dataset document(s).")
            else:
                st.error("No non-empty .txt files were found in the dataset folder.")

    with col3:
        if st.button("Clear loaded documents"):
            st.session_state.documents = {}
            reset_derived_state()
            st.success("Cleared the document collection.")

    if st.session_state.documents:
        document_metric_bar()
        st.subheader("Document Collection")
        for name, text in st.session_state.documents.items():
            with st.expander(name):
                st.text_area("Content", text, height=180, key=f"doc_view_{name}", disabled=True)


elif page == "Text Preprocessing":
    st.header("Text Preprocessing")
    if require_documents():
        st.subheader("Preprocessing options")
        options = normalizer_controls(
            "prep",
            defaults={
                "lowercase": True,
                "handle_hyphens": True,
                "remove_stopwords": True,
                "method": "lemmatization",
            },
        )

        if st.button("Apply preprocessing and build inverted index"):
            preprocessed = {}
            for name, text in st.session_state.documents.items():
                preprocessed[name] = preprocess_text(text, **options)
            st.session_state.preprocessed_docs = preprocessed
            index, doc_tokens = build_inverted_index(st.session_state.documents, options)
            st.session_state.inverted_index = index
            st.session_state.doc_tokens = doc_tokens
            st.success("Preprocessing complete and inverted index generated.")

        if st.session_state.preprocessed_docs:
            rows = []
            for name, steps in st.session_state.preprocessed_docs.items():
                rows.append(
                    {
                        "Document": name,
                        "Original Tokens": len(steps["Tokenization"]),
                        "Final Tokens": len(steps["Final"]),
                        "Unique Final Terms": len(set(steps["Final"])),
                    }
                )
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

            selected_doc = st.selectbox("Inspect preprocessing steps", list(st.session_state.preprocessed_docs.keys()))
            steps = st.session_state.preprocessed_docs[selected_doc]
            for step_name, tokens in steps.items():
                with st.expander(f"{step_name}: {len(tokens)} tokens"):
                    st.write(tokens[:120])

            st.subheader("Inverted index sample")
            sample_rows = []
            for term, postings in list(st.session_state.inverted_index.items())[:25]:
                sample_rows.append(
                    {
                        "Term": term,
                        "Document Frequency": len(postings),
                        "Postings": postings_rows(postings, st.session_state.documents),
                    }
                )
            st.dataframe(pd.DataFrame(sample_rows), use_container_width=True)

        st.subheader("Stemming vs Lemmatization")
        query = st.text_input("Test query for retrieval-quality comparison", value="machine learning algorithms")
        if st.button("Compare stemming and lemmatization"):
            comparison_rows, payload, better = compare_stem_and_lemma(st.session_state.documents, query)
            st.session_state.exp_results["stem_vs_lemma"] = {
                "query": query,
                "comparison_rows": comparison_rows,
                "better": better,
                "stem_score": payload["stemming"]["avg_top_score"],
                "lemma_score": payload["lemmatization"]["avg_top_score"],
            }
            st.dataframe(pd.DataFrame(comparison_rows), use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Stemming ranked results**")
                st.dataframe(
                    pd.DataFrame(ranked_rows(payload["stemming"]["scores"], st.session_state.documents)),
                    use_container_width=True,
                )
            with col2:
                st.markdown("**Lemmatization ranked results**")
                st.dataframe(
                    pd.DataFrame(ranked_rows(payload["lemmatization"]["scores"], st.session_state.documents)),
                    use_container_width=True,
                )

            st.info(
                "The comparison uses cosine similarity between the processed query vector and each processed document vector. "
                f"For this query, {better} has the stronger or equal average top-3 score."
            )


elif page == "Inverted Index":
    st.header("Inverted Index")
    if require_documents():
        options = normalizer_controls(
            "index",
            defaults={
                "lowercase": True,
                "handle_hyphens": True,
                "remove_stopwords": True,
                "method": "stemming",
            },
        )

        if st.button("Build inverted index"):
            index, doc_tokens = build_inverted_index(st.session_state.documents, options)
            st.session_state.inverted_index = index
            st.session_state.doc_tokens = doc_tokens
            st.success(f"Built an inverted index with {len(index)} unique terms.")

        if st.session_state.inverted_index:
            query = st.text_input("Enter term query", value="retrieval")
            if query:
                query_terms = preprocess_query(query, options)
                scores = defaultdict(float)
                matched_terms = defaultdict(list)
                for term in query_terms:
                    postings = st.session_state.inverted_index.get(term, {})
                    for doc_id, freq in postings.items():
                        scores[doc_id] += freq
                        matched_terms[doc_id].append(term)

                st.write(f"Processed query terms: {query_terms}")
                rows = ranked_rows(scores, st.session_state.documents, matched_terms)
                if rows:
                    st.dataframe(pd.DataFrame(rows), use_container_width=True)
                else:
                    st.warning("No documents matched the processed query terms.")

            st.subheader("Dictionary and postings")
            term_filter = st.text_input("Filter terms in index", value="")
            terms = [
                term
                for term in st.session_state.inverted_index
                if term_filter.lower() in term.lower()
            ][:50]
            for term in terms:
                with st.expander(f"{term} | df={len(st.session_state.inverted_index[term])}"):
                    st.dataframe(
                        pd.DataFrame(postings_rows(st.session_state.inverted_index[term], st.session_state.documents)),
                        use_container_width=True,
                    )


elif page == "Phrase Query Processing":
    st.header("Phrase Query Processing")
    if require_documents():
        st.write("Phrase indexes preserve word order. Stop-word removal is off by default because it can create phrases that did not occur contiguously in the original text.")
        phrase_options = normalizer_controls(
            "phrase",
            defaults={
                "lowercase": True,
                "handle_hyphens": True,
                "remove_stopwords": False,
                "method": "stemming",
            },
        )

        if st.button("Build biword and positional indexes"):
            biword, positional, phrase_tokens = build_phrase_indices(st.session_state.documents, phrase_options)
            st.session_state.biword_index = biword
            st.session_state.positional_index = positional
            st.session_state.phrase_tokens = phrase_tokens
            st.success(f"Built {len(biword)} biwords and {len(positional)} positional term entries.")

        if st.session_state.biword_index and st.session_state.positional_index:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Biword index sample**")
                st.dataframe(
                    pd.DataFrame(
                        [
                            {"Biword": term, "Documents": doc_list(postings, st.session_state.documents)}
                            for term, postings in list(st.session_state.biword_index.items())[:20]
                        ]
                    ),
                    use_container_width=True,
                )
            with col2:
                st.markdown("**Positional index sample**")
                positional_rows = []
                for term, postings in list(st.session_state.positional_index.items())[:20]:
                    positional_rows.append(
                        {
                            "Term": term,
                            "Positions by Doc ID": {doc_id: positions for doc_id, positions in postings.items()},
                        }
                    )
                st.dataframe(pd.DataFrame(positional_rows), use_container_width=True)

            query = st.text_input("Enter phrase query", value="machine learning algorithms")
            query_tokens = preprocess_query(query, phrase_options)
            biword_results = search_biword_phrase(
                query_tokens,
                st.session_state.biword_index,
                st.session_state.positional_index,
            )
            positional_results = search_positional_phrase(query_tokens, st.session_state.positional_index)
            false_positives = biword_results - positional_results

            st.write(f"Processed phrase tokens: {query_tokens}")
            col1, col2, col3 = st.columns(3)
            col1.metric("Biword Results", len(biword_results))
            col2.metric("Positional Results", len(positional_results))
            col3.metric("Biword False Positives", len(false_positives))

            result_rows = pd.DataFrame(
                [
                    {
                        "Method": "Biword Index",
                        "Documents": ", ".join(doc_list(biword_results, st.session_state.documents)) or "None",
                    },
                    {
                        "Method": "Positional Index",
                        "Documents": ", ".join(doc_list(positional_results, st.session_state.documents)) or "None",
                    },
                    {
                        "Method": "Biword-only false positives",
                        "Documents": ", ".join(doc_list(false_positives, st.session_state.documents)) or "None",
                    },
                ]
            )
            st.dataframe(result_rows, use_container_width=True)

            st.session_state.exp_results["phrase_query"] = {
                "query": query,
                "query_tokens": query_tokens,
                "biword_count": len(biword_results),
                "positional_count": len(positional_results),
                "false_positive_count": len(false_positives),
            }

            st.info(
                "A biword index answers longer phrase queries by intersecting adjacent word-pair postings. "
                "That is fast, but it can accept a document where the pairs occur in separate places. "
                "The positional index verifies exact offsets, so it is the more accurate phrase-query method."
            )


elif page == "Dictionary Search (BST & B-Tree)":
    st.header("Dictionary Search using BST and B-Tree")
    if require_documents():
        tree_options = normalizer_controls(
            "tree",
            defaults={
                "lowercase": True,
                "handle_hyphens": True,
                "remove_stopwords": True,
                "method": "stemming",
            },
        )

        if st.button("Build dictionary trees"):
            postings, terms = build_term_postings(st.session_state.documents, tree_options)
            items = sorted(postings.items())
            btree = BTree(degree=8)
            for term, term_postings in items:
                btree.insert(term, term_postings)

            st.session_state.term_postings = postings
            st.session_state.terms = terms
            st.session_state.bst = build_balanced_bst(items)
            st.session_state.btree = btree
            st.success(f"Built dictionary trees for {len(terms)} unique terms.")

        if st.session_state.bst and st.session_state.btree:
            col1, col2, col3 = st.columns(3)
            col1.metric("Dictionary Terms", len(st.session_state.terms))
            col2.metric("BST Height", bst_height(st.session_state.bst))
            col3.metric("B-Tree Height", btree_height(st.session_state.btree.root))

            search_term = st.text_input("Search term", value="retrieval")
            processed_terms = preprocess_query(search_term, tree_options)
            key = processed_terms[0] if processed_terms else search_term.lower().strip()

            if key:
                bst_value, bst_query_time = timed_search(lambda term: bst_search(st.session_state.bst, term), key)
                btree_value, btree_query_time = timed_search(lambda term: st.session_state.btree.search(term), key)
                bst_docs, bst_retrieval_time = retrieval_time(bst_value)
                btree_docs, btree_retrieval_time = retrieval_time(btree_value)

                rows = [
                    {
                        "Structure": "BST",
                        "Processed Term": key,
                        "Found": bool(bst_docs),
                        "Doc Frequency": len(bst_docs),
                        "Query Search Time (ms)": round(bst_query_time, 6),
                        "Retrieval Time (ms)": round(bst_retrieval_time, 6),
                        "Total Time (ms)": round(bst_query_time + bst_retrieval_time, 6),
                    },
                    {
                        "Structure": "B-Tree",
                        "Processed Term": key,
                        "Found": bool(btree_docs),
                        "Doc Frequency": len(btree_docs),
                        "Query Search Time (ms)": round(btree_query_time, 6),
                        "Retrieval Time (ms)": round(btree_retrieval_time, 6),
                        "Total Time (ms)": round(btree_query_time + btree_retrieval_time, 6),
                    },
                ]
                st.dataframe(pd.DataFrame(rows), use_container_width=True)

                if bst_docs:
                    st.write("Retrieved documents:", doc_list(set(bst_docs.keys()), st.session_state.documents))

            st.subheader("Multiple-query experiment")
            if st.button("Run experimental table"):
                terms = list(st.session_state.terms)
                existing = terms[: min(50, len(terms))]
                missing = [f"{term}_missing" for term in existing[: min(10, len(existing))]]
                test_terms = existing + missing

                experiment_rows = []
                for term in test_terms:
                    bst_value, bst_query_time = timed_search(lambda item: bst_search(st.session_state.bst, item), term)
                    btree_value, btree_query_time = timed_search(lambda item: st.session_state.btree.search(item), term)
                    bst_docs, bst_retrieval_time = retrieval_time(bst_value)
                    btree_docs, btree_retrieval_time = retrieval_time(btree_value)
                    experiment_rows.append(
                        {
                            "Query": term,
                            "Found": bool(bst_docs or btree_docs),
                            "BST Query Time (ms)": bst_query_time,
                            "BST Retrieval Time (ms)": bst_retrieval_time,
                            "BST Total (ms)": bst_query_time + bst_retrieval_time,
                            "B-Tree Query Time (ms)": btree_query_time,
                            "B-Tree Retrieval Time (ms)": btree_retrieval_time,
                            "B-Tree Total (ms)": btree_query_time + btree_retrieval_time,
                        }
                    )

                df = pd.DataFrame(experiment_rows)
                st.dataframe(df.round(6), use_container_width=True)

                bst_avg = float(df["BST Total (ms)"].mean())
                btree_avg = float(df["B-Tree Total (ms)"].mean())
                faster = "BST" if bst_avg <= btree_avg else "B-Tree"
                col1, col2, col3 = st.columns(3)
                col1.metric("BST Avg Total", f"{bst_avg:.6f} ms")
                col2.metric("B-Tree Avg Total", f"{btree_avg:.6f} ms")
                col3.metric("Faster Structure", faster)

                st.session_state.exp_results["tree"] = {
                    "queries": len(test_terms),
                    "bst_avg": bst_avg,
                    "btree_avg": btree_avg,
                    "faster": faster,
                }


elif page == "Tolerant Retrieval":
    st.header("Tolerant Retrieval")
    if require_documents():
        st.write("The tolerant retrieval vocabulary keeps readable terms, so spelling, wildcard, and Soundex suggestions are easier to inspect.")
        tolerant_options = {
            "lowercase": True,
            "handle_hyphens": True,
            "remove_stopwords": True,
            "method": "none",
        }
        k_value = st.selectbox("K-gram size", [2, 3, 4], index=1)

        if st.button("Build tolerant retrieval indexes"):
            postings, terms = build_term_postings(st.session_state.documents, tolerant_options)
            st.session_state.term_postings = postings
            st.session_state.terms = terms
            st.session_state.k_value = k_value
            st.session_state.kgram_index = build_kgram_index(terms, k_value)
            st.success(f"Built a {k_value}-gram index with {len(st.session_state.kgram_index)} grams.")

        if st.session_state.kgram_index:
            st.subheader("Wildcard retrieval")
            wildcard_query = st.text_input("Wildcard query using *", value="retriev*")
            wildcard_matches = wildcard_search(
                wildcard_query,
                st.session_state.terms,
                st.session_state.kgram_index,
                st.session_state.k_value,
            )
            wildcard_docs = set()
            for term in wildcard_matches:
                wildcard_docs.update(st.session_state.term_postings.get(term, {}).keys())
            st.write("Matching terms:", wildcard_matches[:25])
            st.write("Retrieved documents:", doc_list(wildcard_docs, st.session_state.documents))

            st.subheader("Spelling correction by edit distance")
            misspelled = st.text_input("Misspelled or imperfect term", value="retrival")
            normalized_misspelled = preprocess_query(misspelled, tolerant_options)
            misspelled_key = normalized_misspelled[0] if normalized_misspelled else misspelled.lower()
            suggestions = sorted(
                [(term, edit_distance(misspelled_key, term)) for term in st.session_state.terms],
                key=lambda item: (item[1], item[0]),
            )[:8]
            suggestion_rows = []
            for term, distance in suggestions:
                docs = set(st.session_state.term_postings.get(term, {}).keys())
                suggestion_rows.append(
                    {
                        "Suggestion": term,
                        "Edit Distance": distance,
                        "Documents": ", ".join(doc_list(docs, st.session_state.documents)),
                    }
                )
            st.dataframe(pd.DataFrame(suggestion_rows), use_container_width=True)

            st.subheader("Phonetic correction with Soundex")
            phonetic_query = st.text_input("Phonetic query", value="lern")
            query_code = soundex(phonetic_query)
            phonetic_matches = [
                term for term in st.session_state.terms if soundex(term) == query_code
            ][:20]
            phonetic_docs = set()
            for term in phonetic_matches:
                phonetic_docs.update(st.session_state.term_postings.get(term, {}).keys())
            st.write(f"Soundex code: {query_code}")
            st.write("Phonetic matches:", phonetic_matches)
            st.write("Retrieved documents:", doc_list(phonetic_docs, st.session_state.documents))

            st.session_state.exp_results["tolerant"] = {
                "wildcard_query": wildcard_query,
                "wildcard_terms": len(wildcard_matches),
                "wildcard_docs": len(wildcard_docs),
                "spelling_query": misspelled,
                "best_suggestion": suggestions[0] if suggestions else None,
                "phonetic_query": phonetic_query,
                "phonetic_matches": len(phonetic_matches),
            }

            st.info(
                "Wildcard matching uses the k-gram index to shortlist candidate dictionary terms, edit distance ranks spelling corrections, "
                "and Soundex groups terms with similar pronunciation. The retrieved document list is the union of postings for accepted suggestions."
            )


elif page == "Inference & Discussion":
    st.header("Inference and Discussion")
    st.write("Run the experiments in each module to populate the evidence below. Static limitations and improvements are included for the report.")

    stem_data = st.session_state.exp_results.get("stem_vs_lemma")
    phrase_data = st.session_state.exp_results.get("phrase_query")
    tree_data = st.session_state.exp_results.get("tree")
    tolerant_data = st.session_state.exp_results.get("tolerant")

    st.subheader("1. Which preprocessing technique improved retrieval quality?")
    if stem_data:
        st.info(
            f"For query '{stem_data['query']}', {stem_data['better']} produced the stronger or equal cosine-similarity evidence. "
            "Lowercasing, stop-word removal, and hyphen splitting reduced noisy vocabulary before the final normalizer was applied."
        )
    else:
        st.warning("Run the stemming-vs-lemmatization comparison in Text Preprocessing.")

    st.subheader("2. Was stemming or lemmatization better for this dataset?")
    if stem_data:
        st.dataframe(pd.DataFrame(stem_data["comparison_rows"]), use_container_width=True)
        st.info(
            f"Based on the average top-3 cosine score, {stem_data['better']} is the better choice for the current query and collection. "
            "Stemming usually reduces vocabulary more aggressively, while lemmatization keeps more readable dictionary forms."
        )
    else:
        st.warning("No stemming-vs-lemmatization result is available yet.")

    st.subheader("3. Which phrase query index was more accurate?")
    if phrase_data:
        st.info(
            f"For query '{phrase_data['query']}', the biword index returned {phrase_data['biword_count']} document(s), "
            f"the positional index returned {phrase_data['positional_count']} document(s), and {phrase_data['false_positive_count']} biword-only false positive(s) were observed. "
            "The positional index is more accurate because it checks exact offsets for all terms."
        )
    else:
        st.warning("Run a phrase query after building the phrase indexes.")

    st.subheader("4. Which tree structure was faster?")
    if tree_data:
        st.info(
            f"Across {tree_data['queries']} queries, BST average total time was {tree_data['bst_avg']:.6f} ms and "
            f"B-Tree average total time was {tree_data['btree_avg']:.6f} ms. The faster structure in this in-memory experiment was {tree_data['faster']}."
        )
    else:
        st.warning("Run the experimental table in Dictionary Search.")

    st.subheader("5. How tolerant was the retrieval model?")
    if tolerant_data:
        best_suggestion = tolerant_data["best_suggestion"]
        suggestion_text = f"{best_suggestion[0]} at edit distance {best_suggestion[1]}" if best_suggestion else "none"
        st.info(
            f"Wildcard query '{tolerant_data['wildcard_query']}' matched {tolerant_data['wildcard_terms']} term(s) and retrieved {tolerant_data['wildcard_docs']} document(s). "
            f"The best spelling suggestion for '{tolerant_data['spelling_query']}' was {suggestion_text}. "
            f"Soundex query '{tolerant_data['phonetic_query']}' matched {tolerant_data['phonetic_matches']} term(s)."
        )
    else:
        st.warning("Build tolerant retrieval indexes and try wildcard, edit-distance, and phonetic queries.")

    st.subheader("6. Limitations")
    st.info(
        "- The system is intended for small to medium text collections in memory.\n"
        "- It does not include user relevance judgments, so retrieval-quality comparison uses cosine similarity as a proxy.\n"
        "- Soundex is English-oriented and weak for many technical terms.\n"
        "- Edit-distance suggestions become expensive on very large vocabularies.\n"
        "- The app does not parse PDF or DOCX collections as retrieval documents."
    )

    st.subheader("7. Improvements")
    st.info(
        "- Add BM25 or TF-IDF ranking for the main search results.\n"
        "- Add Boolean queries, proximity queries, and query expansion.\n"
        "- Store indexes on disk for larger collections.\n"
        "- Add relevance feedback and evaluation metrics such as precision, recall, and MAP when labels are available.\n"
        "- Add document parsers for PDF, DOCX, and HTML."
    )
