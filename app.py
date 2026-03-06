import os
import re
import json
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
import difflib
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_ollama import OllamaLLM
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

load_dotenv()

app = Flask(__name__)
CORS(app)

# ================================
# CONFIG & MODELS (STRICTLY LOCAL)
# ================================
# In MoSPI production, these paths would point to a local 'models/' directory
BI_ENCODER_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
OLLAMA_MODEL = "llama3:8b" # Government compliant local LLM

# MO SPI SYNONYM MAP (Meeting Safety)
ACRONYM_MAP = {
    "gdp": "gross domestic product gva national accounts nas",
    "wpi": "wholesale price index inflation primary articles",
    "cpi": "consumer price index inflation rural urban",
    "iip": "index of industrial production manufacturing mining electricity",
    "asi": "annual survey of industries factories employment profit",
    "nas": "national accounts statistics gdp gva saving",
    "nss": "national sample survey drinking water latrine",
    "pus": "periodic labour force survey unemployment lfpr",
    "plfs": "periodic labour force survey unemployment lfpr"
}

print(f"[INFO] Loading Enterprise Models from Hugging Face / Local Cache...")
try:
    # BGE-M3 supports Dense, Sparse, and Multi-vector. We'll start with Dense + BM25 for stability.
    bi_encoder = SentenceTransformer(BI_ENCODER_MODEL)
    cross_encoder = CrossEncoder(RERANKER_MODEL)
    print(f"[INFO] Models loaded successfully.")
except Exception as e:
    print(f"[ERROR] Model loading failed: {e}")

llm = OllamaLLM(model=OLLAMA_MODEL)

# ================================
# DATA LOADING
# ================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PRODUCTS_FILE = os.path.join(BASE_DIR, "products", "products.json")

with open(PRODUCTS_FILE, "r", encoding="utf-8", errors="ignore") as f:
    raw_products = json.load(f)

DATASETS = []
INDICATORS = []
FILTERS = []

for ds_name, ds_info in raw_products.get("datasets", {}).items():
    DATASETS.append({"code": ds_name, "name": ds_name})
    for ind in ds_info.get("indicators", []):
        ind_code = f"{ds_name}_{ind['name']}"
        
        # Metadata Augmentation: Collect filter options to enrich keywords
        extra_keywords = []
        for f_dict in ind.get("filters", []):
            for f_name, options in f_dict.items():
                if f_name.lower() in ["indicator", "item", "group", "category", "subgroup", "subgroup - 1"]:
                    extra_keywords.extend([str(o) for o in options if str(o).lower() != "select all"])
        
        enrichment = " ".join(extra_keywords[:50])
        kw = f"{ds_name} {ind['name']} {ind.get('description', '')} {enrichment}".lower()
        if ds_name == "NAS" or ds_name == "ASI":
            print(f"[DEBUG] {ind_code} Keywords: {kw[:100]}...")
            
        INDICATORS.append({
            "code": ind_code,
            "name": ind["name"],
            "desc": ind.get("description", ""),
            "parent": ds_name,
            "keywords": kw
        })
        # Correctly parse MoSPI dictionary-based filters
        for f_dict in ind.get("filters", []):
            for f_name, options in f_dict.items():
                for opt in options:
                    FILTERS.append({
                        "parent": ind_code,
                        "filter_name": f_name,
                        "option": str(opt)
                    })

# ================================
# HYBRID INDEXING (BM25 + DENSE)
# ================================
print(f"[INFO] Indexing {len(INDICATORS)} indicators for Hybrid Search...")

# 1. BM25 Index (Lexical/Keyword)
tokenized_corpus = [i["keywords"].split() for i in INDICATORS]
bm25 = BM25Okapi(tokenized_corpus)

# 2. Dense Index (FAISS/Memory)
indicator_texts = [i["keywords"] for i in INDICATORS]
embeddings = bi_encoder.encode(indicator_texts, convert_to_numpy=True, show_progress_bar=True)
embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

# ================================
# ENHANCED LOGIC
# ================================

def clean_text(t):
    t = (t or "").lower()
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()

def map_year_to_option(query, options, dataset=None):
    """
    MoSPI Specific Year Mapping (Production Grade)
    """
    q_lower = query.lower()
    # Extract only the 20xx part
    match = re.search(r"\b(20\d{2})\b", q_lower)
    if not match:
        return None
    
    target_year = match.group(1)
    
    # Priority 1: Exact Match (e.g. "2022")
    for opt in options:
        if opt["option"] == target_year:
            return opt

    # Priority 2: Fiscal Year Start Match (e.g. "2021" matches "2021-22")
    # This is critical for NAS/WPI
    for opt in options:
        if opt["option"].startswith(target_year):
            return opt
            
    # Priority 3: Fiscal Year End Match (e.g. "2022" matches "2021-22")
    for opt in options:
        if opt["option"].endswith(target_year):
            return opt

    return None

def select_best_filter_option(query, raw_query, filter_name, options, cross_encoder):
    if not options: return None
    
    fname_lower = filter_name.lower()
    q_lower = (raw_query or "").lower()
    
    # SPECIAL: Year/Financial Year logic
    if "year" in fname_lower:
        match = map_year_to_option(q_lower, options)
        if match: return match

    # Standard Fuzzy/Semantic Match
    opt_texts = [str(o["option"]).lower() for o in options]
    
    # 1. Exact Word Match (Fast)
    for i, opt_text in enumerate(opt_texts):
        if len(opt_text) > 2 and opt_text in q_lower:
            return options[i]
            
    # 2. Semantic Rerank (Precise)
    pairs = [[raw_query, f"{filter_name} {opt}"] for opt in opt_texts]
    scores = cross_encoder.predict(pairs)
    best_idx = int(np.argmax(scores))
    
    if scores[best_idx] > 0.3: # Confidence threshold
        return options[best_idx]
        
    return options[0] # Default to 1st (Select All)

def resolve_filters(query, raw_query, indicator_code):
    related = [f for f in FILTERS if f["parent"] == indicator_code]
    if not related: return {}
    
    grouped = {}
    for f in related:
        grouped.setdefault(f["filter_name"], []).append(f)
        
    results = {}
    for fname, opts in grouped.items():
        best = select_best_filter_option(query, raw_query, fname, opts, cross_encoder)
        if best:
            results[fname] = best["option"]
            
    return results

def rewrite_query_with_llm(query):
    """
    Normalization for Government Queries (Safe & Restricted)
    """
    # Optimized prompt for Llama 3 local
    prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n" \
             f"You are a MoSPI indicator expert. Simplify the query by extracting core product, indicator, and year. " \
             f"Strictly keep acronyms like GDP, WPI, ASI, NAS. Output only the clean terms.<|eot_id|>\n" \
             f"<|start_header_id|>user<|end_header_id|>\n{query}<|eot_id|>\n" \
             f"<|start_header_id|>assistant<|end_header_id|>\n"
    try:
        # Use a short timeout for local inference
        response = llm.invoke(prompt).strip()
        # Remove markdown/extra text
        response = response.split('\n')[0].strip()
        return response if response else query
    except:
        return query

def enterprise_hybrid_search(query, raw_query=None, top_k=20):
    q_clean = clean_text(query)
    q_tokens = q_clean.split()
    
    # Check original query too for domain routing (Meeting Safety)
    raw_clean = clean_text(raw_query or query)
    raw_tokens = raw_clean.split()
    lookup_tokens = list(set(q_tokens + raw_tokens))
    
    # 1. Query Expansion (Synconyms for Meeting Safety)
    expanded_tokens = list(q_tokens)
    for token in lookup_tokens:
        if token in ACRONYM_MAP:
            expanded_tokens.extend(ACRONYM_MAP[token].split())
    
    # 2. Lexical Score (BM25)
    bm25_scores = bm25.get_scores(expanded_tokens)
    
    # 3. Semantic Score (Dense)
    q_emb = bi_encoder.encode([q_clean], convert_to_numpy=True)
    q_emb /= np.linalg.norm(q_emb, axis=1, keepdims=True)
    semantic_scores = np.dot(embeddings, q_emb.T).flatten()
    
    # 4. Hybrid Combination
    alpha = 0.5 
    mx_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1
    combined_scores = alpha * (bm25_scores / mx_bm25) + (1 - alpha) * semantic_scores
    
    # 5. Hard Domain Routing (Meeting-Safe Feature)
    forced_parent = None
    for token in lookup_tokens:
        if token == "gdp": forced_parent = "NAS"
        elif token == "wpi": forced_parent = "WPI"
        elif token == "iip": forced_parent = "IIP"
        elif token == "asi": forced_parent = "ASI"
        elif token == "cpi": forced_parent = "CPI"
        elif token in ["plfs", "pus"]: forced_parent = "PLFS"

    results = []
    for idx, score in enumerate(combined_scores):
        ind = INDICATORS[idx]
        final_score = float(score)
        
        # Apply massive boost if it's the forced product
        if forced_parent and ind["parent"] == forced_parent:
            final_score += 10.0 # Unbeatable boost
        
        results.append({
            **ind,
            "hybrid_score": final_score
        })
    
    # Sort and take top_k
    results = sorted(results, key=lambda x: x["hybrid_score"], reverse=True)[:top_k]

    # Filter out non-forced results if routing is active (Maximum Safety)
    if forced_parent:
        results = [r for r in results if r["parent"] == forced_parent]
    
    # 5. Neural Reranking (BGE-Reranker)
    if results:
        # Pass enriched keywords to reranker for better context
        pairs = [[query, f"{r['parent']} {r['name']} {r['keywords']}"] for r in results]
        rerank_scores = cross_encoder.predict(pairs)
        for i, r in enumerate(results):
            r["final_score"] = float(rerank_scores[i])
        
        results = sorted(results, key=lambda x: x["final_score"], reverse=True)
        
    return results

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    raw_query = data.get('query', '')
    
    # Step 1: LLM Rewrite
    clean_q = rewrite_query_with_llm(raw_query)
    print(f"[DEBUG] Raw: {raw_query} | Clean: {clean_q}")
    
    # Step 2: Hybrid Search + Rerank
    top_results = enterprise_hybrid_search(clean_q, raw_query=raw_query)
    
    if not top_results:
        return jsonify({"error": "No results found"}), 404
        
    best = top_results[0]
    
    # Step 3: Filter Extraction
    filters = resolve_filters(clean_q, raw_query, best["code"])
    
    return jsonify({
        "dataset": best["parent"],
        "indicator": best["name"],
        "confidence": round(float(best.get("final_score", 0)), 2),
        "filters": filters
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5010, debug=True)
