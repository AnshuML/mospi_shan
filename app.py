from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os, json, re
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
from datetime import datetime
import difflib

# ================================
# CONFIG
# ================================
USE_QDRANT = True
try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qmodels
except Exception:
    USE_QDRANT = False

# ================================
# LLM (QUERY REWRITER ONLY)
# ================================
from langchain_ollama import ChatOllama

try:
    rewriter_llm = ChatOllama(
        model="llama3:70b",
        base_url="http://localhost:11434",
        temperature=0.3
    )

    rewriter_llm.invoke("ping")
    print(" Ollama is running")

except Exception as e:
    print(" Ollama is not running")


# ================================
# REGEX
# ================================
YEAR_PATTERN = re.compile(r"\b(20\d{2})\b")

# ================================
# HELPERS
# ================================
def clean_text(t):
    t = (t or "").lower()
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()

def normalize_confidence(scores, min_conf=50, max_conf=95):
    if not scores:
        return []
    mn, mx = min(scores), max(scores)
    if mn == mx:
        return [min_conf] * len(scores)
    return [round(min_conf + (s - mn)/(mx - mn)*(max_conf - min_conf), 2) for s in scores]



#########
BASE_YEAR_PATTERN = re.compile(r"(20\d{2})")

def detect_base_year(query):
    q = query.lower()

    if "base year" or " base" in q:
        m = BASE_YEAR_PATTERN.search(q)
        if m:
            return int(m.group(1))

    return None


def resolve_cpi_conflict(results, query):
    """
    Only when CPI and CPI2 both present in top results
    """
    datasets = [r["parent"] for r in results]

    if "CPI" not in datasets or "CPI2" not in datasets:
        return results  # kuch mat chhedo

    base_year = detect_base_year(query)

    # ---------- case 1: user ne base year bola ----------
    if base_year:
        if base_year >= 2024:
            # CPI2 rakho
            return [r for r in results if r["parent"] != "CPI"]
        else:
            # CPI rakho
            return [r for r in results if r["parent"] != "CPI2"]

    # ---------- case 2: base year nahi bola ----------
    return [r for r in results if r["parent"] != "CPI"]


# ================================
# LLM QUERY REWRITE
# ================================
def rewrite_query_with_llm(user_query):
    prompt =  f"""
You are a QUERY NORMALIZATION ENGINE for a data analytics system.

Task:
Rewrite the user query safely with controlled semantic normalization.

STRICT RULES:
1. DO NOT add any new information
2. DO NOT infer missing filters
3. DO NOT assume any category
4. DO NOT enrich meaning
5. ONLY rewrite words that already exist in the query
6. NEVER inject new concepts
7. NEVER add sector/gender/state unless explicitly present
8. Output ONLY rewritten query
9. No explanation
10.If the query contains a known dataset short form (CPI, IIP, NAS, PLFS, ASI, HCES, NSS, AISHE, WPI, TUS, NFHS, ENVSTAT, EC, ESI), append its full form in the rewritten query while keeping the short form unchanged (e.g., "CPI" → "CPI Consumer Price Index"), and do not expand anything not explicitly present.


SPECIAL RULE (VERY IMPORTANT):

If the query contains "IIP" and also contains any month name 
(January–December or short forms like Jan, Feb, etc.), 
then add the word "monthly" to the query.

If query contains both "year" and "base year", clearly separate them:


Examples:
"IIP July data" → "IIP monthly July data"
"IIP for December" → "IIP monthly December"
"IIP Aug 2022" → "IIP monthly Aug 2022"
"gdp for year 2023-24 base year 2022-23" → "gdp year:2023-24 base_year:2022-23"

DO NOT apply this rule to any other dataset.
If query is about CPI, GDP, PLFS etc → do nothing.

FREQUENCY NORMALIZATION RULE (VERY IMPORTANT):

At the END of your rewritten query, append a frequency tag in the format [FREQ:value].
Detect the user's INTENDED data frequency from the query:

- If user explicitly says "quarterly", "quartarly", "quarter", "Q1", "Q2", "Q3", "Q4", "jul-sep", "oct-dec", "jan-mar", "apr-jun" → append [FREQ:quarterly]
- If user explicitly says "annually", "annual", "yearly" → append [FREQ:annually]
- If user explicitly says "monthly" as a FREQUENCY (e.g. "monthly data", "give monthly", "show monthly") → append [FREQ:monthly]
- If user mentions a specific month name (January, February, etc.) → append [FREQ:monthly]
- IMPORTANT: If "monthly" is used as an ADJECTIVE before salary/earnings/wage/income/pay (e.g. "monthly salary", "monthly earnings", "monthly wage"), this is NOT a frequency. Do NOT append [FREQ:monthly] for this.
- If no frequency is mentioned or unclear → do NOT append any [FREQ:] tag

Examples:
"Female average monthly salary quarterly 2023-24 female urban Bihar" → "female average monthly salary quarterly 2023-24 female urban Bihar [FREQ:quarterly]"
"Female average monthly salary 2023-24 female urban Bihar" → "female average monthly salary 2023-24 female urban Bihar"
"give me monthly data for PLFS" → "give me monthly data for PLFS Periodic Labour Force Survey [FREQ:monthly]"
"CPI data January 2024" → "CPI Consumer Price Index data January 2024 [FREQ:monthly]"
"PLFS annual data 2023-24" → "PLFS Periodic Labour Force Survey annual data 2023-24 [FREQ:annually]"
"Female average monthly salary in the public sector of frequency(quarterly),year(2023-24)" → "female average monthly salary in the public sector frequency quarterly year 2023-24 [FREQ:quarterly]"




ALLOWED OPERATIONS:
- spelling correction
- grammar correction
- casing normalization
- synonym normalization
- semantic mapping ONLY if the word exists explicitly in text

CRITICAL RULE (VERY IMPORTANT):
- If the user query is ONLY a dataset or product name
  (examples: IIP, CPI, CPIALRL, HCES, ASI, NAS, PLFS, CPI2, AISHE, WPI, TUS, NFHS, ENVSTAT, EC, EC4, EC5, EC6, ESI, NSS77, NSS78, NSS79C, ASUSE),
  then RETURN THE QUERY EXACTLY AS IT IS.
- Dataset names must NEVER be replaced with normal English words.


STRICT SEMANTIC MAP (ONLY IF WORD EXISTS):
- gao, gaon, village → rural
- shehar, city, metro → urban
- purush, aadmi, mard, man, men → male
- mahila, aurat, lady, women → female
- ladka → male
- ladki → female

❌ FORBIDDEN:
- Do NOT infer urban from city names
- Do NOT infer rural from state names
- Do NOT infer gender from profession
- Do NOT infer sector from geography
- Do NOT add any category automatically

Examples:
RAW: "mens judge in village"
→ "male judge in rural"

RAW: "Gini Coefficient for urban india in 2023-24"
→ "Gini Coefficient for urban in 2023-24"

RAW: "factory output gujrat 2022"
→ "factory output Gujarat 2022"

RAW: "men judges in delhi"
→ "male judges in Delhi"

RAW: "factory output in gujrat for 2022 in gao"
→ "factory output in Gujarat for 2022 in rural"

RAW: "data for mahila workers"
→ "data for female workers"

RAW: "gaon ke factory worker"
→ "rural factory worker"

RAW: "factory output in mumbai"
→ "factory output in Mumbai"

User Query:
"{user_query}"
"""
    try:
        out = rewriter_llm.invoke(prompt).content.strip()
        out = out.replace('"', '').replace("\n", " ").strip()
        return out
    except:
        return user_query

# ================================
# YEAR NORMALIZATION
# ================================
def normalize_year_string(s):
    return re.sub(r"[^0-9]", "", str(s))


def map_year_to_option(user_year, options):
    y = int(user_year)

    # WPI explicit check: prioritize exact calendar year match first
    if options and options[0].get("parent") == "WPI":
        norm_options = {normalize_year_string(o["option"]): o for o in options}
        exact_match = str(y)
        if exact_match in norm_options:
            return norm_options[exact_match]

    targets = [
         f"{y}{y+1}",            # → "20232024"
        f"{y}{str(y+1)[-2:]}",  # → "202324"  ← NEW!
        f"{y-1}{y}",            # → "20222023"
        f"{y-1}{str(y)[-2:]}",  # → "202223"  ← NEW!
        str(y)                   # → "2023"
    ]
    norm_options = {normalize_year_string(o["option"]): o for o in options}
    for t in targets:
        if t in norm_options:
            return norm_options[t]
    return None

# ================================
# UNIVERSAL FILTER NORMALIZER
# ================================
def universal_filter_normalizer(ind_code, filters_json):
    flat = []
    def recurse(key, value):
        if isinstance(value, list) and all(isinstance(x, str) for x in value):
            for opt in value:
                flat.append({"parent": ind_code,"filter_name": key,"option": opt})
        elif isinstance(value, list) and all(isinstance(x, dict) for x in value):
            for item in value:
                for k, v in item.items():
                    if k.lower() in ["name", "title", "label"]:
                        flat.append({"parent": ind_code,"filter_name": key,"option": v})
                    else:
                        recurse(k, v)
        elif isinstance(value, dict):
            for k, v in value.items():
                recurse(k, v)

    for f in filters_json:
        if isinstance(f, dict):
            for k, v in f.items():
                recurse(k, v)
    return flat


#############LLM 
# ================================
# SMART FILTER ENGINE
# ================================
def select_best_filter_option(query, filter_name, options, cross_encoder,raw_query=None):
    q_lower = query.lower()
    raw_lower=(raw_query or query).lower()
    fname_lower = filter_name.lower()
     
    # =========================
    # FREQUENCY FILTER
    # =========================
    if fname_lower in ["frequency"]:

        # --- PRIORITY 1: Check LLM [FREQ:xxx] tag (NEW CHANGE) ---
        freq_tag_match = re.search(r'\[freq:\s*(\w+)\]', q_lower)
        if freq_tag_match:
            llm_freq = freq_tag_match.group(1).strip()
            for opt in options:
                if opt["option"].lower().startswith(llm_freq) or llm_freq.startswith(opt["option"].lower()):
                    return opt

        # --- PRIORITY 2: Detect if "monthly" is adjective (NEW CHANGE) ---
        monthly_as_adjective = False
        if "monthly" in q_lower or "monthly" in raw_lower:
            monthly_adj_patterns = [
                r"monthly\s+(salary|salaries|earning|earnings|income|wage|wages|pay|expenditure|consumption|average|gross)",
                r"(average|mean|total|gross)\s+monthly",
            ]
            check_text = q_lower + " " + raw_lower
            if any(re.search(p, check_text) for p in monthly_adj_patterns):
                monthly_as_adjective = True

        # --- Check for explicit mention (checks both LLM query and raw query) ---
        for keyword in ["annually", "quarterly", "monthly", "annual"]:
            if keyword == "monthly" and monthly_as_adjective:
                continue  # skip — "monthly" is adjective here, not frequency
            if keyword in q_lower or keyword in raw_lower:
                for opt in options:
                    if opt["option"].lower().startswith(keyword) or keyword.startswith(opt["option"].lower()):
                        return opt

        # --- Fuzzy match for misspelled frequency (e.g. "quartely" → "quarterly") ---
        for keyword in ["annually", "annual", "quarterly", "monthly"]:
            if keyword == "monthly" and monthly_as_adjective:
                continue
            for word in q_lower.split() + raw_lower.split():
                if difflib.SequenceMatcher(None, keyword, word).ratio() > 0.80:
                    for opt in options:
                        if opt["option"].lower().startswith(keyword) or keyword.startswith(opt["option"].lower()):
                            return opt

        # --- Month names → Monthly (full names only to avoid "may" false positive) ---
        month_names = [
            "january", "february", "march", "april", "june",
            "july", "august", "september", "october", "november", "december"
        ]
        if any(m in q_lower for m in month_names):
            for opt in options:
                if opt["option"].lower() in ["monthly", "month"]:
                    return opt

        # --- Quarter keywords → Quarterly ---
        quarter_keywords = ["quarter", "quarterly", "q1", "q2", "q3", "q4",
                            "jul-sep", "oct-dec", "jan-mar", "apr-jun"]
        if any(qk in q_lower for qk in quarter_keywords):
            for opt in options:
                if opt["option"].lower() in ["quarterly"]:
                    return opt

        # --- Year format "2023-24" or standalone year → Annually ---
        if re.search(r"\d{4}[-/]\d{2,4}", q_lower) or YEAR_PATTERN.search(q_lower):
            for opt in options:
                if opt["option"].lower() in ["annually", "annual"]:
                    return opt

        # --- No frequency clue → Select All ---
        return {
            "parent": options[0]["parent"],
            "filter_name": filter_name,
            "option" : options[0]["option"]
        }

    # =========================
    # YEAR FILTER
    # =========================
    if "year" in fname_lower and "base" not in fname_lower:
        year_match = YEAR_PATTERN.search(q_lower)

        # user ne year nahi bola → Select All
        if not year_match:
            return {
                "parent": options[0]["parent"],
                "filter_name": filter_name,
                "option": "Select All"
            }

        user_year = year_match.group(1)

        mapped = map_year_to_option(user_year, options)
        if mapped:
            return mapped

        pairs = [(query, f"{filter_name} {o['option']}") for o in options]
        scores = cross_encoder.predict(pairs)
        return options[int(np.argmax(scores))]

    # =========================
    # BASE YEAR FILTER (FINAL FIX)
    # =========================
    if "base" in fname_lower and "year" in fname_lower:

        # 🔹 check if user explicitly mentioned base year
        for opt in options:
            opt_text = str(opt["option"]).lower()
            if opt_text in q_lower:
                return opt

        # 🔹 user ne base year nahi bola → latest base year pick karo
        def extract_start_year(opt):
            m = re.search(r"\d{4}", str(opt["option"]))
            return int(m.group(0)) if m else 0

        latest = max(options, key=lambda o: extract_start_year(o))
        return latest

    # =========================
    # OTHER FILTERS
    # =========================
    mentioned = []

    for opt in options:
        opt_text = str(opt.get("option", "")).lower().strip()
        if not opt_text:
            continue

        if opt_text in q_lower:
            mentioned.append(opt)
            continue

        #for word in q_lower.split():
            #if difflib.SequenceMatcher(None, opt_text, word).ratio() > 0.80:
                #mentioned.append(opt)
                #break
        # Safe synonym mapping ONLY for specific filters like Religion
        synonyms = {
            "muslim": "islam",
            "muslims": "islam",
            "islamic": "islam",
            "hindu": "hinduism",
            "hindi": "hinduism",
            "sikh": "sikhism",
            "christian": "christianity"
        }

        for word in q_lower.split():
            if word in synonyms:
                mapped = synonyms[word]
                # Safe prefix match ONLY if word is in our dictionary
                if opt_text.startswith(mapped):
                    mentioned.append(opt)
                    break
           
            # Standard fuzzy match for everything else (Old logic intact)
            if difflib.SequenceMatcher(None, opt_text, word).ratio() > 0.80:
                mentioned.append(opt)
                break
    if mentioned:
        pairs = [(query, f"{filter_name} {o['option']}") for o in mentioned]
        scores = cross_encoder.predict(pairs)
        return mentioned[int(np.argmax(scores))]

    return {
        "parent": options[0]["parent"],
        "filter_name": filter_name,
        "option": "Select All"
    }


# ================================
# LOAD PRODUCTS
# ================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PRODUCTS_FILE = os.path.join(BASE_DIR, "products", "products.json")

with open(PRODUCTS_FILE, "r", encoding="utf-8", errors="ignore") as f:
    raw_products = json.load(f)

DATASETS, INDICATORS, FILTERS = [], [], []

for ds_name, ds_info in raw_products.get("datasets", {}).items():
    DATASETS.append({"code": ds_name, "name": ds_name})

    for ind in ds_info.get("indicators", []):
        ind_code = f"{ds_name}_{ind['name']}"
        INDICATORS.append({
            "code": ind_code,
            "name": ind["name"],
            "desc": ind.get("description", ""),
            "parent": ds_name
        })

        flat = universal_filter_normalizer(ind_code, ind.get("filters", []))
        FILTERS.extend(flat)

print(f"[INFO] DATASETS={len(DATASETS)}, INDICATORS={len(INDICATORS)}, FILTERS={len(FILTERS)}")

# ================================
# MODELS
# ================================
bi_encoder = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")

# ================================
# VECTOR DB
# ================================
VECTOR_DIM = bi_encoder.get_sentence_embedding_dimension()
COLLECTION = "indicators_collection"

qclient = None
faiss_index = None

if USE_QDRANT:
    try:
        qclient = QdrantClient(url="http://localhost:6333")
        if COLLECTION not in [c.name for c in qclient.get_collections().collections]:
            qclient.recreate_collection(
                collection_name=COLLECTION,
                vectors_config=qmodels.VectorParams(size=VECTOR_DIM,distance=qmodels.Distance.COSINE)
            )
        print("[INFO] Qdrant ready")
    except Exception as e:
        USE_QDRANT = False
        print("[WARN] Qdrant failed, using FAISS:", e)

# IMPROVED: Prefix indicator name with parent dataset name for better semantic association (e.g. "WPI Wholesale price of Moong")
names = [clean_text(i["parent"] + " " + i["name"]) for i in INDICATORS]
descs = [clean_text(i.get("desc", "")) for i in INDICATORS]

embeddings = (0.4 * bi_encoder.encode(names, convert_to_numpy=True) + 0.6 * bi_encoder.encode(descs, convert_to_numpy=True))
embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

if USE_QDRANT and qclient:
    qclient.upsert(
        collection_name=COLLECTION,
        points=[qmodels.PointStruct(id=i,vector=embeddings[i].tolist(),payload=INDICATORS[i]) for i in range(len(INDICATORS))]
    )
else:
    faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
    faiss_index.add(embeddings.astype("float32"))

# ================================
# SEARCH
# ================================
def search_indicators(query, top_k=25, max_products=3):
    q_vec = bi_encoder.encode([clean_text(query)], convert_to_numpy=True)
    q_vec /= np.linalg.norm(q_vec, axis=1, keepdims=True)

    if USE_QDRANT and qclient:
        hits = qclient.query_points(
            collection_name=COLLECTION,
            query=q_vec[0].tolist(),
            limit=top_k
        ).points
        candidates = [h.payload for h in hits]
    else:
        _, I = faiss_index.search(q_vec.astype("float32"), top_k)
        candidates = [INDICATORS[i] for i in I[0] if i >= 0]

    scores = cross_encoder.predict([(query, c["name"] + " " + c.get("desc", "")) for c in candidates])
    for i, c in enumerate(candidates):
        c["score"] = float(scores[i])

    candidates.sort(key=lambda x: x["score"], reverse=True)

    # CPI conflict resolve ONLY if both present
    candidates = resolve_cpi_conflict(candidates, query)

    # ----------------------------------------------------
    # NEW: WPI-SPECIFIC RERANKING & DISAMBIGUATION
    # ----------------------------------------------------
    q_lower = query.lower()
    is_wpi_query = "wpi" in q_lower or "wholesale" in q_lower
    
    if is_wpi_query:
        # Re-rank WPI indicators by checking if query words match their specific filters
        for c in candidates:
            if c["parent"] == "WPI":
                ind_code = c["code"]
                ind_filters = [f for f in FILTERS if f["parent"] == ind_code]
                
                match_bonus = 0
                for filt in ind_filters:
                    opt_lower = str(filt.get("option", "")).lower()
                    # Assign a strong bonus if a specific filter item exactly appears in query
                    if len(opt_lower) > 3 and opt_lower in q_lower:
                        match_bonus += 0.6
                
                c["score"] += match_bonus
        
        # Suppress CPI returning when WPI explicitly asked for
        candidates = [c for c in candidates if c["parent"] not in ["CPI", "CPIALRL"]]
        
        # Re-sort after bonus application
        candidates.sort(key=lambda x: x["score"], reverse=True)
    
    # --- GENERIC DATASET BOOST ---
    # Strong boost if the user explicitly mentioned a dataset name or code
    q_words = set(q_lower.split())
    
    # Map common keywords to dataset codes
    dataset_keywords = {
        "wholesale": "WPI",
        "wpi": "WPI",
        "consumer": "CPI",
        "cpi": "CPI",
        "labour": "PLFS",
        "plfs": "PLFS",
        "factory": "ASI",
        "asi": "ASI",
        "national": "NAS",
        "nas": "NAS",
        "iip": "IIP"
    }
    
    mentioned_codes = set()
    for word in q_words:
        if word in dataset_keywords:
            mentioned_codes.add(dataset_keywords[word].lower())
    
    # Also check if any full dataset code is in the query (case insensitive)
    for ds in DATASETS:
        code_low = ds["code"].lower()
        if code_low in q_lower:
            mentioned_codes.add(code_low)

    if mentioned_codes:
        for c in candidates:
            if c["parent"].lower() in mentioned_codes:
                # Strong boost (2.0) to ensure it tops the list if explicitly requested
                c["score"] += 2.0
        # Re-sort again
        candidates.sort(key=lambda x: x["score"], reverse=True)
    # ----------------------------------------------------

    seen, final = set(), []
    for c in candidates:

        if c["parent"] not in seen:
            seen.add(c["parent"])
            final.append(c)
        if len(final) == max_products:
            break


    return final




###################query capture 


import uuid
from datetime import datetime

LOG_FILE = os.path.join(BASE_DIR, "logs", "queries.jsonl")

def save_query_log(raw_query, rewritten_query, response_json):
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

    record = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.utcnow().isoformat(),
        "raw_query": raw_query,
        "rewritten_query": rewritten_query,
        "response": response_json
    }

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ================================
# FLASK
# ================================
app = Flask(__name__, template_folder="templates")
CORS(app)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    raw_q = request.json.get("query", "").strip()
    if not raw_q:
        return jsonify({"error": "query required"}), 400

    #  LLM rewrite
    q = rewrite_query_with_llm(raw_q)

    print("RAW :", raw_q)
    print("LLM :", q)

    top_results = search_indicators(q)
    confidences = normalize_confidence([r["score"] for r in top_results])

    results = []

    for ind, conf in zip(top_results, confidences):
        # 500 Fix: Use fallback if dataset code not found in DATASETS
        dataset = next((d for d in DATASETS if d["code"] == ind["parent"]), {"name": ind["parent"], "code": ind["parent"]})
        related_filters = [f for f in FILTERS if f["parent"] == ind["code"]]

        grouped = {}
        for f in related_filters:
            grouped.setdefault(f["filter_name"], []).append(f)

        best_filters = []
        for fname, opts in grouped.items():
            best_opt = select_best_filter_option(
                query=q,
                raw_query=raw_q,
                filter_name=fname,
                options=opts,
                cross_encoder=cross_encoder
            )
            best_filters.append({
                "filter_name": fname,
                "option": best_opt["option"]
            })

        results.append({
            "dataset": dataset["name"],
            "indicator": ind["name"],
            "confidence": conf,
            "filters": best_filters
        })
    response = {"results": results}
        #  SAVE OUTPUT
    save_query_log(
        raw_query=raw_q,
        rewritten_query=q,
        response_json=response
    )

    #return jsonify(response)

    return jsonify({"results": results})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5009)
