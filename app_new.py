

# from flask import Flask, request, jsonify, render_template
# from flask_cors import CORS
# import os, json, re
# import numpy as np
# from sentence_transformers import SentenceTransformer, CrossEncoder
# import faiss

# USE_QDRANT = True
# try:
#     from qdrant_client import QdrantClient
#     from qdrant_client.http import models as qmodels
# except Exception:
#     USE_QDRANT = False
#     faiss = None

# from datetime import datetime
# import difflib

# YEAR_PATTERN = re.compile(r"\b(20\d{2})\b")

# DEFAULT_KEYWORDS = {
#     "select all",
#     "all",
#     "rural+urban",
#     "all ages",
#     "annual",
#     "annually"
# }


# def select_best_filter_option(
#     query,
#     filter_name,
#     options,
#     cross_encoder
# ):
#     q_lower = query.lower()
#     fname_lower = filter_name.lower()
#     current_year = datetime.now().year

#     # ---------------- Year special handling ----------------
#     if fname_lower == "year":
#         query_has_year = bool(YEAR_PATTERN.search(q_lower))

#         def extract_year(opt):
#             text = str(opt.get("option", ""))
#             m = YEAR_PATTERN.search(text)
#             return int(m.group(1)) if m else None

#         #  If user did NOT mention a year → latest valid year
#         if not query_has_year:
#             valid_years = []
#             for opt in options:
#                 y = extract_year(opt)
#                 if y is not None and y <= current_year:
#                     valid_years.append((y, opt))

#             if valid_years:
#                 return max(valid_years, key=lambda t: t[0])[1]

#         #  Fallback to ML (same as your other file)
#         pairs = [(query, f"{filter_name} {o['option']}") for o in options]
#         scores = cross_encoder.predict(pairs)
#         return options[int(np.argmax(scores))]

#     # ---------------- All other filters ----------------
#     mentioned = []

#     for opt in options:
#         opt_text = str(opt.get("option", "")).strip()
#         if not opt_text:
#             continue

#         if opt_text.lower() in q_lower:
#             mentioned.append(opt)
#         else:
#             for word in q_lower.split():
#                 if difflib.SequenceMatcher(
#                     None, opt_text.lower(), word
#                 ).ratio() > 0.8:
#                     mentioned.append(opt)
#                     break

#     if mentioned:
#         pairs = [(query, f"{filter_name} {o['option']}") for o in mentioned]
#         scores = cross_encoder.predict(pairs)
#         return mentioned[int(np.argmax(scores))]

#     # 🔹 Default option fallback
#     def is_default(opt):
#         return str(opt.get("option", "")).strip().lower() in DEFAULT_KEYWORDS

#     defaults = [o for o in options if is_default(o)]
#     if defaults:
#         return defaults[0]

#     #  Final ML fallback
#     pairs = [(query, f"{filter_name} {o['option']}") for o in options]
#     scores = cross_encoder.predict(pairs)
#     return options[int(np.argmax(scores))]

# # =========================================================
# # HELPERS
# # =========================================================
# def clean_text(t):
#     t = (t or "").lower()
#     t = re.sub(r"[^a-z0-9\s]", " ", t)
#     return re.sub(r"\s+", " ", t).strip()


# def normalize_confidence(scores, min_conf=50, max_conf=95):
#     if not scores:
#         return []
#     mn, mx = min(scores), max(scores)
#     if mn == mx:
#         return [min_conf] * len(scores)
#     return [
#         round(min_conf + (s - mn) / (mx - mn) * (max_conf - min_conf), 2)
#         for s in scores
#     ]


# # =========================================================
# # IIP NORMALIZER (NEW HIERARCHY SUPPORT)
# # =========================================================
# def normalize_iip_filters(dataset_name, indicator_name, indicator_json):
#     """
#     Flattens new IIP hierarchy into standard FILTERS format
#     """
#     flat_filters = []
#     ind_code = f"{dataset_name}_{indicator_name}"

#     for base in indicator_json.get("indicators1", []):
#         base_year = base.get("name")

#         flat_filters.append({
#             "parent": ind_code,
#             "filter_name": "Base Year",
#             "option": base_year
#         })

#         for freq in base.get("Indicator2", []):
#             freq_name = freq.get("name")

#             flat_filters.append({
#                 "parent": ind_code,
#                 "filter_name": "Frequency",
#                 "option": freq_name
#             })

#             for f in freq.get("filters", []):

#                 # Financial Year
#                 if "financial_year" in f:
#                     for y in f["financial_year"]:
#                         flat_filters.append({
#                             "parent": ind_code,
#                             "filter_name": "Year",
#                             "option": y
#                         })

#                 # Type → Category → SubCategory
#                 if "type" in f:
#                     for t in f["type"]:
#                         type_name = t.get("name")

#                         flat_filters.append({
#                             "parent": ind_code,
#                             "filter_name": "Type",
#                             "option": type_name
#                         })

#                         for cat in t.get("Category", []):
#                             cat_name = cat.get("name")

#                             flat_filters.append({
#                                 "parent": ind_code,
#                                 "filter_name": "Category",
#                                 "option": cat_name
#                             })

#                             for sub in cat.get("SubCategory", []):
#                                 flat_filters.append({
#                                     "parent": ind_code,
#                                     "filter_name": "SubCategory",
#                                     "option": sub
#                                 })

#     return flat_filters


# # =========================================================
# # LOAD PRODUCTS
# # =========================================================
# PRODUCTS_FILE = os.path.join("products", "products.json")
# with open(PRODUCTS_FILE, "r", encoding="utf-8", errors="ignore") as f:
#     raw_products = json.load(f)

# DATASETS, INDICATORS, FILTERS = [], [], []

# for ds_name, ds_info in raw_products.get("datasets", {}).items():
#     DATASETS.append({"code": ds_name, "name": ds_name})

#     for ind in ds_info.get("indicators", []):

#         ind_code = f"{ds_name}_{ind['name']}"
#         INDICATORS.append({
#             "code": ind_code,
#             "name": ind["name"],
#             "desc": ind.get("description", ""),
#             "parent": ds_name
#         })

#         #  IIP new hierarchy
#         if ds_name == "IIP" and "indicators1" in ind:
#             FILTERS.extend(
#                 normalize_iip_filters(ds_name, ind["name"], ind)
#             )

#         #  All other products (old flat structure)
#         else:
#             for f in ind.get("filters", []):
#                 if isinstance(f, dict):
#                     for fname, options in f.items():
#                         for opt in options:
#                             FILTERS.append({
#                                 "parent": ind_code,
#                                 "filter_name": fname,
#                                 "option": opt
#                             })

# print(f"[INFO] Datasets={len(DATASETS)}, Indicators={len(INDICATORS)}, Filters={len(FILTERS)}")


# # =========================================================
# # MODELS
# # =========================================================
# bi_encoder = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")
# cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")


# # =========================================================
# # VECTOR DB SETUP
# # =========================================================
# VECTOR_DIM = bi_encoder.get_sentence_embedding_dimension()
# COLLECTION = "indicators_collection"

# qclient = None
# faiss_index = None

# if USE_QDRANT:
#     try:
#         qclient = QdrantClient(url="http://localhost:6333")
#         if COLLECTION not in [c.name for c in qclient.get_collections().collections]:
#             qclient.recreate_collection(
#                 collection_name=COLLECTION,
#                 vectors_config=qmodels.VectorParams(
#                     size=VECTOR_DIM,
#                     distance=qmodels.Distance.COSINE
#                 )
#             )
#         print("[INFO] Qdrant ready")
#     except Exception as e:
#         USE_QDRANT = False
#         print("[WARN] Qdrant failed, using FAISS:", e)


# # =========================================================
# # INDEX INDICATORS
# # =========================================================
# names = [clean_text(i["name"]) for i in INDICATORS]
# descs = [clean_text(i.get("desc", "")) for i in INDICATORS]

# embeddings = (
#     0.3 * bi_encoder.encode(names, convert_to_numpy=True)
#     + 0.7 * bi_encoder.encode(descs, convert_to_numpy=True)
# )
# embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

# if USE_QDRANT and qclient:
#     qclient.upsert(
#         collection_name=COLLECTION,
#         points=[
#             qmodels.PointStruct(
#                 id=i,
#                 vector=embeddings[i].tolist(),
#                 payload=INDICATORS[i]
#             )
#             for i in range(len(INDICATORS))
#         ]
#     )
# else:
#     faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
#     faiss_index.add(embeddings.astype("float32"))


# # =========================================================
# # SEARCH LOGIC (TOP-3 UNIQUE PRODUCTS)
# # =========================================================
# def search_indicators(query, top_k=25, max_products=3):
#     q_vec = bi_encoder.encode([clean_text(query)], convert_to_numpy=True)
#     q_vec /= np.linalg.norm(q_vec, axis=1, keepdims=True)

#     candidates = []

#     if USE_QDRANT and qclient:
#         hits = qclient.search(
#             collection_name=COLLECTION,
#             query_vector=q_vec[0].tolist(),
#             limit=top_k
#         )
#         candidates = [h.payload for h in hits]
#     else:
#         _, I = faiss_index.search(q_vec.astype("float32"), top_k)
#         candidates = [INDICATORS[i] for i in I[0] if i >= 0]

#     if not candidates:
#         return []

#     scores = cross_encoder.predict(
#         [(query, c["name"] + " " + c.get("desc", "")) for c in candidates]
#     )

#     for i, c in enumerate(candidates):
#         c["score"] = float(scores[i])

#     candidates.sort(key=lambda x: x["score"], reverse=True)

#     seen, final = set(), []
#     for c in candidates:
#         if c["parent"] not in seen:
#             seen.add(c["parent"])
#             final.append(c)
#         if len(final) == max_products:
#             break

#     return final


# # =========================================================
# # FLASK APP
# # =========================================================
# app = Flask(__name__, template_folder="templates")
# CORS(app)


# @app.route("/")
# def home():
#     return render_template("index.html")


# @app.route("/predict", methods=["POST"])
# #@app.route("/search/predict", methods=["POST"])
# def predict():
#     q = request.json.get("query", "").strip()
#     if not q:
#         return jsonify({"error": "query required"}), 400

#     top_results = search_indicators(q)
#     confidences = normalize_confidence([r["score"] for r in top_results])

#     results = []

#     for ind, conf in zip(top_results, confidences):
#         dataset = next(d for d in DATASETS if d["code"] == ind["parent"])
#         related_filters = [f for f in FILTERS if f["parent"] == ind["code"]]

#         grouped = {}
#         for f in related_filters:
#             grouped.setdefault(f["filter_name"], []).append(f)

#         best_filters = []
#         for fname, opts in grouped.items():
#             best_opt = select_best_filter_option(
#                 query=q,
#                 filter_name=fname,
#                 options=opts,
#                 cross_encoder=cross_encoder
#             )
#             best_filters.append({
#               "filter_name": fname,
#               "option": best_opt["option"]
#         })

#         results.append({
#             "dataset": dataset["name"],
#             "indicator": ind["name"],
#             "confidence": conf,
#             "filters": best_filters
#         })

#     return jsonify({"results": results})


# if __name__ == "__main__":
#     app.run(debug=True, host="0.0.0.0", port=5009)

# # ####### aman kumar --------
 








# from flask import Flask, request, jsonify, render_template
# from flask_cors import CORS
# import os, json, re
# import numpy as np
# from sentence_transformers import SentenceTransformer, CrossEncoder
# import faiss
# from datetime import datetime

# # =========================================================
# # LLM (OLLAMA 70B)
# # =========================================================
# from langchain_ollama import ChatOllama

# llm = ChatOllama(
#     model="llama3:70b",
#     base_url="http://localhost:11434",
#     temperature=0,
#     request_timeout=180
# )

# # =========================================================
# # HELPERS
# # =========================================================
# YEAR_PATTERN = re.compile(r"\b(20\d{2})\b")

# def clean_text(t):
#     t = (t or "").lower()
#     t = re.sub(r"[^a-z0-9\s]", " ", t)
#     return re.sub(r"\s+", " ", t).strip()


# def normalize_confidence(scores, min_conf=40, max_conf=95):
#     if not scores:
#         return []
#     mn, mx = min(scores), max(scores)
#     if mn == mx:
#         return [min_conf] * len(scores)
#     return [
#         round(min_conf + (s - mn) / (mx - mn) * (max_conf - min_conf), 2)
#         for s in scores
#     ]


# def normalize_llm_value(value, options):
#     if value is None:
#         return None
#     v = str(value).strip().lower()
#     for opt in options:
#         if str(opt).strip().lower() == v:
#             return opt
#     return None


# # =========================================================
# # LOAD PRODUCTS.JSON
# # =========================================================
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# PRODUCTS_FILE = os.path.join(BASE_DIR, "products", "products.json")



# def universal_filter_normalizer(ind_code, filters_json):
#     flat = []

#     def recurse(key, value):
#         # Flat list
#         if isinstance(value, list) and all(isinstance(x, str) for x in value):
#             for opt in value:
#                 flat.append({
#                     "parent": ind_code,
#                     "filter_name": key,
#                     "option": opt
#                 })

#         # List of dicts → hierarchy
#         elif isinstance(value, list) and all(isinstance(x, dict) for x in value):
#             for item in value:
#                 for k, v in item.items():
#                     if k.lower() in ["name", "title", "label"]:
#                         flat.append({
#                             "parent": ind_code,
#                             "filter_name": key,
#                             "option": v
#                         })
#                     else:
#                         recurse(k, v)

#         # Dict → go deeper
#         elif isinstance(value, dict):
#             for k, v in value.items():
#                 recurse(k, v)

#     for f in filters_json:
#         if isinstance(f, dict):
#             for k, v in f.items():
#                 recurse(k, v)

#     return flat


# with open(PRODUCTS_FILE, "r", encoding="utf-8") as f:
#     raw_products = json.load(f)

# DATASETS, INDICATORS, FILTERS = [], [], []

# for ds_name, ds_info in raw_products.get("datasets", {}).items():
#     DATASETS.append({"code": ds_name, "name": ds_name})
#     for ind in ds_info.get("indicators", []):
#         ind_code = f"{ds_name}_{ind['name']}"
#         INDICATORS.append({
#             "code": ind_code,
#             "name": ind["name"],
#             "desc": ind.get("description", ""),
#             "parent": ds_name
#         })
#         # for f in ind.get("filters", []):
#         #     if isinstance(f, dict):
#         #         for fname, options in f.items():
#         #             for opt in options:
#         #                 FILTERS.append({
#         #                     "parent": ind_code,
#         #                     "filter_name": fname,
#         #                     "option": opt
#         #                 })
#         # 🔥 THIS LINE
#         flat = universal_filter_normalizer(ind_code, ind.get("filters", []))
#         FILTERS.extend(flat)
# print(f"[INFO] Indicators={len(INDICATORS)}, Filters={len(FILTERS)}")

# # =========================================================
# # SEMANTIC MODELS (INDICATOR SEARCH ONLY)
# # =========================================================
# bi_encoder = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")
# cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")

# # =========================================================
# # FAISS INDEX
# # =========================================================
# VECTOR_DIM = bi_encoder.get_sentence_embedding_dimension()
# faiss_index = faiss.IndexFlatL2(VECTOR_DIM)

# names = [clean_text(i["name"]) for i in INDICATORS]
# descs = [clean_text(i.get("desc", "")) for i in INDICATORS]

# embeddings = (
#     0.3 * bi_encoder.encode(names, convert_to_numpy=True)
#   + 0.7 * bi_encoder.encode(descs, convert_to_numpy=True)
# )
# embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
# faiss_index.add(embeddings.astype("float32"))




# ##########



# # =========================================================
# # INDICATOR SEARCH
# # =========================================================
# def search_indicators(query, top_k=20, max_products=3):
#     q_vec = bi_encoder.encode([clean_text(query)], convert_to_numpy=True)
#     q_vec /= np.linalg.norm(q_vec, axis=1, keepdims=True)

#     _, I = faiss_index.search(q_vec.astype("float32"), top_k)
#     candidates = [INDICATORS[i] for i in I[0] if i >= 0]

#     scores = cross_encoder.predict(
#         [(query, c["name"] + " " + c.get("desc", "")) for c in candidates]
#     )

#     for i, c in enumerate(candidates):
#         c["score"] = float(scores[i])

#     candidates.sort(key=lambda x: x["score"], reverse=True)

#     final, seen = [], set()
#     for c in candidates:
#         if c["parent"] not in seen:
#             final.append(c)
#             seen.add(c["parent"])
#         if len(final) == max_products:
#             break

#     return final

# # =========================================================
# #  LLM FILTER RESOLUTION (ADVANCED)
# # =========================================================
# def resolve_filters_with_llm(user_query, grouped_filters):
#     current_year = datetime.now().year

#     prompt = f"""
# You are an INTENT EXTRACTION ENGINE for a DATA ANALYTICS SYSTEM.

# User Query:
# "{user_query}"

# Available filters and allowed values:
# {json.dumps(grouped_filters, ensure_ascii=False)}

# IMPORTANT RULES:
# 1. If user mentions a specific value → select it
# 2. If user mentions a YEAR RANGE or "last N years":
#    - Return all matching years (comma separated)
# 3. If user {user_query} does NOT ask for a filter from {grouped_filters}→ return "Select All" 
# 4. Handle spelling mistakes and synonyms
# 5. Use ONLY provided options
# 6. NEVER explain
# 7. Output STRICT JSON ONLY
# 8. if user not ask directly for sector in the {user_query} then always return "Select All" for the sector
# 9. Analyse my question : {user_query} and the filters {grouped_filters} , if question is not having any relavant filter_name then select all for that filter
# 10. predominantly do not use any condition for sector 
# 11. Make sure 
# Examples:
# - "last 2 years" → "2022, 2023" 
# - "cloth price in 2023" → "2023"
# - "village area" → Rural
# - "metro city" → Urban

# Return JSON ONLY.
# """

#     try:
#         raw = llm.invoke(prompt).content
#     except Exception as e:
#         print("[WARN] LLM failed:", e)
#         return {k: "Select All" for k in grouped_filters}

#     try:
#         start = raw.index("{")
#         end = raw.rindex("}") + 1
#         parsed = json.loads(raw[start:end])
#     except Exception:
#         parsed = {}

#     final = {}

#     for fname, options in grouped_filters.items():
#         llm_val = parsed.get(fname)

#         # ---------- MULTI YEAR SUPPORT ----------
#         if isinstance(llm_val, str) and "," in llm_val:
#             years = []
#             for y in llm_val.split(","):
#                 y = y.strip()
#                 if y in options:
#                     years.append(y)
#             final[fname] = ", ".join(years) if years else "Select All"
#             continue

#         normalized = normalize_llm_value(llm_val, options)
#         final[fname] = normalized if normalized else "Select All"

#     return final

# # =========================================================
# # FLASK APP
# # =========================================================
# app = Flask(__name__)
# CORS(app)
# @app.route("/")
# def home():
#     return render_template("index.html")
# @app.route("/predict", methods=["POST"])
# def predict():
#     q = request.json.get("query", "").strip()
#     if not q:
#         return jsonify({"error": "query required"}), 400

#     top_results = search_indicators(q)
#     confidences = normalize_confidence([r["score"] for r in top_results])

#     results = []

#     for ind, conf in zip(top_results, confidences):
#         dataset = next(d for d in DATASETS if d["code"] == ind["parent"])
#         related_filters = [f for f in FILTERS if f["parent"] == ind["code"]]

#         grouped = {}
#         for f in related_filters:
#             grouped.setdefault(f["filter_name"], []).append(str(f["option"]))

#         resolved_filters = resolve_filters_with_llm(q, grouped)

#         results.append({
#             "dataset": dataset["name"],
            
#             "indicator": ind["name"],
#             "confidence": conf,
#             "filters": [
#                 {"filter_name": k, "option": v}
#                 for k, v in resolved_filters.items()
#             ]
#         })

#     return jsonify({"results": results})


# if __name__ == "__main__":
#     app.run(debug=True, host="0.0.0.0", port=5009)













# from flask import Flask, request, jsonify, render_template
# from flask_cors import CORS
# import os, json, re
# import numpy as np
# from datetime import datetime
# from sentence_transformers import SentenceTransformer, CrossEncoder
# import faiss
# from difflib import get_close_matches

# # =========================================================
# # LLM (OLLAMA)
# # =========================================================
# from langchain_ollama import ChatOllama

# llm = ChatOllama(
#     model="llama3:70b",
#     base_url="http://localhost:11434",
#     temperature=0,
#     request_timeout=180
# )

# # =========================================================
# # REGEX
# # =========================================================
# YEAR_PATTERN = re.compile(r"\b(20\d{2})\b")
# LAST_N_PATTERN = re.compile(r"last\s+(\d+)\s+year")
# MONTHS = ["january","february","march","april","may","june",
#           "july","august","september","october","november","december"]

# # =========================================================
# # HELPERS
# # =========================================================
# def clean_text(t):
#     t = (t or "").lower()
#     t = re.sub(r"[^a-z0-9\s]", " ", t)
#     return re.sub(r"\s+", " ", t).strip()

# def normalize_confidence(scores, min_conf=40, max_conf=95):
#     if not scores:
#         return []
#     mn, mx = min(scores), max(scores)
#     if mn == mx:
#         return [min_conf]*len(scores)
#     return [round(min_conf+(s-mn)/(mx-mn)*(max_conf-min_conf),2) for s in scores]

# def fuzzy_match(val, options):
#     if not val: 
#         return None
#     matches = get_close_matches(val.lower(), [o.lower() for o in options], n=1, cutoff=0.7)
#     if matches:
#         for o in options:
#             if o.lower() == matches[0]:
#                 return o
#     return None

# # =========================================================
# # YEAR ENGINE
# # =========================================================
# def normalize_year_string(s):
#     return re.sub(r"[^0-9]", "", s)

# def smart_year_match(year, options):
#     y = int(year)
#     targets = [f"{y}{y+1}", f"{y-1}{y}", str(y)]
#     norm = {normalize_year_string(o):o for o in options}
#     for t in targets:
#         if t in norm:
#             return norm[t]
#     return None

# def last_n_years(n, options, current_year):
#     res=[]
#     norm = {normalize_year_string(o):o for o in options}
#     for i in range(n):
#         y=current_year-i
#         for t in [f"{y}{y+1}", f"{y-1}{y}", str(y)]:
#             if t in norm:
#                 res.append(norm[t]); break
#     return res

# # =========================================================
# # UNIVERSAL FILTER NORMALIZER
# # =========================================================
# def universal_filter_normalizer(ind_code, filters_json):
#     flat=[]
#     def recurse(key,value):
#         if isinstance(value,list) and all(isinstance(x,str) for x in value):
#             for opt in value:
#                 flat.append({"parent":ind_code,"filter_name":key,"option":opt})
#         elif isinstance(value,list) and all(isinstance(x,dict) for x in value):
#             for item in value:
#                 for k,v in item.items():
#                     if k.lower() in ["name","title","label"]:
#                         flat.append({"parent":ind_code,"filter_name":key,"option":v})
#                     else:
#                         recurse(k,v)
#         elif isinstance(value,dict):
#             for k,v in value.items():
#                 recurse(k,v)
#     for f in filters_json:
#         if isinstance(f,dict):
#             for k,v in f.items():
#                 recurse(k,v)
#     return flat

# # =========================================================
# # LOAD PRODUCTS.JSON
# # =========================================================
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# PRODUCTS_FILE = os.path.join(BASE_DIR, "products", "products.json")

# with open(PRODUCTS_FILE,"r",encoding="utf-8") as f:
#     raw_products=json.load(f)

# DATASETS,INDICATORS,FILTERS=[],[],[]

# for ds_name,ds_info in raw_products.get("datasets",{}).items():
#     DATASETS.append({"code":ds_name,"name":ds_name})
#     for ind in ds_info.get("indicators",[]):
#         ind_code=f"{ds_name}_{ind['name']}"
#         INDICATORS.append({
#             "code":ind_code,
#             "name":ind["name"],
#             "desc":ind.get("description",""),
#             "parent":ds_name
#         })
#         flat=universal_filter_normalizer(ind_code, ind.get("filters",[]))
#         FILTERS.extend(flat)

# print(f"[OK] Indicators={len(INDICATORS)} Filters={len(FILTERS)}")

# # =========================================================
# # MODELS
# # =========================================================
# bi_encoder = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")
# cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")

# # =========================================================
# # FAISS
# # =========================================================
# VECTOR_DIM = bi_encoder.get_sentence_embedding_dimension()
# faiss_index = faiss.IndexFlatL2(VECTOR_DIM)

# names=[clean_text(i["name"]) for i in INDICATORS]
# descs=[clean_text(i.get("desc","")) for i in INDICATORS]

# embeddings=(0.3*bi_encoder.encode(names,convert_to_numpy=True)+
#             0.7*bi_encoder.encode(descs,convert_to_numpy=True))
# embeddings/=np.linalg.norm(embeddings,axis=1,keepdims=True)
# faiss_index.add(embeddings.astype("float32"))

# # =========================================================
# # SEARCH
# # =========================================================
# def search_indicators(query, top_k=20, max_products=3):
#     q_vec=bi_encoder.encode([clean_text(query)],convert_to_numpy=True)
#     q_vec/=np.linalg.norm(q_vec,axis=1,keepdims=True)
#     _,I=faiss_index.search(q_vec.astype("float32"),top_k)
#     candidates=[INDICATORS[i] for i in I[0] if i>=0]
#     scores=cross_encoder.predict([(query,c["name"]+" "+c.get("desc","")) for c in candidates])
#     for i,c in enumerate(candidates): c["score"]=float(scores[i])
#     candidates.sort(key=lambda x:x["score"],reverse=True)
#     final,seen=[],set()
#     for c in candidates:
#         if c["parent"] not in seen:
#             final.append(c); seen.add(c["parent"])
#         if len(final)==max_products: break
#     return final

# # =========================================================
# # SMART FILTER ENGINE
# # =========================================================
# def resolve_filters(user_query, grouped_filters):
#     current_year=datetime.now().year
#     ql=user_query.lower()
#     years_found=YEAR_PATTERN.findall(user_query)
#     last_n=LAST_N_PATTERN.search(ql)
#     found_month=None
#     for m in MONTHS:
#         if m in ql:
#             found_month=m.capitalize()

#     # ---- LLM semantic intent ----
#     prompt=f"""
# You are an intent extraction engine.

# User Query:
# "{user_query}"

# Filters:
# {json.dumps(grouped_filters,ensure_ascii=False)}

# Rules:
# - Select only if user intent clear
# - If not asked → Select All
# - Use allowed values only
# - Output JSON only
# """
#     try:
#         raw=llm.invoke(prompt).content
#         raw=raw[raw.find("{"):raw.rfind("}")+1]
#         llm_out=json.loads(raw)
#     except:
#         llm_out={}

#     final={}
#     for fname,options in grouped_filters.items():
#         fl=fname.lower()
#         selected=None

#         # ---- YEAR ----
#         if "year" in fl:
#             if last_n:
#                 yrs=last_n_years(int(last_n.group(1)), options, current_year)
#                 selected=", ".join(yrs) if yrs else "Select All"
#             elif years_found:
#                 mapped=smart_year_match(years_found[0], options)
#                 selected=mapped if mapped else "Select All"
#             else:
#                 selected="Select All"

#         # ---- MONTH ----
#         elif "month" in fl:
#             if found_month:
#                 match=fuzzy_match(found_month, options)
#                 selected=match if match else "Select All"
#             else:
#                 selected="Select All"

#         # ---- OTHER FILTERS ----
#         else:
#             llm_val=llm_out.get(fname)
#             if llm_val:
#                 exact=fuzzy_match(str(llm_val), options)
#                 selected=exact if exact else "Select All"
#             else:
#                 selected="Select All"

#         final[fname]=selected

#     return final

# # =========================================================
# # FLASK
# # =========================================================
# app=Flask(__name__)
# CORS(app)

# @app.route("/")
# def home():
#     return render_template("index.html")

# @app.route("/predict",methods=["POST"])
# def predict():
#     q=request.json.get("query","").strip()
#     if not q:
#         return jsonify({"error":"query required"}),400

#     top_results=search_indicators(q)
#     confidences=normalize_confidence([r["score"] for r in top_results])
#     results=[]

#     for ind,conf in zip(top_results,confidences):
#         dataset=next(d for d in DATASETS if d["code"]==ind["parent"])
#         related=[f for f in FILTERS if f["parent"]==ind["code"]]

#         grouped={}
#         for f in related:
#             grouped.setdefault(f["filter_name"],[]).append(str(f["option"]))

#         resolved=resolve_filters(q, grouped)

#         results.append({
#             "dataset":dataset["name"],
#             "indicator":ind["name"],
#             "confidence":conf,
#             "filters":[{"filter_name":k,"option":v} for k,v in resolved.items()]
#         })

#     return jsonify({"results":results})

# if __name__=="__main__":
#     app.run(debug=True,host="0.0.0.0",port=5009)




# from flask import Flask, request, jsonify, render_template
# from flask_cors import CORS
# import os, json, re
# import numpy as np
# from sentence_transformers import SentenceTransformer, CrossEncoder
# import faiss
# from datetime import datetime
# import difflib

# # ================================
# # CONFIG
# # ================================
# USE_QDRANT = True
# try:
#     from qdrant_client import QdrantClient
#     from qdrant_client.http import models as qmodels
# except Exception:
#     USE_QDRANT = False

# # ================================
# # REGEX
# # ================================
# YEAR_PATTERN = re.compile(r"\b(20\d{2})\b")

# # ================================
# # HELPERS
# # ================================
# def clean_text(t):
#     t = (t or "").lower()
#     t = re.sub(r"[^a-z0-9\s]", " ", t)
#     return re.sub(r"\s+", " ", t).strip()


# def normalize_confidence(scores, min_conf=50, max_conf=95):
#     if not scores:
#         return []
#     mn, mx = min(scores), max(scores)
#     if mn == mx:
#         return [min_conf] * len(scores)
#     return [round(min_conf + (s - mn)/(mx - mn)*(max_conf - min_conf), 2) for s in scores]


# # ================================
# # YEAR NORMALIZATION
# # ================================
# def normalize_year_string(s):
#     return re.sub(r"[^0-9]", "", str(s))


# def map_year_to_option(user_year, options):
#     """
#     2022 -> 2022-23 or 2021-22
#     """
#     y = int(user_year)

#     targets = [
#         f"{y}{y+1}",     # 20222023
#         f"{y-1}{y}",     # 20212022
#         str(y)           # 2022
#     ]

#     norm_options = {normalize_year_string(o["option"]): o for o in options}

#     for t in targets:
#         if t in norm_options:
#             return norm_options[t]

#     return None


# # ================================
# # UNIVERSAL FILTER NORMALIZER
# # ================================
# def universal_filter_normalizer(ind_code, filters_json):
#     flat = []

#     def recurse(key, value):
#         # list of strings
#         if isinstance(value, list) and all(isinstance(x, str) for x in value):
#             for opt in value:
#                 flat.append({
#                     "parent": ind_code,
#                     "filter_name": key,
#                     "option": opt
#                 })

#         # list of dicts
#         elif isinstance(value, list) and all(isinstance(x, dict) for x in value):
#             for item in value:
#                 for k, v in item.items():
#                     if k.lower() in ["name", "title", "label"]:
#                         flat.append({
#                             "parent": ind_code,
#                             "filter_name": key,
#                             "option": v
#                         })
#                     else:
#                         recurse(k, v)

#         # dict
#         elif isinstance(value, dict):
#             for k, v in value.items():
#                 recurse(k, v)

#     for f in filters_json:
#         if isinstance(f, dict):
#             for k, v in f.items():
#                 recurse(k, v)

#     return flat


# # ================================
# # SMART FILTER ENGINE (NO LLM)
# # ================================
# def select_best_filter_option(query, filter_name, options, cross_encoder):
#     q_lower = query.lower()
#     fname_lower = filter_name.lower()

#     # ---------- YEAR ----------
#     if fname_lower == "year":
#         year_match = YEAR_PATTERN.search(q_lower)

#         # Rule: user ne year nahi bola → Select All
#         if not year_match:
#             return {
#                 "parent": options[0]["parent"],
#                 "filter_name": filter_name,
#                 "option": "Select All"
#             }

#         user_year = year_match.group(1)
#         mapped = map_year_to_option(user_year, options)

#         if mapped:
#             return mapped

#         # semantic fallback
#         pairs = [(query, f"{filter_name} {o['option']}") for o in options]
#         scores = cross_encoder.predict(pairs)
#         return options[int(np.argmax(scores))]

#     # ---------- OTHER FILTERS ----------
#     mentioned = []

#     for opt in options:
#         opt_text = str(opt.get("option", "")).lower().strip()
#         if not opt_text:
#             continue

#         # direct match
#         if opt_text in q_lower:
#             mentioned.append(opt)
#             continue

#         # fuzzy spelling match
#         for word in q_lower.split():
#             if difflib.SequenceMatcher(None, opt_text, word).ratio() > 0.80:
#                 mentioned.append(opt)
#                 break

#     # semantic ranking
#     if mentioned:
#         pairs = [(query, f"{filter_name} {o['option']}") for o in mentioned]
#         scores = cross_encoder.predict(pairs)
#         return mentioned[int(np.argmax(scores))]

#     # Rule: user ne filter hi nahi manga
#     return {
#         "parent": options[0]["parent"],
#         "filter_name": filter_name,
#         "option": "Select All"
#     }


# # ================================
# # LOAD PRODUCTS
# # ================================
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# PRODUCTS_FILE = os.path.join(BASE_DIR, "products", "products.json")

# with open(PRODUCTS_FILE, "r", encoding="utf-8", errors="ignore") as f:
#     raw_products = json.load(f)

# DATASETS, INDICATORS, FILTERS = [], [], []

# for ds_name, ds_info in raw_products.get("datasets", {}).items():
#     DATASETS.append({"code": ds_name, "name": ds_name})

#     for ind in ds_info.get("indicators", []):
#         ind_code = f"{ds_name}_{ind['name']}"

#         INDICATORS.append({
#             "code": ind_code,
#             "name": ind["name"],
#             "desc": ind.get("description", ""),
#             "parent": ds_name
#         })

#         # universal flattening (works for any hierarchy)
#         flat = universal_filter_normalizer(ind_code, ind.get("filters", []))
#         FILTERS.extend(flat)

# print(f"[INFO] DATASETS={len(DATASETS)}, INDICATORS={len(INDICATORS)}, FILTERS={len(FILTERS)}")

# # ================================
# # MODELS
# # ================================
# bi_encoder = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")
# cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")

# # ================================
# # VECTOR DB
# # ================================
# VECTOR_DIM = bi_encoder.get_sentence_embedding_dimension()
# COLLECTION = "indicators_collection"

# qclient = None
# faiss_index = None

# if USE_QDRANT:
#     try:
#         qclient = QdrantClient(url="http://localhost:6333")
#         if COLLECTION not in [c.name for c in qclient.get_collections().collections]:
#             qclient.recreate_collection(
#                 collection_name=COLLECTION,
#                 vectors_config=qmodels.VectorParams(
#                     size=VECTOR_DIM,
#                     distance=qmodels.Distance.COSINE
#                 )
#             )
#         print("[INFO] Qdrant ready")
#     except Exception as e:
#         USE_QDRANT = False
#         print("[WARN] Qdrant failed, using FAISS:", e)

# # ================================
# # INDEXING
# # ================================
# names = [clean_text(i["name"]) for i in INDICATORS]
# descs = [clean_text(i.get("desc", "")) for i in INDICATORS]

# embeddings = (
#     0.3 * bi_encoder.encode(names, convert_to_numpy=True)
#     + 0.7 * bi_encoder.encode(descs, convert_to_numpy=True)
# )
# embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

# if USE_QDRANT and qclient:
#     qclient.upsert(
#         collection_name=COLLECTION,
#         points=[
#             qmodels.PointStruct(
#                 id=i,
#                 vector=embeddings[i].tolist(),
#                 payload=INDICATORS[i]
#             )
#             for i in range(len(INDICATORS))
#         ]
#     )
# else:
#     faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
#     faiss_index.add(embeddings.astype("float32"))

# # ================================
# # SEARCH
# # ================================
# def search_indicators(query, top_k=25, max_products=3):
#     q_vec = bi_encoder.encode([clean_text(query)], convert_to_numpy=True)
#     q_vec /= np.linalg.norm(q_vec, axis=1, keepdims=True)

#     candidates = []

#     if USE_QDRANT and qclient:
#         hits = qclient.search(
#             collection_name=COLLECTION,
#             query_vector=q_vec[0].tolist(),
#             limit=top_k
#         )
#         candidates = [h.payload for h in hits]
#     else:
#         _, I = faiss_index.search(q_vec.astype("float32"), top_k)
#         candidates = [INDICATORS[i] for i in I[0] if i >= 0]

#     scores = cross_encoder.predict(
#         [(query, c["name"] + " " + c.get("desc", "")) for c in candidates]
#     )

#     for i, c in enumerate(candidates):
#         c["score"] = float(scores[i])

#     candidates.sort(key=lambda x: x["score"], reverse=True)

#     seen, final = set(), []
#     for c in candidates:
#         if c["parent"] not in seen:
#             seen.add(c["parent"])
#             final.append(c)
#         if len(final) == max_products:
#             break

#     return final


# # ================================
# # FLASK
# # ================================
# app = Flask(__name__, template_folder="templates")
# CORS(app)

# @app.route("/")
# def home():
#     return render_template("index.html")

# @app.route("/predict", methods=["POST"])
# def predict():
#     q = request.json.get("query", "").strip()
#     if not q:
#         return jsonify({"error": "query required"}), 400

#     top_results = search_indicators(q)
#     confidences = normalize_confidence([r["score"] for r in top_results])

#     results = []

#     for ind, conf in zip(top_results, confidences):
#         dataset = next(d for d in DATASETS if d["code"] == ind["parent"])
#         related_filters = [f for f in FILTERS if f["parent"] == ind["code"]]

#         grouped = {}
#         for f in related_filters:
#             grouped.setdefault(f["filter_name"], []).append(f)

#         best_filters = []
#         for fname, opts in grouped.items():
#             best_opt = select_best_filter_option(
#                 query=q,
#                 filter_name=fname,
#                 options=opts,
#                 cross_encoder=cross_encoder
#             )
#             best_filters.append({
#                 "filter_name": fname,
#                 "option": best_opt["option"]
#             })

#         results.append({
#             "dataset": dataset["name"],
#             "indicator": ind["name"],
#             "confidence": conf,
#             "filters": best_filters
#         })

#     return jsonify({"results": results})


# if __name__ == "__main__":
#     app.run(debug=True, host="0.0.0.0", port=5009)






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
        temperature=0
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

ALLOWED OPERATIONS:
- spelling correction
- grammar correction
- casing normalization
- synonym normalization
- semantic mapping ONLY if the word exists explicitly in text

CRITICAL RULE (VERY IMPORTANT):
- If the user query is ONLY a dataset or product name
  (examples: IIP, CPI, CPIALRL, HCES, ASI, NSS, PLFS,CPI2),
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
    targets = [
        f"{y}{y+1}",
        f"{y-1}{y}",
        str(y)
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

# ================================
# SMART FILTER ENGINE
# ================================
def select_best_filter_option(query, filter_name, options, cross_encoder):
    q_lower = query.lower()
    fname_lower = filter_name.lower()

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

        for word in q_lower.split():
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

names = [clean_text(i["name"]) for i in INDICATORS]
descs = [clean_text(i.get("desc", "")) for i in INDICATORS]

embeddings = (0.3 * bi_encoder.encode(names, convert_to_numpy=True) + 0.7 * bi_encoder.encode(descs, convert_to_numpy=True))
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
        hits = qclient.search(collection_name=COLLECTION,query_vector=q_vec[0].tolist(),limit=top_k)
        candidates = [h.payload for h in hits]
    else:
        _, I = faiss_index.search(q_vec.astype("float32"), top_k)
        candidates = [INDICATORS[i] for i in I[0] if i >= 0]

    scores = cross_encoder.predict([(query, c["name"] + " " + c.get("desc", "")) for c in candidates])
    for i, c in enumerate(candidates):
        c["score"] = float(scores[i])

    candidates.sort(key=lambda x: x["score"], reverse=True)

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
        dataset = next(d for d in DATASETS if d["code"] == ind["parent"])
        related_filters = [f for f in FILTERS if f["parent"] == ind["code"]]

        grouped = {}
        for f in related_filters:
            grouped.setdefault(f["filter_name"], []).append(f)

        best_filters = []
        for fname, opts in grouped.items():
            best_opt = select_best_filter_option(
                query=q,
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
