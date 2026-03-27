import streamlit as st
import pandas as pd
from streamlit_gsheets import GSheetsConnection
from datetime import datetime
import numpy as np
import re
import torch
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import faiss

# --- PAGE SETUP ---
st.markdown(
    "<h1 style='font-size: 32px; text-transform: uppercase;'>⚖️ beyond case-level retrieval: proposing an argument-level schema for verbatim argument retrieval from indian judicial decisions</h1>", 
    unsafe_allow_html=True
)

st.markdown(
    "<h3 style='font-size: 20px; text-transform: uppercase; color: gray;'>A Pilot Study Demonstrated on Section 115 of the Code of Civil Procedure, 1908</h3>", 
    unsafe_allow_html=True
)

st.info("""
**About this Prototype:** This site is a proof of concept supporting the research idea by **[Soubhagyashree Das](https://www.linkedin.com/in/soubhagyashree/?lipi=urn%3Ali%3Apage%3Ad_flagship3_profile_view_base_contact_details%3BXh%2Fws045RXynY8fZcxx2qA%3D%3D)**, a law student.

**What to Expect:** Instead of just returning full case files, this tool is designed to drill down into the specifics. By entering case facts or a legal issue, the system retrieves highly relevant **verbatim arguments** made in court, mapped directly to the court's reasoning and the underlying rule of law. It leverages a hybrid AI search approach to surface precise legal arguments concerning Section 115 of the CPC.
""")
st.divider()

# --- 1. CACHE DATA LOADING ---
@st.cache_data
def load_data():
    EXCEL_PATH = 'Sectioncpc115.xlsx'
    df = pd.read_excel(EXCEL_PATH)
    df.columns = df.columns.str.strip().str.replace('"', '').str.upper()
    df = df.dropna(subset=['ARGUMENT TEXT'], how='all').reset_index(drop=True)

    def normalise_outcome(val):
        v = str(val).strip().upper()
        if v in ['NAN', '-', '', 'NONE']: return 'Unknown'
        if 'PARTIAL' in v: return 'Partially Accepted'
        if 'ACCEPT' in v: return 'Accepted'
        if 'REJECT' in v: return 'Rejected'
        return str(val).strip()

    df['OUTCOME_CLEAN'] = df['ACCEPTED / REJECTED'].apply(normalise_outcome)

    def build_searchable_text(row):
        rl  = str(row.get('RULE OF LAW', '') or '')
        kw  = str(row.get('KEYWORD', '') or '')
        iss = str(row.get('ISSUE', '') or '')
        arg = str(row.get('ARGUMENT TEXT', '') or '')
        cr  = str(row.get('COURT REASONING', '') or '')
        fct = str(row.get('FACTS', '') or '')
        parts = [arg, cr, fct, iss, f"{rl} {rl} {rl}", f"{kw} {kw} {kw}"]
        combined = ' '.join(p for p in parts if p and p.lower() not in ['nan','none',''])
        return re.sub(r'\s+', ' ', combined).strip()

    texts = [build_searchable_text(row) for _, row in df.iterrows()]
    valid_mask = [len(t) > 20 for t in texts]
    df = df[valid_mask].reset_index(drop=True)
    texts = [t for t, v in zip(texts, valid_mask) if v]

    principle_texts = []
    for _, row in df.iterrows():
        rl  = str(row.get('RULE OF LAW', '') or '')
        kw  = str(row.get('KEYWORD', '') or '')
        iss = str(row.get('ISSUE', '') or '')
        t   = re.sub(r'\s+', ' ', f"{rl} {kw} {iss}").strip()
        principle_texts.append(t if len(t) > 5 else 'general revision jurisdiction Section 115 CPC')

    return df, texts, principle_texts

# --- 2. CACHE MODELS AND INDEXES ---
@st.cache_resource
def load_models_and_build_indexes(texts, principle_texts):
    # Stage 1: all-MiniLM
    model_stage1 = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model_stage1.encode(texts, batch_size=32, convert_to_numpy=True)
    faiss.normalize_L2(embeddings)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    # FIX 2: Added stop_words='english' to prevent garbage connector word matches
    tfidf_full = TfidfVectorizer(max_features=15000, ngram_range=(1, 3), min_df=1, sublinear_tf=True, stop_words='english')
    matrix_full = tfidf_full.fit_transform(texts)

    tfidf_principle = TfidfVectorizer(max_features=10000, ngram_range=(1, 3), min_df=1, sublinear_tf=True, stop_words='english')
    matrix_principle = tfidf_principle.fit_transform(principle_texts)

    # Stage 2: InLegalBERT
    INLEGALBERT_MODEL = "law-ai/InLegalBERT"
    inlegal_tokenizer = AutoTokenizer.from_pretrained(INLEGALBERT_MODEL)
    inlegal_model = AutoModel.from_pretrained(INLEGALBERT_MODEL)
    inlegal_model.eval()

    return model_stage1, index, tfidf_full, matrix_full, tfidf_principle, matrix_principle, inlegal_tokenizer, inlegal_model

with st.spinner("Loading AI Models and Legal Database... (This takes a minute on startup)"):
    df, texts, principle_texts = load_data()
    model_stage1, index, tfidf_full, matrix_full, tfidf_principle, matrix_principle, inlegal_tokenizer, inlegal_model = load_models_and_build_indexes(texts, principle_texts)

# --- 3. HELPER FUNCTIONS ---
LEGAL_SYNONYMS = {
    'written statement': 'written statement pleading amendment Order 6 Rule 17',
    'plaint': 'plaint suit proceeding filing',
    'injunction': 'injunction interim order stay interlocutory temporary',
    'stay': 'stay injunction interim order suspension',
    'appeal': 'appeal appellate decree final order first appeal',
    'revision': 'revision revisional jurisdiction Section 115 CPC supervisory High Court',
    'review': 'review revision recall order',
    'execution': 'execution decree executing court compliance',
    'jurisdiction': 'jurisdiction vested material irregularity clause (a) clause (b) clause (c) Section 115',
    'no jurisdiction': 'jurisdiction not vested clause (a) Section 115 CPC excess of jurisdiction',
    'wrong jurisdiction': 'jurisdiction not vested clause (a) Section 115 material irregularity',
    'exceeded jurisdiction': 'jurisdiction not vested clause (a) material irregularity Section 115',
    'interlocutory': 'interlocutory interim order not finally decided suit proceeding proviso amendment 1999',
    'interim order': 'interim order interlocutory not finally decided maintainability proviso',
    'final order': 'final order decree case decided suit disposed',
    'preliminary issue': 'preliminary issue case decided interlocutory order Section 115 clause (c)',
    'tenant': 'tenant landlord rent eviction lease Rent Act Rent Control',
    'landlord': 'landlord tenant rent eviction Rent Control Act',
    'plaintiff': 'plaintiff petitioner appellant party suit',
    'defendant': 'defendant respondent written statement',
    'petitioner': 'petitioner revision applicant High Court Section 115',
    'respondent': 'respondent opposite party revision',
    'high court': 'High Court supervisory jurisdiction Art 227 Article 227 revision Section 115',
    'article 226': 'Article 226 writ certiorari judicial order civil court',
    'article 227': 'Article 227 supervisory jurisdiction subordinate court interlocutory',
    'writ': 'writ certiorari Article 226 Article 227 supervisory jurisdiction',
    'certiorari': 'certiorari writ Article 226 judicial order civil court jurisdictional error',
    'supervisory': 'supervisory jurisdiction Article 227 High Court subordinate court',
    'mandamus': 'mandamus writ jurisdiction High Court Article 226',
    'evidence': 'evidence re-appreciation finding of fact perverse misreading',
    'finding of fact': 'finding of fact re-appreciation evidence perverse revisional jurisdiction',
    'perverse': 'perverse finding no evidence misreading re-appreciation',
    'concurrent finding': 'concurrent finding both courts revisional jurisdiction interference',
    're-appreciate': 're-appreciation evidence finding of fact revisional jurisdiction exceeded',
    'second appeal': 'second appeal appellate jurisdiction re-appreciation findings',
    'material irregularity': 'material irregularity clause (c) Section 115 CPC exercise jurisdiction',
    'error of law': 'error of law jurisdiction Section 115 material irregularity',
    'refused to decide': 'failed to exercise jurisdiction clause (b) Section 115',
    'declined to hear': 'failed to exercise jurisdiction clause (b) Section 115',
    'ignored': 'failed to exercise jurisdiction material irregularity clause (b) clause (c)',
    'dismissed': 'dismissed rejected order interlocutory revision',
    'arbitration': 'arbitration Section 37 Arbitration Act Article 227 judicial interference',
    'lok adalat': 'Lok Adalat award decree Legal Services Authorities Act execution',
    'rent control': 'Rent Control Act legality propriety revisional jurisdiction re-appreciation',
    'amendment': 'amendment 1999 proviso Section 115 interlocutory finally dispose suit',
    'subletting': 'subletting oral written consent Rent Restriction Act revisional jurisdiction',
    'registered document': 'registered document sham transaction inference revisional jurisdiction',
    'clause a': 'jurisdiction not vested clause (a) Section 115 CPC excess jurisdiction',
    'clause b': 'failed to exercise jurisdiction clause (b) Section 115 CPC',
    'clause c': 'material irregularity clause (c) Section 115 CPC exercise jurisdiction',
    'not vested': 'jurisdiction not vested clause (a) Section 115 CPC',
    'failed to exercise': 'failed to exercise jurisdiction clause (b) Section 115',
}

def expand_query(query):
    expanded = query
    query_lower = query.lower()
    matched = []
    for term, expansion in LEGAL_SYNONYMS.items():
        if term.lower() in query_lower:
            expanded += ' ' + expansion
            matched.append(term)
    return expanded, matched

def get_inlegalbert_embedding(text, max_length=512):
    inputs = inlegal_tokenizer(text, return_tensors='pt', truncation=True, max_length=max_length, padding=True)
    with torch.no_grad():
        outputs = inlegal_model(**inputs)
    attention_mask = inputs['attention_mask']
    token_embeddings = outputs.last_hidden_state
    mask_expanded = attention_mask.unsqueeze(-1).float()
    sum_embeddings = (token_embeddings * mask_expanded).sum(dim=1)
    sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
    return (sum_embeddings / sum_mask).squeeze().numpy()

def inlegalbert_similarity(query_text, candidate_text):
    q_emb = get_inlegalbert_embedding(query_text)
    c_emb = get_inlegalbert_embedding(candidate_text)
    dot = np.dot(q_emb, c_emb)
    norm = np.linalg.norm(q_emb) * np.linalg.norm(c_emb)
    return float(dot / norm) if norm > 0 else 0.0

# FIX 1: Absolute Normalization Functions
def norm_dict_abs(d, clip_max=0.85):
    """Normalise against a fixed ceiling so low absolute scores stay low."""
    return {k: min(v / clip_max, 1.0) for k, v in d.items()}

def norm_arr_abs(arr, clip_max=0.85):
    return np.clip(arr / clip_max, 0, 1.0)

def search_system(query, top_k=5, candidate_pool=20):
    expanded_query, matched_terms = expand_query(query)

    # --- STAGE 1: HYBRID SEARCH ---
    query_embedding = model_stage1.encode([expanded_query], convert_to_numpy=True)
    faiss.normalize_L2(query_embedding)
    k = min(candidate_pool * 4, len(texts))
    sem_scores_raw, sem_indices = index.search(query_embedding, k)
    
    FAISS_MIN_RAW = 0.20
    sem_scores = {
        int(idx): float(score) 
        for idx, score in zip(sem_indices[0], sem_scores_raw[0]) 
        if idx >= 0 and float(score) >= FAISS_MIN_RAW
    }

    qvec_full = tfidf_full.transform([expanded_query])
    scores_full_raw = cosine_similarity(qvec_full, matrix_full).flatten()

    qvec_principle = tfidf_principle.transform([expanded_query])
    scores_principle_raw = cosine_similarity(qvec_principle, matrix_principle).flatten()

    # ==========================================
    # UPGRADED OUT-OF-DOMAIN (OOD) VETO
    # ==========================================
    max_tfidf = float(scores_full_raw.max()) if len(scores_full_raw) > 0 else 0.0
    
    max_sem = max(sem_scores.values()) if sem_scores else 0.0
    
    if len(matched_terms) == 0 and max_tfidf < 0.05 and max_sem < 0.30:
        return [{"OOD_FLAG": True}]
    # ==========================================

    sem_norm = norm_dict_abs(sem_scores, clip_max=0.85)
    full_norm = norm_arr_abs(scores_full_raw, clip_max=0.85)
    principle_norm = norm_arr_abs(scores_principle_raw, clip_max=0.85)

    final_scores = {}
    for idx in range(len(texts)):
        s_norm = sem_norm.get(idx, 0)
        f_norm = float(full_norm[idx])
        p_norm = float(principle_norm[idx])
        
        if f_norm == 0.0 and p_norm == 0.0:
            s_norm = s_norm * 0.10 
            
        final_scores[idx] = (s_norm * 0.40) + (f_norm * 0.30) + (p_norm * 0.30)

    ranked = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:candidate_pool]
    MIN_RELEVANCE_THRESHOLD = 0.32 

    candidates = []
    for idx, score in ranked:
        if score < MIN_RELEVANCE_THRESHOLD: continue
        row = df.iloc[idx]
        candidates.append({
            'hybrid_score': round(score, 4), 'matched_synonyms': matched_terms,
            'case_name': str(row.get('CASE NAME', 'N/A')), 'citation': str(row.get('CITATION', 'N/A')),
            'forum': str(row.get('FORUM', 'N/A')), 'party': str(row.get('PARTY', 'N/A')),
            'issue': str(row.get('ISSUE', 'N/A')), 'argument': str(row.get('ARGUMENT TEXT', 'N/A')),
            'arg_para': str(row.get('ARG PARA NO.', 'N/A')), 'court_reasoning': str(row.get('COURT REASONING', 'N/A')),
            'court_para': str(row.get('COURT PARA NO.', 'N/A')), 'outcome': str(row.get('OUTCOME_CLEAN', 'N/A')),
            'rule_of_law': str(row.get('RULE OF LAW', 'N/A')), 'keyword': str(row.get('KEYWORD', 'N/A')),
            'link': str(row.get('LINK', 'N/A'))
        })

    # --- STAGE 2: INLEGALBERT RERANKING ---
    reranked = []
    for c in candidates:
        candidate_text = f"{c['argument']} {c['rule_of_law']}"
        raw_bert_sim = inlegalbert_similarity(query, candidate_text)
        
        rl_score_norm = max(0.0, (raw_bert_sim - 0.75) / 0.25)
        
        final = (0.70 * c['hybrid_score']) + (0.30 * rl_score_norm)
        reranked.append({**c, 'score': round(final, 4)})

    reranked.sort(key=lambda x: x['score'], reverse=True)
    return [r for r in reranked[:top_k] if r['score'] >= MIN_RELEVANCE_THRESHOLD]


# --- HELPER FUNCTION FOR DATABASE ---
def append_to_sheet(row_data):
    conn = st.connection("gsheets", type=GSheetsConnection)
    try:
        existing_data = conn.read(worksheet="Sheet1", ttl=0)
    except Exception as e:
        st.error(f"Could not read sheet. Please ensure a tab named 'Sheet1' exists.")
        return
    existing_data = existing_data.dropna(how="all")
    expected_cols = ["Timestamp", "Query", "Top Cases", "User Review"]
    
    if not all(col in existing_data.columns for col in expected_cols):
        existing_data = pd.DataFrame(columns=expected_cols)
        
    new_df = pd.DataFrame([row_data])
    updated_data = pd.concat([existing_data, new_df], ignore_index=True)
    updated_data = updated_data.fillna("")
    updated_data = updated_data.astype(str)
    updated_data.columns = updated_data.columns.astype(str)
    conn.update(worksheet="Sheet1", data=updated_data)

# FIX 4: Input Pre-flight Validator (Updated with Keyword Salad Detector)
def is_valid_legal_query(query: str) -> tuple[bool, str]:
    q = query.strip().lower()
    words = q.split()
    
    if len(words) < 3:
        return False, "Please describe the legal issue in at least 3–4 words."
        
    if sum(c.isalpha() for c in q) < 10:
        return False, "Please enter a meaningful legal query."
        
    connectors = {'the', 'a', 'an', 'to', 'in', 'of', 'and', 'was', 'is', 'for', 'on', 'by', 'with', 'that', 'from', 'under', 'after', 'before'}
    
    if len(words) >= 5 and not any(word in connectors for word in words):
        return False, "Your input looks like a list of random keywords. Please write a natural sentence describing the facts or legal issue (e.g., 'The trial court dismissed the appeal...')."
        
    return True, ""

# --- UI FRONTEND ---
st.markdown("### Enter Case Facts & Issue")
user_query = st.text_area("Type your query in plain language here...", height=150, placeholder="Example: The trial court allowed an amendment to the plaint after the trial had commenced...")

if 'search_results' not in st.session_state:
    st.session_state.search_results = None
if 'last_query' not in st.session_state:
    st.session_state.last_query = ""

# ==========================================
# SEARCH BUTTON LOGIC
# ==========================================
if st.button("Search Arguments", type="primary"):
    if user_query.strip():
        valid, reason = is_valid_legal_query(user_query)
        if not valid:
            st.warning(reason) 
        else:
            with st.spinner("Searching and Re-ranking..."):
                results = search_system(user_query, top_k=5)
                
                # Check for OOD Flag from the search_system
                if results and "OOD_FLAG" in results[0]:
                    st.warning("⚖️ **Out of Domain:** Your query is valid, but it appears to be about a completely different area of law (e.g., criminal, family, or corporate law). This tool is highly specialized only for civil revisions under **Section 115 of the Civil Procedure Code**.")
                    st.session_state.search_results = None 
                    st.session_state.last_query = user_query
                elif results:
                    st.session_state.search_results = results
                    st.session_state.last_query = user_query
                    try:
                        top_cases = " | ".join([r['case_name'] for r in results])
                        log_data = {
                            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "Query": user_query,
                            "Top Cases": top_cases,
                            "User Review": "Auto-saved search log" 
                        }
                        append_to_sheet(log_data)
                    except Exception as e:
                        st.warning(f"Background save failed (Search Log): {e}")
                else:
                    st.warning("No relevant arguments found. Try rephrasing.")
                    st.session_state.search_results = None
    else:
        st.warning("Please enter a query first.")

# ==========================================
# DISPLAY RESULTS
# ==========================================
if st.session_state.search_results:
    results = st.session_state.search_results
    st.success(f"Found {len(results)} highly relevant arguments.")
    
    if results[0]['matched_synonyms']:
        st.info(f"**Synonyms matched:** {', '.join(results[0]['matched_synonyms'])}")
    
    for i, r in enumerate(results):
        with st.expander(f"#{i+1} | {r['case_name']} ({r['outcome']}) - Score: {r['score']}", expanded=(i==0)):
            st.markdown(f"**Citation:** {r['citation']} | **Forum:** {r['forum']} | **Party:** {r['party']}")
            if r['issue'] != 'nan': st.markdown(f"**Issue:** {r['issue']}")
            st.markdown(f"**Argument (Para {r['arg_para']}):**\n> {r['argument']}")
            st.markdown(f"**Court Reasoning (Para {r['court_para']}):**\n> {r['court_reasoning']}")
            if r['rule_of_law'] != 'nan': st.markdown(f"**Principle:** {r['rule_of_law']}")
            if r['link'] != 'nan': st.markdown(f"[Read Full Source]({r['link']})")

    st.divider()
    
    # OPTIONAL FEEDBACK BOX
    st.markdown("#### 📝 Optional: Help improve this research!")
    with st.form("feedback_form", clear_on_submit=True):
        feedback_text = st.text_area("Did these results help? Were any arguments irrelevant?")
        submit_feedback = st.form_submit_button("Submit Feedback")
        
        if submit_feedback and feedback_text.strip():
            try:
                top_cases = " | ".join([r['case_name'] for r in st.session_state.search_results])
                feedback_data = {
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Query": st.session_state.last_query,
                    "Top Cases": top_cases,
                    "User Review": feedback_text 
                }
                append_to_sheet(feedback_data)
                st.success("Thank you! Your feedback has been recorded.")
            except Exception as e:
                st.error(f"Failed to save feedback: {e}")
