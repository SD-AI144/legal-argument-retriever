# Legal Argument Retriever ⚖️
> **Zero-Hallucination, "Retrieval-Only" Precedent Search for Indian Civil Litigation**

[![Streamlit App](https://img.shields.io/badge/Streamlit-Live%20Demo-FF4B4B?style=for-the-badge&logo=streamlit)](https://legal-argument-retriever-9r98krof2zjwb2jnvzbw9a.streamlit.app/)
[![SSRN Preprint](https://img.shields.io/badge/SSRN-Research%20Paper-003366?style=for-the-badge&logo=ssrn)](https://ssrn.com/abstract=6530680)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)](https://python.org)

**Legal Argument Retriever** is a proof-of-concept legal intelligence system engineered specifically to eliminate AI hallucinations in Indian court filings. Built on a pilot dataset mapping **Section 115 of the Code of Civil Procedure, 1908 (CPC)**, the system enforces a strict **"retrieval-only" architecture**: it does not generate, summarize, or paraphrase text. Instead, it surfaces verbatim, human-verified judicial arguments, holdings, and rebuttals directly linked to source judgment paragraphs.

---

## 🎯 The Problem

Existing Indian legal AI tools rely heavily on Generative AI / Large Language Model (LLM) summaries ("GPT wrappers"). In fast-paced litigation environments, this reliance creates catastrophic liabilities:

1. **The Hallucination Crisis & Judicial Misconduct:** As demonstrated in *Gummadi Usha Rani v. Sure Mallikarjuna Rao* (2026 SCC OnLine SC 341), submitting AI-generated non-existent precedents is classified by the Supreme Court of India as **legal misconduct**, carrying severe professional and judicial consequences.
2. **Summarisation as Erasure:** Generative summaries flatten nuanced legal definitions (e.g., the shifting judicial interpretation of *"consultation"* under Article 124(2)). In common law, the authority of a precedent resides strictly in the **verbatim syllable** of the court's reasoning, not in an LLM's simplified paraphrase.
3. **Logic Blindness & Context Collapse:** Keyword and standard vector models routinely confuse petitioner claims with respondent rebuttals, or present rejected arguments as winning holdings simply because vocabulary matches the query.
4. **The Structural Access Barrier:** With over 5.34 crore pending cases in Indian courts, junior and first-generation advocates lack access to the instant precedent recall enjoyed by senior advocates.

---

## 💡 The Approach: "Memory as a Service"

To build a zero-hallucination platform, this project adopts three foundational design mandates:
```
                      [ User Natural Language Query ]
                                     │
                                     ▼
                      [ InLegalBERT Embedding Engine ]
                                     │
                                     ▼
           ┌──────────────────────────────────────────────────┐
           │            Retrieval-Only Data Schema            │
           ├──────────────────────────────────────────────────┤
           │ 📍 Verbatim Anchor    ➜ Paragraph-exact text     │
           │ 🎯 Strategy Signal    ➜ Accepted/Rejected/Partial │
           │ ⚔️ Adversarial Mapping ➜ Claim vs. Rebuttal Pair  │
           └──────────────────────────────────────────────────┘
```

* **Architectural Immunity to Hallucination:** The system contains zero text-generation capabilities. It cannot invent citations or text because it can only return stored, exact extractions.
* **Verbatim Anchors:** Every retrieved argument is pinned to an exact paragraph number and judgment citation.
* **Strategy Signals:** Each entry indexes whether the court **Accepted**, **Rejected**, or **Partially Accepted** a given argument on Section 115 CPC revisions.
* **Adversarial Attribution:** Maps petitioner arguments directly against respondent counter-arguments, treating legal research as an active courtroom dialogue rather than a static document search.
* **Domain Embeddings (`InLegalBERT`):** Uses `InLegalBERT` (trained on 5.4 million Indian legal documents by IIT Kharagpur) to bridge natural language queries (e.g., *"tenant refused rent"*) with formal statutory formulations under Section 115 CPC.
* **100% Human-in-the-Loop Verification:** All 99 pilot dataset entries were manually verified against primary source court judgments for lexical fidelity.

---

## ⚠️ What It Gets Wrong (Current Limitations)

* **Domain Scope:** The current dataset (`Sectioncpc115.xlsx`) is strictly limited to 99 verified entries under Section 115 CPC. It does not yet cover broader CPC sections or criminal law statutes.
* **Static Storage Backend:** The proof-of-concept runs on a local Excel/pandas pipeline inside `app.py` rather than a scalable distributed vector database.
* **Archive OCR Vulnerability:** The pipeline relies on structured text ingestion and remains sensitive to dirty OCR errors or mixed-language court scans (English with Odia, Telugu, or Kannada) frequently found on High Court portals.

---

## 🛠️ What To Fix Next (Roadmap)

- [ ] **Vector DB Migration:** Transition from the static backend to **ChromaDB / Qdrant** for sub-second similarity search across tens of thousands of judgments.
- [ ] **Hybrid Retrieval Pipeline:** Combine `InLegalBERT` dense embeddings with **BM25 sparse keyword matching** to handle precise statutory citations alongside semantic queries.
- [ ] **Automated PDF Parsing & OCR Correction:** Implement specialized layout-aware PDF parsers to clean degraded scanned court PDFs directly from High Court and Supreme Court repositories.
- [ ] **Expansion Beyond CPC 115:** Scale the curated argument mapping to include key civil and criminal statutory provisions (e.g., Section 397/401 CrPC / BNSS, Article 226/227 writs).

---
@article{legal_argument_retriever_2026,
  title={Beyond the Hallucination: Building a 'Retrieval-Only' Future for Indian Legal Tech},
  author={SD-AI144},
  journal={SSRN Electronic Journal},
  year={2026},
  url={[https://ssrn.com/abstract=6530680](https://ssrn.com/abstract=6530680)}
}


1. Clone the repository:
   ```bash
   git clone [https://github.com/SD-AI144/legal-argument-retriever.git](https://github.com/SD-AI144/legal-argument-retriever.git)
   cd legal-argument-retriever
