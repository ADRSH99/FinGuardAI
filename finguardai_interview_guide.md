# FinGuardAI — Technical Interview Preparation Guide

> **Role:** Senior Software Engineer & Technical Interviewer  
> **Goal:** Teach you this project so you can explain it confidently in any interview.

---

## 1. PROJECT OVERVIEW

### What problem does it solve?

Financial fraud is a real-time problem. By the time a bank analyst manually reviews a suspicious transaction, the money is already gone. FinGuardAI solves this by **automatically scoring every transaction the instant it is submitted**, using a consensus of four machine learning models, and giving a plain-English explanation of *why* it was flagged — not just a score.

### Who is it for?

| User | What they do |
|---|---|
| **Customer / End-user** | Submits a transaction via the User Portal. Gets an instant fraud verdict + explanation. |
| **Bank Admin / Analyst** | Logs into the Admin Dashboard. Monitors flagged transactions, global feature importance trends, and per-model accuracy benchmarks. |

### High-Level Workflow (3 sentences)

1. A user fills in a transaction form (amount, sender/receiver balances, type).
2. The Flask backend feeds those numbers through 4 pre-trained ML models, averages their fraud probabilities, runs SHAP to explain which features drove the decision, and optionally calls Google Gemini for a natural-language summary.
3. The result — verdict, probability, SHAP breakdown, and LLM explanation — is returned to the browser and stored in memory for the admin dashboard.

---

### 1-Minute Interview Answer

> "FinGuardAI is a real-time financial fraud detection web app. A user submits a transaction and our backend immediately runs it through an **ensemble of four deep learning models** — a TCN, a TCN+BiLSTM with Attention, a CNN+BiLSTM, and an LSTM+Random Forest hybrid. We average their fraud probability scores, then use **SHAP** to generate feature-level explanations so the user knows *why* the system flagged the transaction. An optional Google Gemini call wraps all of that into a readable, natural-language summary. The admin side shows aggregated SHAP importance trends and a per-transaction history across the current session."

---

### 3-Minute Interview Answer

> "The core problem is that fraud detection models are black boxes — they spit out a 0 or 1 and nobody knows why. FinGuardAI is my attempt to combine **accuracy** with **explainability**.
>
> On the ML side, I trained four architecturally distinct models on the PaySim financial transactions dataset. Using multiple architectures is deliberate — a TCN is great at capturing temporal patterns, an LSTM handles sequential dependencies, a CNN extracts local feature patterns, and a Random Forest acts as a robust classical baseline. By **averaging their probabilities** rather than trusting a single model, I reduce variance and improve generalization — this is the core idea behind ensemble learning.
>
> On the explainability side, I use **SHAP KernelExplainer** to compute how much each of the five input features — amount, old/new sender balance, old/new receiver balance — contributed to the final prediction. This is surfaced both to the end user (per-transaction SHAP bars) and the admin (aggregated global feature importance chart).
>
> The backend is a lightweight **Flask** app. Flask was the right choice here because the core logic lives in the ML layer, not the web layer — I didn't need Django's ORM or full MVC overhead. The frontend is plain HTML/CSS/JS with Chart.js for the admin charts, keeping it simple and fast to load.
>
> The biggest design constraint I worked within is that the models are heterogeneous — some are Keras `.keras` files, some are PyTorch `.pt` archives, and one is a hybrid of a Keras LSTM feature extractor + a scikit-learn Random Forest. The `ml_stub.py` loader abstracts all of that behind a single `get_fraud_prediction()` function."

---

## 2. ARCHITECTURE

### ASCII System Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                        BROWSER (Client)                      │
│                                                              │
│  ┌──────────────────┐        ┌──────────────────────────┐   │
│  │  User Portal     │        │  Admin Dashboard          │   │
│  │  (index.html)    │        │  (admin.html)             │   │
│  │                  │        │                           │   │
│  │  - Form submit   │        │  - KPI Cards              │   │
│  │  - SHAP bars     │        │  - Chart.js graphs        │   │
│  │  - LLM text      │        │  - Transaction table      │   │
│  └────────┬─────────┘        └────────────┬──────────────┘   │
│           │  POST /api/transaction         │  GET /admin      │
└───────────┼──────────────────────────────-┼─────────────────┘
            │                               │
            ▼                               ▼
┌───────────────────────────────────────────────────────────┐
│                      app.py  (Flask)                       │
│                                                            │
│   /              → index.html (user portal)                │
│   /login, /logout → session-based admin auth               │
│   /admin         → aggregates transactions[], renders HTML  │
│   /api/transaction → calls ml_stub + llm_stub              │
│   /api/admin/feature_importance → JSON for AJAX            │
└──────────┬────────────────────────┬───────────────────────┘
           │                        │
           ▼                        ▼
┌──────────────────┐     ┌──────────────────────┐
│  utils/ml_stub.py│     │  utils/llm_stub.py   │
│                  │     │                       │
│  On import:      │     │  generate_context_    │
│  - Loads 4 models│     │  explanation()        │
│  At request:     │     │                       │
│  - predict_      │     │  1. Try Gemini API    │
│    ensemble()    │     │  2. Fallback: local   │
│  - SHAP values   │     │     stub text         │
└──────────┬───────┘     └──────────────────────┘
           │
     ┌─────┼──────────────────────────────┐
     ▼     ▼           ▼                  ▼
  [TCN]  [TCN+BiLSTM  [CNN+BiLSTM]   [LSTM+RF
  .keras  +Attention]  .keras]         Hybrid]
          .pt                          .keras + .pkl
```

### Request / Data Flow (step by step)

```
1. User fills form → script.js auto-calculates new balances
2. script.js → POST /api/transaction  { JSON payload }
3. app.py → get_fraud_prediction(data)
4. ml_stub → builds x5 = [amount, obOrg, nbOrg, obDest, nbDest]  (shape 1×5)
5. For each model:
      a. Optional feature engineering (CNN+ → 17 features, reshape)
      b. Optional log1p scaling (PyTorch models)
      c. Forward pass → scalar probability
6. Ensemble average → avg_prob, breakdown list
7. SHAP KernelExplainer on ensemble wrapper → 5 SHAP values
8. SHAP KernelExplainer per individual model → per-model SHAP
9. Return dict: {is_fraud, probability, model_breakdown, shap_values}
10. app.py → generate_context_explanation(data, prediction)
11. llm_stub → Gemini call (or local stub) → explanation string
12. Build response_data → append to transactions[] list → return JSON
13. script.js receives JSON → updateUIWithResults() → animate UI
```

---

## 3. TECH STACK — DEEP DIVE

### Backend

| Tech | What it does | Why chosen | Alternatives & Tradeoffs |
|---|---|---|---|
| **Flask** | HTTP routing, template rendering, session management | Minimal footprint; the ML layer *is* the application — no ORM or migrations needed | Django (too heavy), FastAPI (async would be overkill for sync ML inference) |
| **Python** | Language for everything backend | Best ML library ecosystem; unanimous choice for data science stacks | — |
| **`os.urandom(24)`** | Generates random secret key for Flask sessions | Each restart generates a new key, suitable for demos; sessions can't be forged | In production: load from env variable |

### Machine Learning

| Tech | What it does | Why chosen | Tradeoffs |
|---|---|---|---|
| **TensorFlow / Keras** | Loads and runs `.keras` model files (TCN, CNN+BiLSTM, LSTM extractor) | Standard for saving/loading Keras sequential and functional models | Heavy dependency; TF 2.x oneDNN warnings on CPU |
| **PyTorch** | Loads `.pt` model files (TCN+BiLSTM+Attention) | Models trained in PyTorch research style; `torch.jit.load` for TorchScript models | Two separate deep learning frameworks in one project is unusual but necessary given model origins |
| **scikit-learn / joblib** | Loads `best_rf_model.pkl` RF classifier; `RobustScaler` for CNN+ | Industry standard for classical ML; `joblib` handles large numpy arrays efficiently | Pickle-based serialization has version sensitivity |
| **SHAP** | KernelExplainer computes feature importance for any black-box function | Model-agnostic; works with the ensemble wrapper function directly | Slow (100 sample perturbations per call); TreeExplainer would be faster for RF-only |
| **NumPy** | All feature arrays, reshaping, background arrays for SHAP | Universal tensor manipulation in Python ML | — |

### Frontend

| Tech | What it does | Why chosen | Tradeoffs |
|---|---|---|---|
| **HTML/CSS/JS** (Vanilla) | UI structure, dark glassmorphism styling, interactivity | Minimal overhead; no build step needed | Not reusable components like React; fine for a focused demo |
| **Chart.js** (CDN) | Bar chart (risk timeline) + Doughnut chart (SHAP aggregate) | Zero configuration needed; renders on `<canvas>` | D3.js would offer more customization |
| **Jinja2** | Server-side templating (`{% extends %}`, `{{ }}`) | Built into Flask; perfect for passing Python data to HTML | React/Vue would enable real-time admin without page reload |
| **Google Fonts (Outfit)** | Typography | Modern, readable sans-serif typeface | — |

### LLM / Utilities

| Tech | What it does | Why chosen | Tradeoffs |
|---|---|---|---|
| **google-generativeai** | Calls Gemini 1.5 Flash to generate natural-language fraud explanations | Free tier; fast for short prompts; contextual language | Requires API key; adds network latency; graceful fallback implemented |
| **python-dotenv** | Loads `GEMINI_API_KEY` from `.env` file | Keeps secrets out of source code | In production use a secrets manager (AWS Secrets Manager, Vault) |

---

## 4. FOLDER & FILE STRUCTURE

```
FinGuardAI/
├── app.py                  ← Flask app: routes, session, orchestration
├── requirements.txt        ← Pip dependencies
├── README.md
│
├── utils/
│   ├── ml_stub.py          ← Model loading, ensemble inference, SHAP
│   └── llm_stub.py         ← Gemini call + local fallback explanation
│
├── models/
│   ├── tcn_final.keras                   ← Keras TCN (5 features)
│   ├── finguard_paysim_final.pt          ← PyTorch TCN+BiLSTM+Attention
│   ├── best_rf_lstm.pt / .zip            ← PyTorch LSTM+RF (zip archive)
│   ├── best_rf_model.pkl                 ← Standalone sklearn RF (fallback)
│   ├── FinguardAI_cnn+/
│   │   ├── fraud_model.keras             ← CNN+BiLSTM Keras model
│   │   ├── scaler.pkl                    ← RobustScaler for 7 continuous features
│   │   └── attention_layer.py            ← Custom Keras AttentionLayer class
│   └── rf_lstm_2_shubh/
│       ├── lstm_extractor.keras          ← LSTM feature extractor
│       ├── rf_hybrid_classifier.pkl      ← RF trained on LSTM embeddings
│       └── pipeline_config.json          ← seq_len, optimal_threshold
│
├── templates/
│   ├── base.html           ← Master layout: nav, fonts, Chart.js CDN
│   ├── index.html          ← User portal (extends base)
│   ├── admin.html          ← Admin dashboard (extends base)
│   └── login.html          ← Admin login form
│
├── static/
│   ├── style.css           ← Dark glassmorphism design system
│   ├── script.js           ← Transaction form logic, fetch, UI animation
│   └── logo.png            ← App logo
│
└── test scripts (not in production path):
    ├── test_loader_only.py
    ├── test_4_models.py
    ├── verify_app.py
    └── debug_loader.py
```

### File-by-File Responsibilities

#### `app.py`
- **Purpose:** The application entry point and HTTP orchestrator.
- **Responsibilities:** Defines 5 routes, manages `transactions[]` in-memory list, aggregates SHAP values for admin charts, enforces session-based auth for `/admin`.
- **Interactions:** Imports `get_fraud_prediction` from `ml_stub` and `generate_context_explanation` from `llm_stub`. Renders Jinja2 templates. Passes computed `shap_labels`, `shap_data`, `timeline_data` to `admin.html`.
- **Why it exists:** Clean separation — the route layer knows nothing about ML internals.

#### `utils/ml_stub.py`
- **Purpose:** Self-contained ML engine. Loads all models once at import time; exposes a single function `get_fraud_prediction()` for the app layer.
- **Responsibilities:** Model loading (5 loaders with fallback logic), heterogeneous inference (`_predict_one` handles sklearn/torch/tf/tf_cnn_sequence/hybrid_lstm_rf types), ensemble averaging (`predict_ensemble`), SHAP wrapper creation and execution, feature normalization.
- **Key design:** `ENSEMBLE_MODELS = _load_all_models()` runs **at module import** — models load once when Flask starts, not on every request. This is critical for performance.
- **Why it exists:** Isolates all ML complexity from the Flask routing layer. App.py doesn't need to know that 3 different frameworks are in play.

#### `utils/llm_stub.py`
- **Purpose:** Natural-language explanation layer with a graceful fallback.
- **Responsibilities:** Reads `GEMINI_API_KEY` from env, constructs a detailed prompt with full transaction + prediction context, calls Gemini 1.5 Flash, falls back to a rule-based template if the key is missing or the call fails.
- **Why it exists:** Explainability for non-technical users. A raw "87% fraud probability" means little; "High risk: the sender's balance was fully wiped in a single TRANSFER" is actionable.

#### `templates/base.html`
- **Purpose:** DRY master layout — navbar, global fonts, Chart.js CDN inclusion, footer.
- **Why Jinja2 inheritance:** `index.html` and `admin.html` both `{% extends 'base.html' %}` and fill `{% block content %}`. Avoids duplicating the nav/head on every page.

#### `static/script.js`
- **Purpose:** All client-side interactivity for the user portal.
- **Key behaviors:**
  1. **Auto-calculate** `newbalanceOrig = oldbalanceOrg - amount` and `newbalanceDest = oldbalanceDest + amount` as the user types.
  2. **Async fetch** `POST /api/transaction` with JSON payload.
  3. **Animate UI** — progress bar width, count-up animation for probability %, color-coded SHAP bars with staggered entry animations.
- **Why vanilla JS:** No build toolchain needed; keeps the project self-contained.

---

## 5. EXECUTION FLOW

### Application Startup

```
python app.py
  │
  ├─ Flask app created, secret_key set
  ├─ from utils.ml_stub import get_fraud_prediction
  │     │
  │     └─ _load_all_models() executes:
  │           ├─ _load_model_1_tcn()       → tf.keras.models.load_model('tcn_final.keras')
  │           ├─ _load_model_2_paysim()    → torch.jit.load('finguard_paysim_final.pt')
  │           ├─ _load_model_4_cnn()       → joblib.load(scaler) + tf.keras.load('fraud_model.keras')
  │           └─ _load_model_5_rf_lstm()   → tf.keras.load('lstm_extractor.keras') + joblib.load(rf.pkl)
  │
  ├─ from utils.llm_stub import generate_context_explanation
  │     └─ load_dotenv() → reads .env for GEMINI_API_KEY
  │
  └─ app.run(debug=True, port=5000) → Flask listens on http://localhost:5000
```

### Per-Request Flow (User submits transaction)

```
[Browser] User clicks "Send Payment"
    │
    ├─ script.js:updateBalances() → computes newbalanceOrig, newbalanceDest
    ├─ script.js → fetch POST /api/transaction  { JSON }
    │
[Flask] app.py:handle_transaction()
    ├─ data = request.json
    ├─ prediction = get_fraud_prediction(data)          ← ml_stub
    │     ├─ Build x5 numpy array (1×5)
    │     ├─ predict_ensemble(ENSEMBLE_MODELS, x5)
    │     │     └─ _predict_one(m, x5) × 4 models
    │     │           ├─ TCN:      log1p(x5) → tf.predict → sigmoid → float
    │     │           ├─ PyTorch:  log1p(x5) → torch.no_grad → sigmoid → float
    │     │           ├─ CNN+:     engineer_features_cnn → (1,24,17) → tf.predict → float
    │     │           └─ LSTM+RF:  (1,5,17) → lstm.predict → feat_padded → rf.predict_proba
    │     ├─ avg_prob = mean of 4 probabilities
    │     ├─ is_fraud = avg_prob >= 0.5
    │     ├─ _run_shap(ensemble_wrapper, background, x5)   ← 100 SHAP perturbations
    │     ├─ _run_shap(per_model_wrapper) × 4             ← per-model SHAP
    │     └─ return {is_fraud, probability, model_breakdown, shap_values}
    │
    ├─ explanation = generate_context_explanation(data, prediction)  ← llm_stub
    │     ├─ If GEMINI_API_KEY present: call Gemini 1.5 Flash API
    │     └─ Else: _generate_mock_explanation() → template string
    │
    ├─ Build response_data dict
    ├─ transactions.append(response_data)      ← admin dashboard storage
    └─ return jsonify(response_data)
    │
[Browser] script.js receives JSON
    └─ updateUIWithResults(prediction, explanation)
          ├─ Animate probability bar (width = prob%)
          ├─ Count-up animation for number display
          ├─ Color code (green/yellow/red by threshold)
          ├─ Populate model breakdown bars
          └─ Animate SHAP feature bars (staggered delay)
```

---

## 6. KEY DESIGN DECISIONS

### Decision 1: Load models once at import time

```python
# ml_stub.py — top level
ENSEMBLE_MODELS = _load_all_models()
```

**Why:** Loading 4 deep learning models takes 3–10 seconds. Doing this per-request would make the app unusable. By loading at module import (which happens once when Flask starts), every subsequent request hits already-loaded model objects in memory.

### Decision 2: Heterogeneous model types unified behind one interface

Each model dict has a `'type'` key (`'tf'`, `'torch'`, `'tf_cnn_sequence'`, `'hybrid_lstm_rf'`). `_predict_one()` switches on this type. The caller (`predict_ensemble`) never knows the difference — it just calls `_predict_one(m, x5)` and gets a float back.

**Pattern used:** Strategy Pattern — swappable inference strategies behind a common interface.

### Decision 3: Ensemble by simple average (not voting)

Rather than majority vote, probabilities are **averaged**. This means a model that is 95% sure (fraud) + a model that is 10% sure = 52.5% → flagged. Voting would require ≥3 models to agree. Simple averaging is more sensitive to strong signals from even one model.

### Decision 4: SHAP KernelExplainer (not TreeExplainer)

KernelExplainer is model-agnostic — it treats each model as a black box and perturbs inputs. This is the only SHAP explainer that works across all four model types (Keras, PyTorch, sklearn). TreeExplainer would be faster for the RF-only model but can't be applied to the ensemble wrapper.

### Decision 5: Graceful degradation everywhere

- If a model fails to load → skip it, log `[FAIL]`, continue.
- If SHAP fails → use random fallback values so the UI never breaks.
- If Gemini key missing → local stub explanation.
- If PyTorch model is a state dict (no `forward`) → deterministic fallback based on input hash.

### Decision 6: In-memory `transactions[]` list

Explicitly noted in README as a known limitation. Suitable for demos and hackathons — no database setup required. In production, this would be replaced with a PostgreSQL table or Redis stream.

### Decision 7: Feature Engineering for CNN+

The CNN+ model was trained on 17 features (including cyclical time encoding, one-hot transaction type, engineered balance ratios). Rather than changing the inference UI, `engineer_features_cnn()` re-derives those 17 features from the 5 core inputs using default assumptions (e.g., `PAYMENT` type, `step=1`). This keeps the user-facing form simple while honoring the model's expected input contract.

---

## 7. INTERVIEW PREPARATION

### Likely Questions & Best Answers

---

**Q: Walk me through this project.**

> Start with the problem, then the solution architecture, then mention the key technical challenge you solved.
> "This is a real-time fraud detection system. The interesting part is the ML layer — I have four architecturally different models, each trained in different frameworks, and I unified them behind a single inference interface. I also added SHAP explainability so the system isn't a black box, and a Gemini-powered explanation layer for non-technical users."

---

**Q: Why did you use an ensemble of models?**

> "Ensemble learning reduces variance. Any single model can have blind spots. A TCN might miss patterns the LSTM catches; the RF provides a classical regularizing vote. By averaging probabilities, the final score is more robust to individual model errors. It also improves recall on rare fraud events — if even one model is highly confident, the average shifts."

---

**Q: What is SHAP and why did you use it?**

> "SHAP stands for SHapley Additive exPlanations. It's a game-theory concept applied to ML — it asks: for this specific prediction, how much did each feature contribute? I used `KernelExplainer` specifically because it's model-agnostic — it works by perturbing the input and observing output changes. I couldn't use faster explainers like TreeExplainer because my ensemble includes neural networks. The payoff is that I can show the user exactly which account balance anomaly triggered the fraud flag."

---

**Q: How does your admin dashboard work?**

> "The admin dashboard is server-side rendered with Jinja2. When an admin hits `/admin`, Flask iterates the in-memory `transactions` list, aggregates the SHAP values across all transactions into a global importance dict, and passes that as `shap_labels` and `shap_data` to the template. Chart.js then renders it as a doughnut chart client-side. The risk timeline bar chart shows fraud probability for the last 20 transactions, color-coded red/yellow/blue by threshold."

---

**Q: What are the limitations of this project?**

> Be honest — interviewers respect self-awareness.
> "Three main ones: First, the `transactions[]` list is in-memory, so all admin analytics reset on restart — in production I'd use PostgreSQL or Redis. Second, admin auth is hardcoded as `admin/admin` — production would need JWT or OAuth. Third, SHAP KernelExplainer with 100 samples is slow (~1-2 seconds per request). For production I'd pre-compute background distributions and use faster explainers where the model type allows."

---

**Q: How would you scale this to production?**

> "Several changes: (1) Move model inference to a dedicated async worker (Celery + Redis) so Flask isn't blocked. (2) Persist transactions to PostgreSQL with proper indexes. (3) Cache model predictions for duplicate transactions using Redis. (4) Containerize with Docker; deploy the ML worker and web server as separate services behind an API gateway. (5) Replace KernelExplainer with a faster approximate method or pre-compute SHAP on background distributions. (6) Add proper auth (JWT), rate limiting, and input validation."

---

**Q: Why Flask and not FastAPI?**

> "Flask was the right tradeoff for this project. The inference is synchronous and blocking (model forward passes are not async-friendly natively), so FastAPI's async advantages don't apply directly. Flask's Jinja2 templating made the admin dashboard trivial to build — I'm passing data directly from Python to HTML without a separate API. If I were building a pure API backend with async ML workers, FastAPI would be the better choice."

---

**Q: How does the feature engineering work for the CNN+ model?**

> "The CNN+ model was trained on 17 features: the 5 core features scaled with a RobustScaler, plus 2 engineered features (balance wipe ratio = amount/old_sender_balance, and destination balance error = |old_dest + amount - new_dest|), plus one-hot encoded transaction type (5 columns), step, and cyclical time encodings (sin/cos of hour and day-of-week). At inference, since the user only provides 5 values, I re-engineer all 17 using default assumptions for the missing contextual fields. The model also expects a sequence of shape (1, 24, 17), so I repeat the single observation 24 times to fill the temporal dimension."

---

**Q: What's the difference between the `type: 'torch'` and `type: 'hybrid_lstm_rf'` handling in `_predict_one`?**

> "Pure PyTorch models (type `torch`) do a direct forward pass through the neural network and return a sigmoid probability. The hybrid (type `hybrid_lstm_rf`) is a two-stage pipeline: the Keras LSTM acts as a feature extractor — its final layer outputs a latent embedding vector rather than a classification. That embedding is then passed to a scikit-learn Random Forest that was trained specifically on those LSTM-generated embeddings. It's a form of transfer learning where the LSTM learns sequential patterns and the RF learns to classify based on them."

---

### Quick-Fire Follow-Ups

| Question | Key Point |
|---|---|
| What dataset was this trained on? | PaySim — a synthetic financial transaction dataset based on real mobile money data |
| What's your best model's F1? | TCN: 87.14% F1, 99.77% accuracy on test set |
| Why not use a single large model? | Ensemble reduces overfitting and variance; different architectures capture different signal types |
| How do you handle model loading failures? | Each loader has try/except; failed models are skipped; app works with however many loaded successfully |
| What's `pipeline_config.json` for? | Stores `sequence_length` and `optimal_threshold` for the LSTM+RF hybrid, set during training |
| Why `log1p` scaling for PyTorch? | Financial amounts span orders of magnitude (₹10 to ₹10M+); log1p compresses that range without a fitted scaler |
| How is the fraud threshold set? | Ensemble average ≥ 0.5 → fraud. The CNN+ notebook used 0.77 internally, but the ensemble uses 0.5 for uniformity |

---

*Guide compiled from full source code analysis — every file read, every function traced.*
