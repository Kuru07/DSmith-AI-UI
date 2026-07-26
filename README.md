# DSmith AI — Streamlit Frontend

> **Autonomous Data Science Agent** — Upload a CSV, pick a target column,
> and let DSmith AI clean the data, select models, train them, and recommend
> the best one.

---

## Table of Contents

1. [Overview](#overview)
2. [How It Works](#how-it-works)
3. [Project Structure](#project-structure)
4. [Getting Started](#getting-started)
   - [Prerequisites](#prerequisites)
   - [Installation](#installation)
   - [Environment Variables](#environment-variables)
   - [Running the App](#running-the-app)
5. [Application Walkthrough](#application-walkthrough)
6. [API Reference](#api-reference)
7. [Error Handling](#error-handling)
8. [Dependencies](#dependencies)

---

## Overview

DSmith AI UI is the Streamlit-based frontend for the **DSmith AI** autonomous
data science platform. It provides a clean, step-by-step interface that lets
users:

- Upload a raw CSV dataset (up to 20 MB).
- Select the column they want a model to predict.
- Trigger a fully automated pipeline that cleans the data, identifies the ML
  problem type, selects candidate models, trains them, and compares their
  performance.
- Download the cleaned dataset produced by the agent.

The frontend communicates exclusively with a **FastAPI backend** that hosts the
agentic pipeline. The UI never runs any ML code itself.

---

## How It Works

```
User Browser
    │
    │  1. Upload CSV + select target
    ▼
Streamlit App  (this repository)
    │
    │  2. POST /analyze  (multipart/form-data)
    ▼
DSmith AI Backend  (FastAPI)
    │
    ├─ Data Agent ──► Cleans dataset ──► cleaned.csv
    └─ ML Agent   ──► Trains models  ──► best_model.joblib
    │
    │  3. JSON result (metrics, best model, download URLs)
    ▼
Streamlit App
    │
    │  4. Renders results, metrics table, download button
    ▼
User Browser
```

The backend performs two sequential agent stages:

| Stage | Responsibility |
|---|---|
| **Data Cleaning Agent** | Fixes data quality issues, imputes missing values, removes duplicates. Produces `cleaned.csv`. |
| **ML Agent** | Reads `cleaned.csv`, determines problem type (classification / regression), selects and trains multiple models, evaluates performance, picks the best. |

---

## Project Structure

```
DSmith AI UI/
├── app.py              # Streamlit application (single-file)
├── requirements.txt    # Pinned Python dependencies
├── .env                # Local environment variables (not committed)
├── .gitignore
└── README.md
```

---

## Getting Started

### Prerequisites

- **Python 3.11+**
- The **DSmith AI backend** running and accessible (locally or deployed).

### Installation

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd "DSmith AI UI"

# 2. Create and activate a virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the project root:

```env
# Full base URL of the DSmith AI FastAPI backend.
# No trailing slash.
API_LINK=https://your-backend.onrender.com
```

| Variable | Required | Description |
|---|---|---|
| `API_LINK` | ✅ Yes | Base URL of the FastAPI backend. The app will display an error and stop if this is missing. |

> **Tip:** When running the backend locally, set `API_LINK=http://localhost:8000`.

### Running the App

```bash
streamlit run app.py
```

The app opens in your browser at `http://localhost:8501` by default.

---

## Application Walkthrough

The UI is divided into four numbered steps:

### Step 1 — Upload Dataset

- Accepts `.csv` files up to **20 MB**.
- Validates that the file is readable, non-empty, and contains at least two
  columns.
- Displays a preview (first 20 rows) and summary metrics (row count, column
  count, total missing values).

### Step 2 — Select Target Column

- Dropdown populated from the uploaded dataset's column headers.
- The **Analyse Dataset** button stays disabled until a target is chosen.

### Step 3 — Autonomous Analysis

- Clicking **🚀 Analyse Dataset** posts the CSV and target column to the
  backend `/analyze` endpoint.
- A spinner is shown while the backend pipeline runs (timeout: 5 minutes).
- On success, the result is stored in `st.session_state` so it persists across
  Streamlit re-runs.

### Step 4 — Analysis Results

Results are rendered in the following sections:

| Section | Content |
|---|---|
| **Top metrics** | Problem type, target column, best model |
| **🧹 Data Cleaning** | Summary of cleaning actions and the step-by-step cleaning plan |
| **🧠 ML Problem Analysis** | Agent's reasoning for choosing classification vs. regression |
| **🤖 Models Evaluated** | Cards for each model (best model highlighted in green) |
| **📈 Model Performance Comparison** | Sortable table of all model metrics; best model shown first |
| **⬇️ Download Cleaned Dataset** | One-click download of the agent-produced `cleaned.csv` |

---

## API Reference

The frontend calls a single backend endpoint:

### `POST /analyze`

Triggers the full autonomous pipeline.

**Request** — `multipart/form-data`

| Field | Type | Description |
|---|---|---|
| `file` | CSV file | The raw dataset to analyse |
| `target_column` | string | Name of the column to predict |

**Response** — `application/json`

```json
{
  "success": true,
  "problem_type": "classification",
  "target_column": "species",
  "best_model": "RandomForestClassifier",
  "selected_models": ["LogisticRegression", "RandomForestClassifier"],
  "problem_reasoning": "...",
  "cleaning": {
    "summary": "...",
    "plan": ["...", "..."],
    "retries": 0
  },
  "training": {
    "retries": 0
  },
  "metrics": {
    "models": {
      "RandomForestClassifier": {
        "accuracy": 0.9667,
        "f1_score": 0.9667
      }
    }
  },
  "downloads": {
    "cleaned_dataset": "/downloads/cleaned/<job-id>"
  }
}
```

### `GET /downloads/cleaned/<job-id>`

Returns the cleaned CSV file produced by the data agent.

---

## Error Handling

The frontend handles all common failure modes gracefully:

| HTTP Status | Cause | UI Response |
|---|---|---|
| `400` | Invalid dataset or unknown target column | Error with available columns listed |
| `413` | File exceeds 20 MB backend limit | Clear size-limit message |
| `422` | Missing required fields | Validation message |
| `500` | Agent failure during cleaning or training | Error with retry suggestion |
| `502 / 503 / 504` | Backend temporarily unavailable | Warning with wait suggestion |
| Timeout | Request exceeded 5-minute limit | Timeout message with retry advice |
| Connection error | Backend unreachable | Connection error with startup note |

A frontend-side 20 MB size check runs before any network request to give
instant feedback without wasting bandwidth.

---

## Dependencies

Core runtime dependencies:

| Package | Purpose |
|---|---|
| `streamlit` | Web UI framework |
| `pandas` | CSV parsing and DataFrame operations |
| `requests` | HTTP calls to the FastAPI backend |
| `python-dotenv` | Loading `API_LINK` from `.env` |

See [`requirements.txt`](./requirements.txt) for the full pinned dependency list.

---

## Related

- **DSmith AI Backend** — The FastAPI service hosting the agentic pipeline
  (data cleaning agent + ML agent). Located in the `DSmith AI` directory.
