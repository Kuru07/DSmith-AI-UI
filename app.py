"""
DSmith AI — Streamlit Frontend
==============================

This module is the user-facing web interface for the DSmith AI autonomous
data science platform. It communicates with a FastAPI backend that runs an
agentic pipeline to clean, analyse, and train machine-learning models on
user-supplied CSV datasets.

Workflow
--------
1. User uploads a CSV file (max 20 MB).
2. User selects the target column to predict.
3. The app posts the file and target to the backend /analyze endpoint.
4. The backend runs the data-cleaning agent followed by the ML agent and
   returns a structured JSON result.
5. The app renders the result: problem type, cleaning summary, model
   comparison table, metric explanations, and a download button for the
   cleaned dataset.

Key Sections
------------
- CONFIGURATION       : Environment variables and Streamlit page setup.
- HELPER FUNCTIONS    : Metric formatting and DataFrame construction.
- HEADER              : Application title and description.
- CHECK API CONFIG    : Graceful stop when API_LINK is missing.
- FILE UPLOAD         : CSV upload widget and byte-level size guard.
- READ DATASET        : Parsing, validation, and preview of the CSV.
- TARGET SELECTION    : Column selector for the supervised-learning label.
- AUTONOMOUS ANALYSIS : Button that triggers the backend pipeline.
- HANDLE API ERRORS   : Per-status-code error messages (400/413/422/500/50x).
- DISPLAY RESULT      : Metrics table, model cards, reasoning, and downloads.

Environment Variables
---------------------
API_LINK : Full base URL of the DSmith AI FastAPI backend, e.g.
           https://your-app.onrender.com
           Loaded from a .env file in the project root via python-dotenv.

Dependencies
------------
See requirements.txt for the pinned dependency list.
Core: streamlit, pandas, requests, python-dotenv.
"""

import io
import os

import pandas as pd
import requests
# pyrefly: ignore [missing-import]
import streamlit as st
from dotenv import load_dotenv


load_dotenv()


# =========================================================
# CONFIGURATION
# =========================================================

API_URL = os.getenv("API_LINK")
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB

st.set_page_config(
    page_title="DSmith AI",
    page_icon="🧠",
    layout="wide",
)


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def format_metric_name(metric_name: str) -> str:
    """
    Convert metric keys into readable names.

    Example:
    f1_score -> F1 Score
    rmse -> RMSE
    r2 -> R²
    """
    names = {
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1 Score",
        "f1_score": "F1 Score",
        "mae": "MAE",
        "mse": "MSE",
        "rmse": "RMSE",
        "r2": "R²",
        "r2_score": "R²",
    }

    return names.get(
        metric_name.lower(),
        metric_name.replace("_", " ").title()
    )


def build_metrics_dataframe(metrics: dict) -> pd.DataFrame:
    """
    Convert metrics returned by the backend into a readable
    model-comparison DataFrame.

    Supports:

    {
        "models": {
            "LogisticRegression": {
                "accuracy": 0.8,
                "precision": 0.79
            },
            ...
        }
    }

    and:

    {
        "LogisticRegression": {
            "accuracy": 0.8
        },
        ...
    }
    """

    if not isinstance(metrics, dict):
        return pd.DataFrame()

    # Backend may wrap model metrics inside "models"
    model_metrics = metrics.get("models", metrics)

    if not isinstance(model_metrics, dict):
        return pd.DataFrame()

    rows = []

    for model_name, values in model_metrics.items():

        if not isinstance(values, dict):
            continue

        row = {
            "Model": model_name
        }

        for metric_name, metric_value in values.items():

            # Avoid nested values in the table
            if isinstance(
                metric_value,
                (dict, list, tuple)
            ):
                continue

            readable_name = format_metric_name(
                metric_name
            )

            # Round numeric metrics
            if isinstance(
                metric_value,
                (int, float)
            ):
                row[readable_name] = round(
                    metric_value,
                    4
                )
            else:
                row[readable_name] = metric_value

        rows.append(row)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


# =========================================================
# HEADER
# =========================================================

st.title("🧠 DSmith AI")

st.subheader("Autonomous Data Science Agent")

st.write(
    """
    Upload a CSV dataset and select the column you want to predict.
    DSmith AI will autonomously inspect and clean the dataset,
    determine the machine-learning problem, select suitable models,
    train them, compare their performance, and recommend the best model.
    """
)

st.divider()


# =========================================================
# CHECK API CONFIGURATION
# =========================================================

if not API_URL:

    st.error(
        "DSmith AI backend URL is not configured."
    )

    st.info(
        "Set API_LINK in your environment variables."
    )

    st.stop()


API_URL = API_URL.rstrip("/")


# =========================================================
# FILE UPLOAD
# =========================================================

st.header("1. Upload Dataset")

uploaded_file = st.file_uploader(
    "Upload a CSV file",
    type=["csv"],
    help="Upload the raw dataset you want DSmith AI to analyse.",
)


# =========================================================
# READ DATASET
# =========================================================

if uploaded_file is not None:

    try:

        file_bytes = uploaded_file.getvalue()

        # -------------------------------------------------
        # FRONTEND FILE SIZE VALIDATION
        # -------------------------------------------------

        if len(file_bytes) > MAX_FILE_SIZE:

            st.error(
                "❌ File is too large. "
                "Maximum allowed size is 20 MB."
            )

            st.stop()

        # -------------------------------------------------
        # READ CSV
        # -------------------------------------------------

        df = pd.read_csv(
            io.BytesIO(file_bytes)
        )

    except Exception as exc:

        st.error(
            f"Could not read the uploaded CSV: {exc}"
        )

        st.stop()


    # =====================================================
    # DATASET VALIDATION
    # =====================================================

    if df.empty:

        st.error(
            "The uploaded dataset is empty."
        )

        st.stop()


    if len(df.columns) < 2:

        st.error(
            "The dataset must contain at least two columns."
        )

        st.stop()


    # =====================================================
    # DATASET SUCCESS MESSAGE
    # =====================================================

    st.success(
        f"Dataset loaded successfully — "
        f"{df.shape[0]:,} rows × "
        f"{df.shape[1]} columns"
    )


    # =====================================================
    # DATASET PREVIEW
    # =====================================================

    with st.expander(
        "Preview Dataset",
        expanded=False
    ):

        st.dataframe(
            df.head(20),
            use_container_width=True
        )


    # =====================================================
    # BASIC DATASET INFORMATION
    # =====================================================

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Rows",
            f"{df.shape[0]:,}"
        )

    with col2:

        st.metric(
            "Columns",
            df.shape[1]
        )

    with col3:

        st.metric(
            "Missing Values",
            int(
                df.isna()
                .sum()
                .sum()
            )
        )


    st.divider()


    # =====================================================
    # TARGET SELECTION
    # =====================================================

    st.header("2. Select Target Column")

    st.write(
        "Choose the column that the machine-learning "
        "model should learn to predict."
    )

    target_column = st.selectbox(
        "Target Column",
        options=df.columns.tolist(),
        index=None,
        placeholder="Select the target column..."
    )


    st.divider()


    # =====================================================
    # AUTONOMOUS ANALYSIS
    # =====================================================

    st.header("3. Autonomous Analysis")

    analyze_button = st.button(
        "🚀 Analyze Dataset",
        type="primary",
        use_container_width=True,
        disabled=target_column is None
    )


    # =====================================================
    # ANALYZE BUTTON
    # =====================================================

    if analyze_button:

        files = {
            "file": (
                uploaded_file.name,
                file_bytes,
                "text/csv"
            )
        }

        data = {
            "target_column": target_column
        }

        try:

            with st.spinner(
                "DSmith AI is analysing your dataset. "
                "Cleaning, training and evaluation may take a moment..."
            ):

                response = requests.post(
                    f"{API_URL}/analyze",
                    files=files,
                    data=data,
                    timeout=300
                )


            # =================================================
            # HANDLE API ERRORS
            # =================================================

            if response.status_code != 200:

                try:

                    error_data = response.json()

                    detail = error_data.get(
                        "detail"
                    )

                except Exception:

                    detail = None


                # ---------------------------------------------
                # 400 - INVALID INPUT
                # ---------------------------------------------

                if response.status_code == 400:

                    if isinstance(
                        detail,
                        dict
                    ):

                        message = detail.get(
                            "message",
                            "Invalid dataset or request."
                        )

                        st.error(
                            f"❌ {message}"
                        )

                        # Target column error
                        if detail.get(
                            "target_column"
                        ):

                            st.write(
                                "Requested target: "
                                f"`{detail['target_column']}`"
                            )

                            available = detail.get(
                                "available_columns"
                            )

                            if available:

                                st.write(
                                    "**Available columns:**"
                                )

                                st.write(
                                    ", ".join(
                                        available
                                    )
                                )

                    else:

                        st.error(
                            f"❌ "
                            f"{detail or 'Invalid request.'}"
                        )


                # ---------------------------------------------
                # 413 - FILE TOO LARGE
                # ---------------------------------------------

                elif response.status_code == 413:

                    st.error(
                        "❌ The uploaded dataset is too large. "
                        "Maximum allowed size is 20 MB."
                    )


                # ---------------------------------------------
                # 422 - FASTAPI VALIDATION
                # ---------------------------------------------

                elif response.status_code == 422:

                    st.error(
                        "❌ Invalid request. "
                        "Please upload a CSV and "
                        "select a target column."
                    )


                # ---------------------------------------------
                # 500 - AGENT FAILURE
                # ---------------------------------------------

                elif response.status_code == 500:

                    if isinstance(
                        detail,
                        dict
                    ):

                        message = detail.get(
                            "message",
                            "DSmith AI could not "
                            "complete the analysis."
                        )

                    else:

                        message = (
                            detail
                            or
                            "DSmith AI could not "
                            "complete the analysis."
                        )

                    st.error(
                        f"❌ {message}"
                    )

                    st.info(
                        "The agent may have encountered "
                        "an issue while cleaning or "
                        "training the dataset."
                    )


                # ---------------------------------------------
                # SERVER UNAVAILABLE
                # ---------------------------------------------

                elif response.status_code in [
                    502,
                    503,
                    504
                ]:

                    st.warning(
                        "⚠️ DSmith AI is temporarily "
                        "unavailable. Please wait a "
                        "moment and try again."
                    )


                # ---------------------------------------------
                # UNKNOWN ERROR
                # ---------------------------------------------

                else:

                    st.error(
                        f"❌ Request failed "
                        f"(HTTP {response.status_code})."
                    )

                st.stop()


            # =================================================
            # API RESULT
            # =================================================

            result = response.json()

            if not result.get(
                "success"
            ):

                st.error(
                    "DSmith AI could not "
                    "complete the analysis."
                )

                st.stop()


            # =================================================
            # STORE RESULT
            # =================================================

            st.session_state[
                "analysis_result"
            ] = result

            st.success(
                "✅ Autonomous analysis "
                "completed successfully!"
            )


        # =====================================================
        # TIMEOUT
        # =====================================================

        except requests.exceptions.Timeout:

            st.error(
                "⏱️ The analysis took longer "
                "than expected."
            )

            st.info(
                "Large datasets or model training "
                "can take more time. Please try "
                "again with a smaller dataset."
            )

            st.stop()


        # =====================================================
        # CONNECTION ERROR
        # =====================================================

        except requests.exceptions.ConnectionError:

            st.error(
                "🔌 Could not connect to "
                "the DSmith AI backend."
            )

            st.info(
                "The backend may be starting up. "
                "Wait a moment and try again."
            )

            st.stop()


        # =====================================================
        # OTHER REQUEST ERROR
        # =====================================================

        except requests.exceptions.RequestException as exc:

            st.error(
                f"API request failed: {exc}"
            )

            st.stop()


# =========================================================
# DISPLAY ANALYSIS RESULT
# =========================================================

if "analysis_result" in st.session_state:

    result = st.session_state[
        "analysis_result"
    ]

    st.divider()

    st.header(
        "📊 Analysis Results"
    )


    # =====================================================
    # TOP RESULT CARDS
    # =====================================================

    col1, col2, col3 = st.columns(3)


    # -----------------------------------------------------
    # PROBLEM TYPE
    # -----------------------------------------------------

    with col1:

        problem_type = result.get(
            "problem_type",
            "Unknown"
        )

        if isinstance(
            problem_type,
            str
        ):

            problem_type = (
                problem_type.title()
            )

        st.metric(
            "Problem Type",
            problem_type
        )


    # -----------------------------------------------------
    # TARGET COLUMN
    # -----------------------------------------------------

    with col2:

        st.metric(
            "Target Column",
            result.get(
                "target_column",
                "Unknown"
            )
        )


    # -----------------------------------------------------
    # BEST MODEL
    # -----------------------------------------------------

    with col3:

        st.metric(
            "Best Model",
            result.get(
                "best_model",
                "Unknown"
            )
        )


    st.divider()


    # =====================================================
    # CLEANING INFORMATION
    # =====================================================

    st.subheader(
        "🧹 Data Cleaning"
    )

    cleaning = result.get(
        "cleaning",
        {}
    )

    cleaning_summary = (
        cleaning.get("summary")
        if isinstance(
            cleaning,
            dict
        )
        else None
    )

    if not cleaning_summary:

        cleaning_summary = result.get(
            "cleaning_summary"
        )


    cleaning_plan = (
        cleaning.get("plan")
        if isinstance(
            cleaning,
            dict
        )
        else None
    )

    if not cleaning_plan:

        cleaning_plan = result.get(
            "cleaning_plan"
        )


    cleaning_retries = (
        cleaning.get("retries")
        if isinstance(
            cleaning,
            dict
        )
        else None
    )


    # -----------------------------------------------------
    # CLEANING SUMMARY
    # -----------------------------------------------------

    if cleaning_summary:

        st.write(
            "**Cleaning Summary**"
        )

        st.write(
            cleaning_summary
        )


    # -----------------------------------------------------
    # CLEANING PLAN
    # -----------------------------------------------------

    if cleaning_plan:

        st.write(
            "**Cleaning Plan**"
        )

        if isinstance(
            cleaning_plan,
            list
        ):

            for step in cleaning_plan:

                st.write(
                    f"- {step}"
                )

        else:

            st.write(
                cleaning_plan
            )


    # -----------------------------------------------------
    # CLEANING RETRIES
    # -----------------------------------------------------

    if cleaning_retries is not None:

        st.caption(
            "Cleaning repair attempts: "
            f"{cleaning_retries}"
        )


    st.divider()


    # =====================================================
    # ML PROBLEM REASONING
    # =====================================================

    reasoning = result.get(
        "problem_reasoning"
    )

    if reasoning:

        st.subheader(
            "🧠 ML Problem Analysis"
        )

        st.write(
            reasoning
        )


    # =====================================================
    # MODELS SELECTED
    # =====================================================

    st.subheader(
        "🤖 Models Evaluated"
    )

    selected_models = result.get(
        "selected_models",
        []
    )

    if selected_models:

        model_columns = st.columns(
            min(
                len(selected_models),
                3
            )
        )

        for index, model in enumerate(
            selected_models
        ):

            column = model_columns[
                index
                % len(model_columns)
            ]

            with column:

                if (
                    model
                    == result.get("best_model")
                ):

                    st.success(
                        f"🏆 {model}"
                    )

                else:

                    st.info(
                        f"🤖 {model}"
                    )

    else:

        st.info(
            "No model-selection "
            "information was returned."
        )


    st.divider()


    # =====================================================
    # MODEL COMPARISON
    # =====================================================

    st.subheader(
        "📈 Model Performance Comparison"
    )

    metrics = result.get(
        "metrics",
        {}
    )

    metrics_df = (
        build_metrics_dataframe(
            metrics
        )
    )


    if not metrics_df.empty:

        # -------------------------------------------------
        # PUT BEST MODEL FIRST
        # -------------------------------------------------

        best_model = result.get(
            "best_model"
        )

        if best_model:

            metrics_df[
                "_best"
            ] = (
                metrics_df["Model"]
                == best_model
            )

            metrics_df = (
                metrics_df
                .sort_values(
                    "_best",
                    ascending=False
                )
                .drop(
                    columns=["_best"]
                )
                .reset_index(
                    drop=True
                )
            )


        # -------------------------------------------------
        # DISPLAY TABLE
        # -------------------------------------------------

        st.dataframe(
            metrics_df,
            use_container_width=True,
            hide_index=True
        )


        # -------------------------------------------------
        # BEST MODEL SUMMARY
        # -------------------------------------------------

        if best_model:

            st.success(
                f"🏆 Best Performing Model: "
                f"**{best_model}**"
            )


        # -------------------------------------------------
        # EXPLAIN METRICS
        # -------------------------------------------------

        with st.expander(
            "What do these metrics mean?"
        ):

            if (
                str(
                    result.get(
                        "problem_type",
                        ""
                    )
                ).lower()
                == "classification"
            ):

                st.markdown(
                    """
                    **Accuracy** — Overall percentage of
                    predictions that were correct.

                    **Precision** — Of the samples predicted
                    as positive, how many were actually positive.

                    **Recall** — Of the actual positive samples,
                    how many the model successfully identified.

                    **F1 Score** — Balance between precision
                    and recall. Higher is generally better.
                    """
                )

            elif (
                str(
                    result.get(
                        "problem_type",
                        ""
                    )
                ).lower()
                == "regression"
            ):

                st.markdown(
                    """
                    **MAE** — Average absolute difference between
                    predictions and actual values. Lower is better.

                    **RMSE** — Similar to MAE but penalises larger
                    prediction errors more strongly. Lower is better.

                    **R²** — Indicates how much of the variation in
                    the target is explained by the model.
                    Higher is generally better.
                    """
                )

            else:

                st.write(
                    "Metrics are calculated using "
                    "the held-out test dataset."
                )


    else:

        st.info(
            "No model comparison metrics "
            "were returned."
        )

        # Useful fallback for unexpected backend structure
        if metrics:

            with st.expander(
                "View Raw Metrics"
            ):

                st.json(
                    metrics
                )


    # =====================================================
    # TRAINING RETRY INFORMATION
    # =====================================================

    training = result.get(
        "training",
        {}
    )

    if isinstance(
        training,
        dict
    ):

        training_retries = (
            training.get(
                "retries"
            )
        )

        if training_retries is not None:

            st.caption(
                "Training repair attempts: "
                f"{training_retries}"
            )


    st.divider()


    # =====================================================
    # DOWNLOAD CLEANED DATASET ONLY
    # =====================================================

    st.header(
        "⬇️ Download Cleaned Dataset"
    )

    st.write(
        "Download the dataset produced by "
        "DSmith AI after preprocessing."
    )

    downloads = result.get(
        "downloads",
        {}
    )

    cleaned_endpoint = (
        downloads.get(
            "cleaned_dataset"
        )
        if isinstance(
            downloads,
            dict
        )
        else None
    )


    if cleaned_endpoint:

        try:

            cleaned_response = requests.get(
                f"{API_URL}{cleaned_endpoint}",
                timeout=120
            )


            # -------------------------------------------------
            # DOWNLOAD AVAILABLE
            # -------------------------------------------------

            if (
                cleaned_response.status_code
                == 200
            ):

                st.download_button(
                    label=(
                        "📥 Download Cleaned Dataset"
                    ),
                    data=cleaned_response.content,
                    file_name="cleaned_dataset.csv",
                    mime="text/csv",
                    use_container_width=True
                )


            # -------------------------------------------------
            # FILE EXPIRED
            # -------------------------------------------------

            elif (
                cleaned_response.status_code
                == 404
            ):

                st.warning(
                    "⚠️ The cleaned dataset has "
                    "expired or is no longer available. "
                    "Run the analysis again to regenerate it."
                )


            # -------------------------------------------------
            # OTHER DOWNLOAD ERROR
            # -------------------------------------------------

            else:

                st.warning(
                    "Could not retrieve the "
                    "cleaned dataset."
                )


        except requests.exceptions.RequestException:

            st.warning(
                "Could not connect to the backend "
                "to retrieve the cleaned dataset."
            )

    else:

        st.info(
            "Cleaned dataset download "
            "is unavailable."
        )


    st.caption(
        "Generated datasets are stored temporarily. "
        "Download the cleaned dataset after analysis."
    )