"""
Data Insight Assistant - Full-feature app.py
Features:
- Login system (basic)
- Upload CSV/XLSX with auto-clean
- AI summary & column insights (OpenAI)
- Smart chart builder (Altair) + AI chart recommendation
- Simple trend forecasting (Linear Regression)
- Chat with your data (session memory)
- Notes/comments per file
- PDF report export (FPDF)
"""

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
import os
import io
import json
import tempfile
from datetime import datetime, timezone
from sklearn.linear_model import LinearRegression
from openai import OpenAI
from fpdf import FPDF

# ---------------------------
# CONFIG
# ---------------------------
st.set_page_config(page_title="Data Insight Assistant", layout="wide")
APP_PASSWORD = st.secrets.get("APP_PASSWORD", os.environ.get("APP_PASSWORD", "demo_pass"))  # change for production
OPENAI_KEY = st.secrets.get("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY", ""))

NOTES_DIR = "notes"
os.makedirs(NOTES_DIR, exist_ok=True)

# ---------------------------
# UTILITIES
# ---------------------------
def load_notes(path):
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

def save_note(path, author, text):
    notes = load_notes(path)
    notes.append({"timestamp": datetime.now(timezone.utc).isoformat(), "author": author, "note": text})
    with open(path, "w", encoding="utf-8") as f:
        json.dump(notes, f, ensure_ascii=False, indent=2)

def notes_to_text(notes):
    return "\n\n".join(f"[{n['timestamp']}] {n['author']} — {n['note']}" for n in notes)

def df_to_sample_text(df, n=10):
    return df.head(n).to_markdown(index=False)

def safe_filename(name: str):
    return "".join(c for c in name if c.isalnum() or c in "-_.").strip()

import warnings
import pandas as pd

def detect_schema_and_fix(df: pd.DataFrame):
    """
    Automatically converts columns to numeric or datetime types where possible,
    detects schema, and returns the cleaned dataframe along with a schema dict.
    """
    # Convert numeric columns safely
    for c in df.columns:
        try:
            df[c] = pd.to_numeric(df[c])
        except Exception:
            pass

    # Detect schema
    schema = {"numeric": [], "categorical": [], "datetime": []}
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            schema["numeric"].append(c)
        else:
            # parse datetimes safely, suppress warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                parsed = pd.to_datetime(df[c], errors="coerce")
            if parsed.notna().sum() / max(1, len(parsed)) > 0.5:
                df[c] = parsed
                schema["datetime"].append(c)
            else:
                schema["categorical"].append(c)
    return df, schema

def create_pdf_report(title, ai_summary, chart_image_path=None, out_path="report.pdf"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, title, ln=True, align="C")
    pdf.ln(6)
    pdf.set_font("Arial", size=11)
    pdf.multi_cell(0, 6, "AI Summary:")
    pdf.set_font("Arial", size=10)
    pdf.multi_cell(0, 6, ai_summary or "No AI summary available.")
    pdf.ln(6)
    if chart_image_path and os.path.exists(chart_image_path):
        pdf.image(chart_image_path, w=180)
    pdf.output(out_path)
    return out_path

# ---------------------------
# AUTHENTICATION (removed)
# ---------------------------
# Login removed so the app loads immediately
st.markdown("## 📊 Data Insight Assistant")
st.caption("Note: authentication removed for demo. Contact me if you need a secured demo.")

# ---------------------------
# OPENAI CLIENT (if available)
# ---------------------------
client = None
if OPENAI_KEY:
    try:
        client = OpenAI(api_key=OPENAI_KEY)
    except Exception as e:
        st.warning("OpenAI client init failed: " + str(e))
else:
    st.info("OpenAI key not set — AI features will be disabled. Set OPENAI_API_KEY in environment or Streamlit secrets.")

# ---------------------------
# APP LAYOUT (tabs)
# ---------------------------
st.title("📊 Data Insight Assistant")
tabs = st.tabs(["Upload & Clean", "AI Insights", "Charts & Recommender", "Forecast", "Chat", "Notes & PDF"])
tab_upload, tab_ai, tab_charts, tab_forecast, tab_chat, tab_notes = tabs

# ---------------------------
# Tab: Upload & Clean
# ---------------------------
with tab_upload:
    st.header("1) Upload & Auto-clean")
    uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx", "xls"])
    if uploaded_file:
        try:
            if uploaded_file.name.lower().endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            st.session_state["raw_df"] = df.copy()
            df, schema = detect_schema_and_fix(df.copy())
            st.session_state["df"] = df
            st.session_state["schema"] = schema
            st.success(f"Loaded `{uploaded_file.name}` — shape: {df.shape}")
            st.subheader("Preview")
            st.dataframe(df.head(50))
            st.subheader("Schema")
            st.write(schema)
            st.subheader("Data Quality")
            st.write("Missing values per column:")
            st.write(df.isnull().sum())
            if st.button("Auto-clean: forward-fill + drop duplicates"):
                df_clean = df.fillna(method="ffill").drop_duplicates().reset_index(drop=True)
                st.session_state["df"] = df_clean
                st.success("Auto-clean applied")
                st.dataframe(df_clean.head(20))
        except Exception as e:
            st.error("Failed to read file: " + str(e))
    else:
        st.info("Upload a CSV or Excel file to get started.")

# ---------------------------
# Tab: AI Insights
# ---------------------------
with tab_ai:
    st.header("2) AI Summary & Column Insights")
    if "df" not in st.session_state:
        st.warning("Upload and clean data first.")
    else:
        df = st.session_state["df"]
        st.subheader("Dataset preview")
        st.dataframe(df.head(10))
        if client is None:
            st.info("OpenAI not configured. Set OPENAI_API_KEY to enable AI.")
        else:
            if st.button("Generate AI Summary"):
                with st.spinner("Analyzing with AI..."):
                    prompt = f"You are a data analyst. Provide 4 concise bullet-point insights (trends, outliers, correlations) for this dataset sample:\n\n{df_to_sample_text(df, 10)}"
                    try:
                        resp = client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[{"role":"user","content":prompt}]
                        )
                        ai_summary = resp.choices[0].message.content
                        st.markdown("### 🤖 AI Summary")
                        st.write(ai_summary)
                        st.session_state["last_ai_summary"] = ai_summary
                    except Exception as e:
                        st.error("AI error: " + str(e))
            if st.button("Generate Column Insights"):
                if client:
                    with st.spinner("Generating column insights..."):
                        prompt = f"You are a data analyst. For each column in this dataset {list(df.columns)}, write 1-2 sentences explaining its meaning, expected ranges, and what to watch for."
                        try:
                            resp = client.chat.completions.create(
                                model="gpt-4o-mini",
                                messages=[{"role":"user","content":prompt}]
                            )
                            st.markdown("### 🧩 Column Insights")
                            st.write(resp.choices[0].message.content)
                        except Exception as e:
                            st.error("AI error: " + str(e))
                else:
                    st.info("OpenAI not configured.")

# ---------------------------
# Tab: Charts & Recommender (Upgraded)
# ---------------------------
with tab_charts:
    st.header("3) Smart Chart Builder & AI Recommender (Upgraded)")

    if "df" not in st.session_state:
        st.warning("Upload data first")
    else:
        df = st.session_state["df"]
        schema = st.session_state.get("schema", {})

        # Auto-detect column types
        numeric_cols = schema.get("numeric", [])
        categorical_cols = schema.get("categorical", []) + schema.get("datetime", [])

        st.markdown("#### Select columns for charting")
        col1, col2 = st.columns([2,1])

        with col1:
            x_cols = st.multiselect(
                "X-axis (select one or more)", 
                options=df.columns.tolist(), 
                default=[df.columns[0]]
            )
            y_cols = st.multiselect(
                "Y-axis (numeric, select one or more)", 
                options=numeric_cols, 
                default=[numeric_cols[0]] if numeric_cols else []
            )

            chart_kind = st.selectbox(
                "Chart kind (Auto-detect by default)",
                ["Auto", "Line", "Bar", "Scatter", "Histogram", "Box", "Stacked Bar", "Multi-line"]
            )

            if st.button("Generate Chart"):
                try:
                    # Heuristic: Auto-detect chart type if selected "Auto"
                    if chart_kind == "Auto":
                        if all(col in numeric_cols for col in y_cols) and all(col in numeric_cols+categorical_cols for col in x_cols):
                            kind = "Multi-line" if len(y_cols) > 1 else "Line"
                        elif all(col in categorical_cols for col in x_cols) and len(y_cols) > 0:
                            kind = "Stacked Bar" if len(y_cols) > 1 else "Bar"
                        elif len(y_cols) == 1 and len(x_cols) == 1:
                            kind = "Scatter"
                        else:
                            kind = "Line"
                    else:
                        kind = chart_kind

                    st.markdown(f"**Rendering {kind} chart**")

                    # Generate charts based on kind
                    if kind in ["Line", "Multi-line"]:
                        for y in y_cols:
                            chart = alt.Chart(df).mark_line(point=True).encode(
                                x=x_cols[0],  # multi-X for line: first column is main axis
                                y=y,
                                tooltip=list(df.columns)
                            )
                            st.altair_chart(chart, use_container_width=True)

                    elif kind == "Bar":
                        for y in y_cols:
                            chart = alt.Chart(df).mark_bar().encode(
                                x=x_cols[0],
                                y=y,
                                tooltip=list(df.columns)
                            )
                            st.altair_chart(chart, use_container_width=True)

                    elif kind == "Stacked Bar":
                        if len(y_cols) > 1 and len(x_cols) == 1:
                            chart = alt.Chart(df).transform_fold(
                                y_cols,
                                as_=['Category','Value']
                            ).mark_bar().encode(
                                x=x_cols[0],
                                y='Value:Q',
                                color='Category:N',
                                tooltip=list(df.columns)
                            )
                            st.altair_chart(chart, use_container_width=True)

                    elif kind == "Scatter":
                        if len(x_cols) >= 1 and len(y_cols) >= 1:
                            chart = alt.Chart(df).mark_circle(size=60).encode(
                                x=x_cols[0],
                                y=y_cols[0],
                                tooltip=list(df.columns)
                            )
                            st.altair_chart(chart, use_container_width=True)

                    elif kind == "Histogram":
                        for col in y_cols:
                            chart = alt.Chart(df).mark_bar().encode(
                                alt.X(col+":Q", bin=True),
                                y='count()'
                            )
                            st.altair_chart(chart, use_container_width=True)

                    elif kind == "Box":
                        for col in y_cols:
                            chart = alt.Chart(df).mark_boxplot().encode(
                                x=x_cols[0] if x_cols else col,
                                y=col
                            )
                            st.altair_chart(chart, use_container_width=True)

                except Exception as e:
                    st.error("Chart error: " + str(e))

        # ---- AI Chart Suggestion Column ----
        with col2:
            st.markdown("#### AI Chart Suggestion")
            if st.button("Ask AI: best chart for my columns"):
                if "df" in st.session_state and y_cols:
                    if client:
                        sample = df_to_sample_text(df, 8)
                        prompt = f"You are a data visualization expert. Given this data sample:\n{sample}\nWhat is the best chart type for X columns {x_cols} and Y columns {y_cols}? Suggest chart type and a one-sentence reason."
                        try:
                            resp = client.chat.completions.create(
                                model="gpt-4o-mini",
                                messages=[{"role":"user","content":prompt}]
                            )
                            st.info(resp.choices[0].message.content)
                        except Exception as e:
                            st.error("AI error: " + str(e))
                    else:
                        st.info("OpenAI key not set — AI chart suggestions disabled.")

# ---------------------------
# Tab: Forecast
# ---------------------------
with tab_forecast:
    st.header("4) Simple Trend Forecasting")
    if "df" not in st.session_state:
        st.warning("Upload data first")
    else:
        df = st.session_state["df"]
        schema = st.session_state.get("schema", {})
        numeric_cols = schema.get("numeric", [])

        if not numeric_cols:
            st.info("No numeric columns to forecast.")
        else:
            target = st.selectbox("Select numeric column to forecast", numeric_cols)
            use_index_as_time = st.checkbox("Use row index as time series (if no datetime column available)", value=True)
            if st.button("Run Forecast (next 5)"):
                try:
                    series = df[target].dropna().values.reshape(-1, 1)
                    X = np.arange(len(series)).reshape(-1, 1)
                    model = LinearRegression().fit(X, series)
                    future_X = np.arange(len(series), len(series) + 5).reshape(-1, 1)
                    preds = model.predict(future_X).flatten()
                    st.write("Predicted next 5 values (approx):", np.round(preds, 2))
                    chart_vals = np.concatenate([series.flatten(), preds])
                    st.line_chart(chart_vals)
                except Exception as e:
                    st.error("Forecast failed: " + str(e))

# ---------------------------
# Tab: Chat with Data
# ---------------------------
with tab_chat:
    st.header("5) Chat with your data (session memory)")
    if "df" not in st.session_state:
        st.warning("Upload data first")
    else:
        df = st.session_state["df"]
        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []

        st.markdown("Ask the AI questions about the dataset. It uses the last 6 messages as context.")
        user_q = st.text_input("Your question")
        if st.button("Ask AI"):
            if not user_q.strip():
                st.info("Type a question first")
            elif client is None:
                st.info("OpenAI key not set.")
            else:
                with st.spinner("AI thinking..."):
                    # Build prompt with small context and data sample
                    recent = st.session_state["chat_history"][-6:]
                    context = "\n".join(f"{m['role']}: {m['content']}" for m in recent)
                    sample = df_to_sample_text(df, 10)
                    prompt = f"You are a helpful data analyst. Context:\n{context}\n\nData sample:\n{sample}\n\nQuestion:\n{user_q}"
                    try:
                        resp = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role":"user","content":prompt}])
                        ans = resp.choices[0].message.content
                        st.session_state["chat_history"].append({"role":"user","content":user_q})
                        st.session_state["chat_history"].append({"role":"assistant","content":ans})
                    except Exception as e:
                        st.error("AI error: " + str(e))

        if st.session_state.get("chat_history"):
            st.markdown("### Conversation")
            for msg in st.session_state["chat_history"][-12:]:
                if msg["role"] == "user":
                    st.markdown(f"**You:** {msg['content']}")
                else:
                    st.markdown(f"**AI:** {msg['content']}")

# ---------------------------
# Tab: Notes & PDF
# ---------------------------
with tab_notes:
    st.header("6) Notes & Export")
    if "df" not in st.session_state:
        st.warning("Upload data first")
    else:
        df = st.session_state["df"]
        uploaded_name = st.session_state.get("raw_df_name", "uploaded_data")
        # if we have file stored in session, use that name; else safe default
        uploaded_name = st.session_state.get("last_uploaded_name", uploaded_name)
        notes_file = os.path.join(NOTES_DIR, f"notes__{safe_filename(uploaded_name)}.json")

        st.subheader("Notes")
        author = st.text_input("Your name for notes", value="Analyst")
        note_text = st.text_area("Write a note about this dataset")
        if st.button("Save Note"):
            if note_text.strip():
                save_note(notes_file, author, note_text.strip())
                st.success("Note saved")
        stored = load_notes(notes_file)
        if stored:
            st.write("Saved notes (recent first):")
            for n in reversed(stored[-20:]):
                st.markdown(f"**{n['author']}** · *{n['timestamp']}*")
                st.write(n['note'])
                st.markdown("---")

        st.subheader("PDF Report")
        report_title = st.text_input("Report title", value=f"Data Report - {datetime.now().strftime('%Y%m%d_%H%M')}")
        include_ai = st.checkbox("Include last AI summary (if available)", value=True)
        if st.button("Generate PDF Report"):
            # create chart image for report
            tmp_img = None
            try:
                # draw a simple sample chart of first numeric column
                num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
                if num_cols:
                    fig, ax = plt.subplots(figsize=(6,3))
                    ax.plot(df[num_cols[0]].dropna().values)
                    ax.set_title(num_cols[0])
                    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                    fig.savefig(tmp.name, bbox_inches="tight")
                    tmp_img = tmp.name
                    tmp.close()
                ai_summary = st.session_state.get("last_ai_summary", "")
                out_path = create_pdf_report(report_title, ai_summary, tmp_img, out_path=f"{report_title}.pdf")
                with open(out_path, "rb") as f:
                    st.download_button("Download PDF", data=f, file_name=f"{report_title}.pdf", mime="application/pdf")
                st.success("PDF ready")
            except Exception as e:
                st.error("Failed to generate PDF: " + str(e))
            finally:
                if tmp_img and os.path.exists(tmp_img):
                    os.unlink(tmp_img)

# ---------------------------
# END
# ---------------------------

