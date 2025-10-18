"""
Data Insight Assistant Pro - Academic & Enterprise Edition
Enhanced with robust data cleaning, statistical analysis, and publication-ready outputs
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
# STATISTICAL IMPORTS
# ---------------------------
from scipy import stats
from scipy.stats import shapiro, normaltest
import warnings
warnings.filterwarnings('ignore')

# ---------------------------
# CONFIGURATION
# ---------------------------
st.set_page_config(page_title="Data Insight Assistant Pro", layout="wide")
APP_PASSWORD = st.secrets.get("APP_PASSWORD", os.environ.get("APP_PASSWORD", "demo_pass"))
OPENAI_KEY = st.secrets.get("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY", ""))

NOTES_DIR = "notes"
os.makedirs(NOTES_DIR, exist_ok=True)

# Role-based access system
USER_ROLES = {
    "student": ["view", "export", "basic_analysis"],
    "researcher": ["view", "analyze", "export", "share", "advanced_analysis"],
    "professor": ["view", "analyze", "export", "share", "manage_users", "create_assignments"],
    "admin": ["all"]
}

# ---------------------------
# DATA CLEANING & UTILITIES
# ---------------------------
def robust_data_cleaning(df: pd.DataFrame):
    """
    Comprehensive data cleaning: handles empty strings, whitespace, NaNs, mixed types
    """
    df_clean = df.copy()
    
    # Clean each column
    for col in df_clean.columns:
        # Handle empty strings and whitespace
        df_clean[col] = df_clean[col].replace(r'^\s*$', np.nan, regex=True)
        df_clean[col] = df_clean[col].replace('', np.nan)
        
        # Try numeric conversion first
        original_dtype = df_clean[col].dtype
        numeric_converted = pd.to_numeric(df_clean[col], errors='coerce')
        
        # If most values converted successfully, use numeric
        if numeric_converted.notna().sum() / len(df_clean) > 0.7:
            df_clean[col] = numeric_converted
        else:
            # Try datetime conversion
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                datetime_converted = pd.to_datetime(df_clean[col], errors='coerce')
                if datetime_converted.notna().sum() / len(df_clean) > 0.7:
                    df_clean[col] = datetime_converted
    
    return df_clean

def detect_schema(df: pd.DataFrame):
    """Auto-detection of numeric, categorical, datetime columns"""
    schema = {"numeric": [], "categorical": [], "datetime": []}
    
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            schema["numeric"].append(col)
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            schema["datetime"].append(col)
        else:
            schema["categorical"].append(col)
    
    return schema

def load_notes(path):
    """Load notes from JSON file"""
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

def save_note(path, author, text):
    """Save note to JSON file"""
    notes = load_notes(path)
    notes.append({"timestamp": datetime.now(timezone.utc).isoformat(), "author": author, "note": text})
    with open(path, "w", encoding="utf-8") as f:
        json.dump(notes, f, ensure_ascii=False, indent=2)

def df_to_sample_text(df, n=10):
    """Convert dataframe sample to markdown text"""
    return df.head(n).to_markdown(index=False)

def safe_filename(name: str):
    """Create safe filename"""
    return "".join(c for c in name if c.isalnum() or c in "-_.").strip()

# ---------------------------
# STATISTICAL ANALYSIS FUNCTIONS
# ---------------------------
def comprehensive_statistical_analysis(df, target_col=None):
    """Run automated statistical tests and return insights"""
    results = {
        "descriptive_stats": {},
        "normality_tests": {}, 
        "correlation_analysis": {},
        "group_comparisons": {},
        "recommendations": []
    }
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Descriptive statistics
    for col in numeric_cols:
        results["descriptive_stats"][col] = {
            "mean": df[col].mean(),
            "std": df[col].std(),
            "min": df[col].min(),
            "max": df[col].max(),
            "skew": df[col].skew(),
            "kurtosis": df[col].kurtosis()
        }
    
    # Normality tests
    for col in numeric_cols[:5]:  # Limit to first 5 columns
        if len(df[col].dropna()) > 3:
            stat, p_value = shapiro(df[col].dropna())
            results["normality_tests"][col] = {
                "test": "Shapiro-Wilk",
                "statistic": stat,
                "p_value": p_value,
                "normal": p_value > 0.05
            }
    
    # Correlation analysis
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr()
        results["correlation_analysis"]["matrix"] = corr_matrix
        
        # Find strong correlations
        strong_corrs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:
                    strong_corrs.append({
                        "var1": corr_matrix.columns[i],
                        "var2": corr_matrix.columns[j], 
                        "correlation": corr_val
                    })
        results["correlation_analysis"]["strong_correlations"] = strong_corrs
    
    # Generate analytical recommendations
    if target_col and target_col in numeric_cols:
        results["recommendations"].append(f"Consider regression analysis with {target_col} as dependent variable")
    
    for col in numeric_cols:
        if abs(results["descriptive_stats"][col]["skew"]) > 2:
            results["recommendations"].append(f"Variable '{col}' is highly skewed - consider transformation")
    
    return results

def run_ttest_independent(df, group_col, value_col):
    """Independent t-test between groups"""
    groups = df[group_col].unique()
    if len(groups) != 2:
        return "T-test requires exactly 2 groups"
    
    group1 = df[df[group_col] == groups[0]][value_col].dropna()
    group2 = df[df[group_col] == groups[1]][value_col].dropna()
    
    t_stat, p_value = stats.ttest_ind(group1, group2)
    
    return {
        "test": "Independent t-test",
        "groups": [str(groups[0]), str(groups[1])],
        "t_statistic": t_stat,
        "p_value": p_value,
        "significant": p_value < 0.05,
        "group1_mean": group1.mean(),
        "group2_mean": group2.mean()
    }

def run_anova(df, group_col, value_col):
    """One-way ANOVA"""
    groups = []
    for group in df[group_col].unique():
        groups.append(df[df[group_col] == group][value_col].dropna())
    
    f_stat, p_value = stats.f_oneway(*groups)
    
    return {
        "test": "One-way ANOVA",
        "f_statistic": f_stat,
        "p_value": p_value,
        "significant": p_value < 0.05,
        "groups_compared": len(groups)
    }

# ---------------------------
# VISUALIZATION FUNCTIONS
# ---------------------------
def create_publication_chart(df, x_col, y_col, chart_type="scatter", style="nature", title=None):
    """Create academic publication-ready charts"""
    
    # Set style based on journal requirements
    if style == "nature":
        plt.style.use('seaborn-v0_8-whitegrid')
        font_size = 8
        line_width = 0.8
        color_palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    elif style == "apa":
        plt.style.use('seaborn-v0_8-white')
        font_size = 10
        line_width = 1.0
        color_palette = ["#000000", "#666666", "#999999"]
    else:
        plt.style.use('seaborn-v0_8-paper')
        font_size = 9
        line_width = 1.0
        color_palette = sns.color_palette("husl", 8)
    
    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
    
    try:
        if chart_type == "scatter":
            ax.scatter(df[x_col], df[y_col], alpha=0.7, s=30, color=color_palette[0])
            # Add trend line
            z = np.polyfit(df[x_col].dropna(), df[y_col].dropna(), 1)
            p = np.poly1d(z)
            ax.plot(df[x_col], p(df[x_col]), "r--", alpha=0.8, linewidth=line_width)
            
        elif chart_type == "bar":
            data_to_plot = df.groupby(x_col)[y_col].mean()
            ax.bar(data_to_plot.index.astype(str), data_to_plot.values, 
                   color=color_palette, alpha=0.8)
            
        elif chart_type == "box":
            df.boxplot(column=y_col, by=x_col, ax=ax, grid=False)
            
        elif chart_type == "violin":
            data_to_plot = []
            labels = []
            for group in df[x_col].unique():
                data_to_plot.append(df[df[x_col] == group][y_col].dropna())
                labels.append(str(group))
            ax.violinplot(data_to_plot, showmeans=True, showmedians=True)
            ax.set_xticks(range(1, len(labels) + 1))
            ax.set_xticklabels(labels)
            
        elif chart_type == "line":
            ax.plot(df[x_col], df[y_col], marker='o', linewidth=line_width, color=color_palette[0])
            
        # Styling for publication
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', which='major', labelsize=font_size-1)
        
        if title:
            ax.set_title(title, fontsize=font_size+2, pad=10)
        else:
            ax.set_title(f"{y_col} by {x_col}", fontsize=font_size+2, pad=10)
            
        ax.set_xlabel(x_col, fontsize=font_size, labelpad=8)
        ax.set_ylabel(y_col, fontsize=font_size, labelpad=8)
        
        plt.tight_layout()
        return fig
        
    except Exception as e:
        st.error(f"Chart creation failed: {str(e)}")
        return None

def export_high_quality_plot(fig, filename="plot.png", format="png", dpi=300):
    """Export plot in publication-quality format"""
    if fig:
        fig.savefig(filename, format=format, dpi=dpi, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        return filename
    return None

# ---------------------------
# AI RESEARCH ASSISTANT
# ---------------------------
def create_research_assistant(client):
    """Specialized AI agent for academic research guidance"""
    
    def research_advisor(question, data_context=None, research_field=None):
        system_prompt = f"""
        You are an expert research methodology advisor and statistical consultant. 
        Field: {research_field or 'General'}
        Data Context: {data_context or 'Not provided'}
        
        Provide:
        1. Statistical test recommendations with justifications
        2. Research methodology guidance
        3. Results interpretation framework
        4. Publication-ready insights
        5. Common pitfalls to avoid
        
        Be rigorous but accessible in your methodological guidance.
        """
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question}
                ],
                temperature=0.3
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Research advisor unavailable: {str(e)}"
    
    return research_advisor

# ---------------------------
# PDF REPORT GENERATION
# ---------------------------
def create_pdf_report(title, ai_summary, statistical_insights, chart_image_path=None, out_path="report.pdf"):
    """Generate comprehensive PDF report"""
    pdf = FPDF()
    pdf.add_page()
    
    # Title
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, title, ln=True, align="C")
    pdf.ln(10)
    
    # AI Summary
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "AI Analysis Summary", ln=True)
    pdf.set_font("Arial", size=10)
    pdf.multi_cell(0, 6, ai_summary or "No AI summary available.")
    pdf.ln(5)
    
    # Statistical Insights
    if statistical_insights:
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "Statistical Insights", ln=True)
        pdf.set_font("Arial", size=10)
        pdf.multi_cell(0, 6, statistical_insights)
        pdf.ln(5)
    
    # Chart
    if chart_image_path and os.path.exists(chart_image_path):
        pdf.image(chart_image_path, w=180)
    
    # Footer
    pdf.ln(10)
    pdf.set_font("Arial", "I", 8)
    pdf.cell(0, 10, f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')} by Data Insight Assistant Pro", ln=True)
    
    pdf.output(out_path)
    return out_path

# ---------------------------
# INITIALIZE SESSION STATE
# ---------------------------
if 'user_role' not in st.session_state:
    st.session_state.user_role = "researcher"

if 'current_workspace' not in st.session_state:
    st.session_state.current_workspace = "default"

# ---------------------------
# OPENAI CLIENT
# ---------------------------
client = None
if OPENAI_KEY:
    try:
        client = OpenAI(api_key=OPENAI_KEY)
        st.session_state.research_assistant = create_research_assistant(client)
    except Exception as e:
        st.warning("OpenAI client init failed: " + str(e))
else:
    st.sidebar.info("OpenAI key not set — AI features disabled")

# ---------------------------
# MAIN APP LAYOUT
# ---------------------------
st.title("📊 Data Insight Assistant Pro")
st.markdown("**Academic & Enterprise Edition** - Statistical analysis, AI research assistance, and publication-ready outputs")

# Enhanced tabs
tabs = st.tabs(["Upload & Clean", "AI Insights", "Statistical Analysis", "Visualization", "Forecast", "Chat", "Notes & PDF"])
tab_upload, tab_ai, tab_stats, tab_viz, tab_forecast, tab_chat, tab_notes = tabs

# ---------------------------
# TAB 1: UPLOAD & CLEAN
# ---------------------------
with tab_upload:
    st.header("📁 Data Upload & Automated Cleaning")
    
    # Role selector
    st.sidebar.markdown("---")
    st.sidebar.subheader("👤 Role Settings")
    demo_role = st.sidebar.selectbox("Select Role", list(USER_ROLES.keys()), index=1)
    st.session_state.user_role = demo_role
    st.sidebar.info(f"**Role:** {st.session_state.user_role}")
    
    uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx", "xls"])
    
    if uploaded_file:
        try:
            # Load data
            if uploaded_file.name.lower().endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            # Store original
            st.session_state["raw_df"] = df.copy()
            st.session_state["last_uploaded_name"] = uploaded_file.name
            
            # Robust cleaning
            with st.spinner("Performing robust data cleaning..."):
                df_clean = robust_data_cleaning(df)
                schema = detect_schema(df_clean)
                
            st.session_state["df"] = df_clean
            st.session_state["schema"] = schema
            
            st.success(f"✅ Loaded `{uploaded_file.name}` — Shape: {df_clean.shape}")
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Data Preview")
                st.dataframe(df_clean.head(20))
            
            with col2:
                st.subheader("Schema Detection")
                st.json(schema)
                
                st.subheader("Data Quality")
                missing_data = df_clean.isnull().sum()
                st.write("Missing values per column:")
                st.dataframe(missing_data[missing_data > 0])
            
            # Auto-clean options
            if st.button("🔄 Apply Advanced Cleaning"):
                df_final = df_clean.fillna(method="ffill").drop_duplicates().reset_index(drop=True)
                st.session_state["df"] = df_final
                st.success("Advanced cleaning applied")
                st.dataframe(df_final.head(20))
                
        except Exception as e:
            st.error(f"Failed to process file: {str(e)}")
    else:
        st.info("👆 Upload a CSV or Excel file to begin analysis")

# ---------------------------
# TAB 2: AI INSIGHTS
# ---------------------------
with tab_ai:
    st.header("🧠 AI Research Assistant & Insights")
    
    if "df" not in st.session_state:
        st.warning("Upload data in the Upload & Clean tab first.")
    else:
        df = st.session_state["df"]
        
        # Research Advisor Section
        st.subheader("🔬 Research Methodology Advisor")
        
        research_field = st.selectbox("Research Field", 
                                    ["Psychology", "Biology", "Economics", "Sociology", 
                                     "Medicine", "Engineering", "Education", "Other"])
        
        research_question = st.text_area("Research Question:", 
                                       placeholder="e.g., 'What statistical test should I use for pre-post intervention analysis?'")
        
        if st.button("🎯 Get Research Advice") and research_question:
            if client:
                with st.spinner("Consulting research advisor..."):
                    advice = st.session_state.research_assistant(
                        research_question, 
                        data_context=f"Data: {df.shape}, Columns: {list(df.columns)}",
                        research_field=research_field
                    )
                    st.markdown("### Research Advisor Response")
                    st.write(advice)
            else:
                st.info("OpenAI not configured for research advisor.")
        
        st.markdown("---")
        
        # AI Analysis Section
        st.subheader("📊 AI-Powered Data Analysis")
        st.dataframe(df.head(10))
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📈 Generate Statistical Summary"):
                if client:
                    with st.spinner("Generating academic insights..."):
                        sample = df_to_sample_text(df, 15)
                        prompt = f"""
                        As a statistical consultant, analyze this dataset:
                        
                        {sample}
                        
                        Provide:
                        1. Key descriptive insights in academic style
                        2. Recommended statistical tests
                        3. Appropriate visualization types
                        4. Research methodology notes
                        5. Data quality assessment
                        
                        Focus on statistical rigor and methodological soundness.
                        """
                        try:
                            resp = client.chat.completions.create(
                                model="gpt-4o-mini",
                                messages=[{"role":"user","content":prompt}]
                            )
                            ai_summary = resp.choices[0].message.content
                            st.session_state["last_ai_summary"] = ai_summary
                            st.markdown("### Academic Analysis")
                            st.write(ai_summary)
                        except Exception as e:
                            st.error(f"AI analysis failed: {str(e)}")
                else:
                    st.info("OpenAI not configured.")
        
        with col2:
            if st.button("🔍 Generate Column Insights"):
                if client:
                    with st.spinner("Analyzing columns..."):
                        prompt = f"""
                        For each column in dataset {list(df.columns)}, provide:
                        - Meaning and expected ranges
                        - Data quality considerations  
                        - Analytical recommendations
                        - Common pitfalls
                        """
                        try:
                            resp = client.chat.completions.create(
                                model="gpt-4o-mini",
                                messages=[{"role":"user","content":prompt}]
                            )
                            st.markdown("### Column Insights")
                            st.write(resp.choices[0].message.content)
                        except Exception as e:
                            st.error(f"AI analysis failed: {str(e)}")
                else:
                    st.info("OpenAI not configured.")

# ---------------------------
# TAB 3: STATISTICAL ANALYSIS
# ---------------------------
with tab_stats:
    st.header("🔬 Statistical Analysis Suite")
    
    if "df" not in st.session_state:
        st.warning("Upload data in the Upload & Clean tab first.")
    else:
        df = st.session_state["df"]
        
        # Comprehensive Analysis
        st.subheader("📋 Comprehensive Statistical Overview")
        
        if st.button("🚀 Run Full Statistical Analysis"):
            with st.spinner("Running comprehensive analysis..."):
                results = comprehensive_statistical_analysis(df)
                
                # Descriptive Statistics
                st.subheader("Descriptive Statistics")
                desc_df = pd.DataFrame(results["descriptive_stats"]).T
                st.dataframe(desc_df.style.format("{:.3f}"))
                
                # Normality Tests
                if results["normality_tests"]:
                    st.subheader("Normality Tests (Shapiro-Wilk)")
                    norm_df = pd.DataFrame(results["normality_tests"]).T
                    st.dataframe(norm_df)
                
                # Correlation Analysis
                if "matrix" in results["correlation_analysis"]:
                    st.subheader("Correlation Matrix")
                    corr_matrix = results["correlation_analysis"]["matrix"]
                    
                    fig, ax = plt.subplots(figsize=(10, 8))
                    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
                    st.pyplot(fig)
                    
                    # Strong Correlations
                    strong_corrs = results["correlation_analysis"].get("strong_correlations", [])
                    if strong_corrs:
                        st.subheader("Strong Correlations (|r| > 0.7)")
                        for corr in strong_corrs:
                            st.write(f"**{corr['var1']}** ↔ **{corr['var2']}**: r = {corr['correlation']:.3f}")
                
                # Recommendations
                if results["recommendations"]:
                    st.subheader("Analytical Recommendations")
                    for rec in results["recommendations"]:
                        st.write(f"• {rec}")
        
        # Hypothesis Testing
        st.subheader("🎯 Hypothesis Testing")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**T-Test (Independent Samples)**")
            group_col = st.selectbox("Grouping Variable", 
                                   [c for c in df.columns if df[c].nunique() == 2],
                                   key="ttest_group")
            value_col = st.selectbox("Numeric Variable", 
                                   df.select_dtypes(include=[np.number]).columns.tolist(),
                                   key="ttest_value")
            
            if st.button("Run T-Test") and group_col and value_col:
                result = run_ttest_independent(df, group_col, value_col)
                if isinstance(result, dict):
                    st.write(f"**T-statistic**: {result['t_statistic']:.3f}")
                    st.write(f"**P-value**: {result['p_value']:.3f}")
                    st.write(f"**Significant**: {'✅ Yes' if result['significant'] else '❌ No'}")
                    st.write(f"**Group Means**: {result['group1_mean']:.2f} vs {result['group2_mean']:.2f}")
        
        with col2:
            st.markdown("**ANOVA (Multiple Groups)**")
            anova_group = st.selectbox("Grouping Variable", 
                                     df.columns,
                                     key="anova_group")
            anova_value = st.selectbox("Numeric Variable", 
                                     df.select_dtypes(include=[np.number]).columns.tolist(),
                                     key="anova_value")
            
            if st.button("Run ANOVA") and anova_group and anova_value:
                result = run_anova(df, anova_group, anova_value)
                st.write(f"**F-statistic**: {result['f_statistic']:.3f}")
                st.write(f"**P-value**: {result['p_value']:.3f}")
                st.write(f"**Significant**: {'✅ Yes' if result['significant'] else '❌ No'}")
                st.write(f"**Groups Compared**: {result['groups_compared']}")

# ---------------------------
# TAB 4: VISUALIZATION
# ---------------------------
with tab_viz:
    st.header("📊 Advanced Visualization")
    
    if "df" not in st.session_state:
        st.warning("Upload data in the Upload & Clean tab first.")
    else:
        df = st.session_state["df"]
        schema = st.session_state.get("schema", {})
        
        st.subheader("Publication-Ready Charts")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            x_col = st.selectbox("X Variable", df.columns, key="viz_x")
        with col2:
            y_col = st.selectbox("Y Variable", 
                               df.select_dtypes(include=[np.number]).columns.tolist(),
                               key="viz_y")
        with col3:
            chart_type = st.selectbox("Chart Type", 
                                    ["scatter", "line", "bar", "box", "violin"],
                                    key="chart_type")
            chart_style = st.selectbox("Journal Style", ["nature", "apa", "academic"])
        
        if st.button("Generate Publication Chart") and x_col and y_col:
            fig = create_publication_chart(df, x_col, y_col, 
                                         chart_type=chart_type, style=chart_style)
            if fig:
                st.pyplot(fig)
                
                # Export options
                export_name = st.text_input("Export Filename", 
                                          value=f"{y_col}_by_{x_col}_{chart_type}")
                if st.button("Export High-Quality Plot"):
                    filename = export_high_quality_plot(fig, f"{export_name}.png")
                    if filename:
                        with open(filename, "rb") as f:
                            st.download_button(
                                "📥 Download Publication Plot",
                                data=f,
                                file_name=f"{export_name}.png",
                                mime="image/png"
                            )

# ---------------------------
# TAB 5: FORECAST
# ---------------------------
with tab_forecast:
    st.header("📈 Trend Forecasting")
    
    if "df" not in st.session_state:
        st.warning("Upload data first")
    else:
        df = st.session_state["df"]
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if not numeric_cols:
            st.info("No numeric columns to forecast.")
        else:
            target = st.selectbox("Select numeric column to forecast", numeric_cols)
            
            if st.button("Run Forecast (next 5 periods)"):
                try:
                    series = df[target].dropna().values.reshape(-1, 1)
                    X = np.arange(len(series)).reshape(-1, 1)
                    model = LinearRegression().fit(X, series)
                    future_X = np.arange(len(series), len(series) + 5).reshape(-1, 1)
                    preds = model.predict(future_X).flatten()
                    
                    st.write("**Predicted next 5 values:**", np.round(preds, 2))
                    
                    # Plot forecast
                    chart_vals = np.concatenate([series.flatten(), preds])
                    st.line_chart(chart_vals)
                    
                except Exception as e:
                    st.error(f"Forecast failed: {str(e)}")

# ---------------------------
# TAB 6: CHAT WITH DATA
# ---------------------------
with tab_chat:
    st.header("💬 Chat with Your Data")
    
    if "df" not in st.session_state:
        st.warning("Upload data first")
    else:
        df = st.session_state["df"]
        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []

        user_q = st.text_input("Ask a question about your data:")
        
        if st.button("Ask AI") and user_q.strip():
            if not client:
                st.info("OpenAI key not set.")
            else:
                with st.spinner("AI thinking..."):
                    recent = st.session_state["chat_history"][-6:]
                    context = "\n".join(f"{m['role']}: {m['content']}" for m in recent)
                    sample = df_to_sample_text(df, 10)
                    prompt = f"Context:\n{context}\n\nData:\n{sample}\n\nQuestion:\n{user_q}"
                    
                    try:
                        resp = client.chat.completions.create(
                            model="gpt-4o-mini", 
                            messages=[{"role":"user","content":prompt}]
                        )
                        ans = resp.choices[0].message.content
                        st.session_state["chat_history"].append({"role":"user","content":user_q})
                        st.session_state["chat_history"].append({"role":"assistant","content":ans})
                    except Exception as e:
                        st.error(f"AI error: {str(e)}")

        # Display conversation
        if st.session_state.get("chat_history"):
            st.markdown("### Conversation History")
            for msg in st.session_state["chat_history"][-12:]:
                if msg["role"] == "user":
                    st.markdown(f"**You:** {msg['content']}")
                else:
                    st.markdown(f"**AI:** {msg['content']}")

# ---------------------------
# TAB 7: NOTES & PDF EXPORT
# ---------------------------
with tab_notes:
    st.header("📝 Notes & Report Generation")
    
    if "df" not in st.session_state:
        st.warning("Upload data first")
    else:
        df = st.session_state["df"]
        uploaded_name = st.session_state.get("last_uploaded_name", "uploaded_data")
        notes_file = os.path.join(NOTES_DIR, f"notes__{safe_filename(uploaded_name)}.json")

        # Notes Section
        st.subheader("Research Notes")
        
        author = st.text_input("Your Name", value="Researcher")
        note_text = st.text_area("Add research note:")
        
        if st.button("Save Note") and note_text.strip():
            save_note(notes_file, author, note_text.strip())
            st.success("Note saved")
        
        # Display saved notes
        stored_notes = load_notes(notes_file)
        if stored_notes:
            st.write("**Saved Notes (most recent first):**")
            for n in reversed(stored_notes[-10:]):
                st.markdown(f"**{n['author']}** · *{n['timestamp']}*")
                st.write(n['note'])
                st.markdown("---")

        # PDF Report Section
        st.subheader("📄 PDF Report Generation")
        
        report_title = st.text_input("Report Title", 
                                   value=f"Data Analysis Report - {datetime.now().strftime('%Y%m%d')}")
        
        if st.button("Generate Comprehensive PDF Report"):
            with st.spinner("Generating professional report..."):
                tmp_img = None
                try:
                    # Create sample chart
                    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
                    if num_cols:
                        fig, ax = plt.subplots(figsize=(8, 4))
                        ax.plot(df[num_cols[0]].dropna().values)
                        ax.set_title(f"Trend: {num_cols[0]}")
                        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                        fig.savefig(tmp.name, bbox_inches="tight")
                        tmp_img = tmp.name
                        tmp.close()
                    
                    # Get AI summary if available
                    ai_summary = st.session_state.get("last_ai_summary", "No AI analysis performed.")
                    
                    # Generate statistical insights
                    stats_insights = "Statistical analysis not yet performed."
                    if st.session_state.get("schema"):
                        numeric_count = len(st.session_state["schema"].get("numeric", []))
                        stats_insights = f"Dataset contains {numeric_count} numeric variables suitable for statistical analysis."
                    
                    # Create PDF
                    out_path = create_pdf_report(report_title, ai_summary, stats_insights, tmp_img)
                    
                    with open(out_path, "rb") as f:
                        st.download_button(
                            "📥 Download PDF Report",
                            data=f,
                            file_name=f"{report_title}.pdf",
                            mime="application/pdf"
                        )
                    st.success("PDF report generated successfully!")
                    
                except Exception as e:
                    st.error(f"Report generation failed: {str(e)}")
                finally:
                    if tmp_img and os.path.exists(tmp_img):
                        os.unlink(tmp_img)

# ---------------------------
# SIDEBAR ENHANCEMENTS
# ---------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Quick Stats")
if "df" in st.session_state:
    df = st.session_state["df"]
    st.sidebar.write(f"**Dataset:** {df.shape[0]} rows × {df.shape[1]} cols")
    st.sidebar.write(f"**Numeric:** {len(df.select_dtypes(include=[np.number]).columns)}")
    st.sidebar.write(f"**Missing:** {df.isnull().sum().sum()}")

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Features")
st.sidebar.success("""
✅ Robust Data Cleaning
✅ Statistical Analysis  
✅ AI Research Assistant
✅ Publication-Ready Charts
✅ PDF Report Generation
✅ Role-Based Access
""")

# ---------------------------
# END OF APPLICATION
# ---------------------------
