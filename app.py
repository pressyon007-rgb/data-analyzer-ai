"""
app.py - Data Analyzer AI Platform with 4 Visual Dashboards, Custom Chart Studio, 
Statistical QA Engine, and Executive Business Q&A
"""

import io
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.utils.multiclass import type_of_target

# ReportLab Imports for PDF Generation
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

# Streamlit Page Configuration
st.set_page_config(
    page_title="Data Analyzer AI Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Theme & Layout Styling
st.markdown("""
    <style>
    .stApp { background-color: #F8FAFC; }
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f1f5f9 100%);
        border: 1px solid #e2e8f0;
        border-top: 4px solid #2563eb;
        border-radius: 12px;
        padding: 18px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        text-align: center;
    }
    .metric-title { color: #64748b; font-size: 14px; font-weight: 600; text-transform: uppercase; }
    .metric-value { color: #1e293b; font-size: 26px; font-weight: 800; margin-top: 4px; }
    .q-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-left: 5px solid #0f766e;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 12px;
    }
    .q-title { font-weight: 700; color: #0f766e; font-size: 16px; }
    .biz-card {
        background-color: #ffffff;
        border: 1px solid #cbd5e1;
        border-left: 5px solid #1e3a8a;
        border-radius: 8px;
        padding: 18px;
        margin-bottom: 15px;
    }
    .biz-title { font-weight: 700; color: #1e3a8a; font-size: 16px; margin-bottom: 6px; }
    </style>
""", unsafe_allow_html=True)

# Dataset Caching & File Loading
@st.cache_data
def load_uploaded_file(file):
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
        df.columns = df.columns.str.strip()
        return df
    except Exception as e:
        st.error(f"Error loading dataset file: {e}")
        return None

# Data Quality Profiler
def analyze_data_quality(df):
    return {
        "total_rows": len(df),
        "total_cols": len(df.columns),
        "total_missing": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum())
    }

# Statistical Question Generator
def generate_analytical_questions(df):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    questions = []
    if cat_cols and num_cols:
        questions.append({
            "category": "Segment Breakdown",
            "question": f"Which {cat_cols[0]} segment drives the highest total value in {num_cols[0]}?",
            "purpose": f"Identifies highest volume contributors across {cat_cols[0]} categories."
        })
    if len(num_cols) >= 2:
        questions.append({
            "category": "Correlation Test",
            "question": f"Is there a linear correlation between {num_cols[0]} and {num_cols[1]}?",
            "purpose": "Evaluates proportional growth relationships between metrics."
        })
    if num_cols:
        questions.append({
            "category": "Distribution & Outliers",
            "question": f"Are there extreme outliers present in {num_cols[0]}?",
            "purpose": "Highlights skewness or severe data variance."
        })
    return questions

# Business Executive Q&A Intelligence Engine
def answer_business_question(df, question_text):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    q_lower = question_text.lower()
    
    # Top / Leading Performers
    if any(k in q_lower for k in ['top', 'best', 'highest', 'max', 'leading']):
        if cat_cols and num_cols:
            top_res = df.groupby(cat_cols[0])[num_cols[0]].sum().sort_values(ascending=False).head(5)
            best_cat = top_res.index[0]
            best_val = top_res.iloc[0]
            return (f"🏆 **Top Performer Findings:**\n\n"
                    f"The top-ranked **{cat_cols[0]}** is **'{best_cat}'** with a cumulative **{num_cols[0]}** of **{best_val:,.2f}**.\n\n"
                    f"**Top 5 Leaderboard:**\n" + "\n".join([f"- **{k}**: {v:,.2f}" for k, v in top_res.items()]))
    
    # Lowest / Bottom / At-Risk Drivers
    elif any(k in q_lower for k in ['lowest', 'bottom', 'worst', 'min', 'least', 'risk']):
        if cat_cols and num_cols:
            low_res = df.groupby(cat_cols[0])[num_cols[0]].sum().sort_values(ascending=True).head(5)
            worst_cat = low_res.index[0]
            worst_val = low_res.iloc[0]
            return (f"⚠️ **Low-Performance Alert:**\n\n"
                    f"The lowest-ranked **{cat_cols[0]}** is **'{worst_cat}'** with a total **{num_cols[0]}** of **{worst_val:,.2f}**.\n\n"
                    f"**Bottom 5 Segment Breakdown:**\n" + "\n".join([f"- **{k}**: {v:,.2f}" for k, v in low_res.items()]))

    # Average Benchmark Metrics
    elif any(k in q_lower for k in ['average', 'mean', 'typical', 'avg']):
        if num_cols:
            avg_val = df[num_cols[0]].mean()
            median_val = df[num_cols[0]].median()
            return (f"📊 **Executive Average Breakdown:**\n\n"
                    f"- **Mean {num_cols[0]}**: {avg_val:,.2f}\n"
                    f"- **Median {num_cols[0]}**: {median_val:,.2f}")

    # Aggregated Revenue / Total Volume
    elif any(k in q_lower for k in ['total', 'sum', 'overall', 'revenue', 'sales', 'volume']):
        if num_cols:
            total_val = df[num_cols[0]].sum()
            return f"💰 **Total Aggregated Volume:** Cumulative **{num_cols[0]}** totals **{total_val:,.2f}** across all records."

    # General Business Context Fallback
    res = "📈 **General Business Data Overview:**\n\n"
    if num_cols:
        res += f"- Analyzed Numerical Metric: **{num_cols[0]}** (Total: {df[num_cols[0]].sum():,.2f}, Avg: {df[num_cols[0]].mean():,.2f})\n"
    if cat_cols:
        res += f"- Primary Segment Dimension: **{cat_cols[0]}** ({df[cat_cols[0]].nunique()} unique values)\n"
    return res

# Machine Learning Predictive Feature Drivers
def train_risk_model(df, target_col):
    data = df.copy().dropna()
    for col in list(data.columns):
        if col != target_col and (col.lower().endswith('id') or data[col].nunique() > 100):
            data = data.drop(columns=[col])

    cat_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
    for col in cat_cols:
        if col != target_col:
            data[col] = LabelEncoder().fit_transform(data[col].astype(str))
        
    X = data.drop(columns=[target_col])
    y = data[target_col]
    
    if type_of_target(y) == 'continuous':
        model = RandomForestRegressor(n_estimators=50, random_state=42)
    else:
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        
    model.fit(X, y)
    return pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_}).sort_values(by='Importance', ascending=False)

# PDF Report Generation (On-Demand)
def generate_pdf_report(quality_info, dashboards_dict):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle('TitleStyle', parent=styles['Heading1'], fontSize=18, textColor=colors.HexColor('#1E3A8A'))
    heading_style = ParagraphStyle('HeadingStyle', parent=styles['Heading2'], fontSize=12, textColor=colors.HexColor('#2563EB'), spaceBefore=10)
    body_style = ParagraphStyle('BodyStyle', parent=styles['Normal'], fontSize=9, textColor=colors.HexColor('#334155'))

    story = [
        Paragraph("Data Analyzer AI - Executive Report", title_style),
        Spacer(1, 10),
        Paragraph("1. Dataset Overview", heading_style)
    ]

    metrics_data = [
        ["Metric Description", "Value"],
        ["Total Rows", f"{quality_info['total_rows']:,}"],
        ["Total Columns", f"{quality_info['total_cols']:,}"],
        ["Missing Values", f"{quality_info['total_missing']:,}"],
        ["Duplicate Rows", f"{quality_info['duplicate_rows']:,}"]
    ]
    t = Table(metrics_data, colWidths=[240, 240])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2563EB')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
    ]))
    story.append(t)
    story.append(Spacer(1, 12))

    story.append(Paragraph("2. Interactive Visual Dashboards", heading_style))
    for dash_title, charts in dashboards_dict.items():
        story.append(Paragraph(f"<b>{dash_title}</b>", body_style))
        for fig_title, fig in charts:
            if fig is not None:
                try:
                    img_bytes = fig.to_image(format="png", width=600, height=280, scale=1)
                    img = Image(io.BytesIO(img_bytes), width=420, height=190)
                    story.append(Paragraph(f"<i>{fig_title}</i>", body_style))
                    story.append(img)
                    story.append(Spacer(1, 6))
                except Exception:
                    pass

    doc.build(story)
    buffer.seek(0)
    return buffer

# Sidebar Navigation Panel
st.sidebar.title("🧠 Data Analyzer AI")
uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel Dataset", type=["csv", "xlsx"])

if uploaded_file is not None:
    with st.spinner("Processing dataset..."):
        df = load_uploaded_file(uploaded_file)

    if df is not None:
        target_field = st.sidebar.selectbox("Select Target Variable:", df.columns)
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        quality_info = analyze_data_quality(df)
        questions_list = generate_analytical_questions(df)

        # -------------------------------------------------------------
        # BUILD 4 DASHBOARDS (4 CHARTS EACH = 16 CHARTS TOTAL)
        # -------------------------------------------------------------
        dashboards = {}

        # Dashboard 1: Executive Overview
        d1 = []
        c1 = cat_cols[0] if cat_cols else df.columns[0]
        n1 = num_cols[0] if num_cols else df.columns[0]
        n2 = num_cols[1] if len(num_cols) > 1 else n1

        d1.append(("Top Categorical Aggregation", px.bar(df.groupby(c1)[n1].sum().reset_index().head(10), x=c1, y=n1, color=c1, title=f"Top {c1} by {n1}")))
        d1.append(("Feature Correlation Scatter", px.scatter(df, x=n1, y=n2, title=f"Scatter: {n1} vs {n2}")))
        d1.append(("Proportion Share Donut Chart", px.pie(df, names=c1, title=f"Share: {c1}", hole=0.4)))
        d1.append(("Correlation Matrix Heatmap", px.imshow(df[num_cols].corr() if len(num_cols) >= 2 else np.array([[1]]), title="Correlation Matrix Heatmap")))
        dashboards["Dashboard 1: Executive Overview"] = d1

        # Dashboard 2: Distribution & Spread
        d2 = []
        for i in range(2):
            col = num_cols[i % len(num_cols)] if num_cols else df.columns[0]
            d2.append((f"Distribution Density: {col}", px.histogram(df, x=col, title=f"Histogram Density: {col}")))
            d2.append((f"Outlier Spread: {col}", px.box(df, y=col, title=f"Outlier Boxplot: {col}")))
        dashboards["Dashboard 2: Distribution & Spread"] = d2

        # Dashboard 3: Composition Analysis
        d3 = []
        for i in range(4):
            cat = cat_cols[i % len(cat_cols)] if cat_cols else df.columns[0]
            if i % 2 == 0:
                d3.append((f"Composition Bar {i+1}", px.bar(df[cat].value_counts().reset_index(), x=cat, y='count', title=f"Frequency Count: {cat}")))
            else:
                d3.append((f"Composition Pie {i+1}", px.pie(df[cat].value_counts().reset_index(), names=cat, values='count', title=f"Share Ratio: {cat}")))
        dashboards["Dashboard 3: Composition Analysis"] = d3

        # Dashboard 4: Advanced Relationships
        d4 = []
        d4.append(("Bivariate Feature Map", px.scatter(df, x=n1, y=n2, color=c1 if cat_cols else None, title="Multivariate Scatter Map")))
        d4.append(("Category Variance Boxplot", px.box(df, x=c1, y=n1, title=f"{n1} Variance Across {c1}")))
        d4.append(("Feature Density Breakdown", px.histogram(df, x=n1, color=c1 if cat_cols else None, title="Stacked Density Plot")))
        d4.append(("Target Column Focus", px.scatter(df, x=n1, y=target_field if target_field in num_cols else n1, title=f"Impact on Target ({target_field})")))
        dashboards["Dashboard 4: Advanced Relationships"] = d4

        # Main UI Tab Layout Navigation
        st.title("Data Analyzer AI Platform")
        
        tab_list = [
            "📋 Dataset Overview", 
            "📊 Dashboard 1", "📊 Dashboard 2", "📊 Dashboard 3", "📊 Dashboard 4",
            "🎨 Custom Chart Studio", 
            "💼 Business Q&A Assistant",
            "❓ Analytical QA Engine", 
            "🤖 Feature Importance"
        ]
        tabs = st.tabs(tab_list)

        # Tab 1: Dataset Summary
        with tabs[0]:
            st.subheader("Dataset Overview Metrics")
            m1, m2, m3, m4 = st.columns(4)
            m1.markdown(f'<div class="metric-card"><div class="metric-title">Total Records</div><div class="metric-value">{quality_info["total_rows"]:,}</div></div>', unsafe_allow_html=True)
            m2.markdown(f'<div class="metric-card"><div class="metric-title">Total Columns</div><div class="metric-value">{quality_info["total_cols"]:,}</div></div>', unsafe_allow_html=True)
            m3.markdown(f'<div class="metric-card"><div class="metric-title">Numeric Columns</div><div class="metric-value">{len(num_cols)}</div></div>', unsafe_allow_html=True)
            m4.markdown(f'<div class="metric-card"><div class="metric-title">Categorical Columns</div><div class="metric-value">{len(cat_cols)}</div></div>', unsafe_allow_html=True)

            st.markdown("---")
            st.dataframe(df.head(10), use_container_width=True)

        # Tabs 2 to 5: 4 Visual Dashboards
        for idx, (dash_name, chart_list) in enumerate(dashboards.items(), start=1):
            with tabs[idx]:
                st.subheader(f"📊 {dash_name}")
                row1_col1, row1_col2 = st.columns(2)
                with row1_col1:
                    st.plotly_chart(chart_list[0][1], use_container_width=True)
                with row1_col2:
                    st.plotly_chart(chart_list[1][1], use_container_width=True)

                row2_col1, row2_col2 = st.columns(2)
                with row2_col1:
                    st.plotly_chart(chart_list[2][1], use_container_width=True)
                with row2_col2:
                    st.plotly_chart(chart_list[3][1], use_container_width=True)

        # Tab 6: CUSTOM CHART STUDIO BUILDER
        with tabs[5]:
            st.subheader("🎨 Custom Chart Studio")
            st.write("Configure dynamic interactive visualizations tailored to your dataset.")
            
            c_col1, c_col2, c_col3 = st.columns(3)
            with c_col1:
                chart_type = st.selectbox("Select Chart Type:", ["Bar Chart", "Line Chart", "Scatter Plot", "Histogram", "Box Plot", "Pie / Donut Chart"])
                x_col = st.selectbox("Select X-Axis Field:", df.columns, index=0)
            with c_col2:
                y_col = st.selectbox("Select Y-Axis Field (Optional for Histograms):", [None] + list(df.columns), index=1 if len(df.columns) > 1 else 0)
                color_col = st.selectbox("Select Color/Group Field (Optional):", [None] + list(df.columns))
            with c_col3:
                color_theme = st.selectbox("Select Color Palette:", ["Viridis", "Plasma", "Turbo", "Spectral", "Bluered"])
                custom_title = st.text_input("Chart Title:", value=f"Custom {chart_type}: {x_col}" + (f" vs {y_col}" if y_col else ""))

            # Build Custom Plotly Visualization
            try:
                if chart_type == "Bar Chart":
                    if y_col:
                        agg_df = df.groupby(x_col)[y_col].sum().reset_index()
                        custom_fig = px.bar(agg_df, x=x_col, y=y_col, color=color_col if color_col else x_col, title=custom_title, color_continuous_scale=color_theme.lower())
                    else:
                        custom_fig = px.bar(df[x_col].value_counts().reset_index(), x=x_col, y='count', color=x_col, title=custom_title)
                
                elif chart_type == "Line Chart":
                    custom_fig = px.line(df, x=x_col, y=y_col, color=color_col, title=custom_title)
                
                elif chart_type == "Scatter Plot":
                    custom_fig = px.scatter(df, x=x_col, y=y_col, color=color_col, title=custom_title)
                
                elif chart_type == "Histogram":
                    custom_fig = px.histogram(df, x=x_col, color=color_col, title=custom_title)
                
                elif chart_type == "Box Plot":
                    custom_fig = px.box(df, x=x_col if y_col else None, y=y_col if y_col else x_col, color=color_col, title=custom_title)
                
                elif chart_type == "Pie / Donut Chart":
                    if y_col:
                        custom_fig = px.pie(df, names=x_col, values=y_col, title=custom_title, hole=0.4)
                    else:
                        custom_fig = px.pie(df, names=x_col, title=custom_title, hole=0.4)

                custom_fig.update_layout(template="plotly_white", height=500)
                st.plotly_chart(custom_fig, use_container_width=True)

            except Exception as e:
                st.error(f"Unable to render custom visualization with selected options: {e}")

        # Tab 7: EXECUTIVE BUSINESS Q&A ENGINE
        with tabs[6]:
            st.subheader("💼 Executive Business Q&A Engine")
            st.write("Extract revenue drivers, operational benchmarks, and risk metrics using business-focused queries.")

            # Automated Executive Summary Insights
            st.markdown("### 📊 Automated Executive Insights")
            if cat_cols and num_cols:
                top_group = df.groupby(cat_cols[0])[num_cols[0]].sum().idxmax()
                top_val = df.groupby(cat_cols[0])[num_cols[0]].sum().max()
                
                st.markdown(f"""
                <div class="biz-card">
                    <div class="biz-title">🎯 Primary Business Driver</div>
                    <p style="margin:0; color:#334155;">
                        The leading performer in <b>{cat_cols[0]}</b> is <b>{top_group}</b> with <b>{top_val:,.2f}</b> in cumulative <b>{num_cols[0]}</b>.
                    </p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("### 💬 Ask a Business Question")
            
            preset_q = st.selectbox("Select or Type a Business Query:", [
                "Custom Question",
                "What is our top performing category?",
                "Which category represents our highest risk / lowest performance?",
                "What is the average metric benchmark?",
                "What is the total overall volume?"
            ])

            if preset_q == "Custom Question":
                biz_user_q = st.text_input("Enter your business query (e.g., 'What is the highest performing segment?'):")
            else:
                biz_user_q = preset_q

            if biz_user_q:
                ans = answer_business_question(df, biz_user_q)
                st.info(ans)

        # Tab 8: STATISTICAL ANALYTICAL QA ENGINE
        with tabs[7]:
            st.subheader("❓ Statistical Analytical QA Engine")
            st.write("Explore automated statistical hypotheses or query columns directly.")
            
            st.markdown("### Pre-Generated Data Hypotheses")
            for q in questions_list:
                st.markdown(f"""
                <div class="q-card">
                    <div class="q-title">[{q['category']}] {q['question']}</div>
                    <p style="margin-top: 5px; color: #475569;"><b>Purpose:</b> {q['purpose']}</p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("### 🔍 Ask a Data Question")
            user_question = st.text_input("Enter a specific question about dataset variables:")
            
            if user_question:
                matched_cols = [c for c in df.columns if c.lower() in user_question.lower()]
                if matched_cols and num_cols:
                    st.success(f"🔍 Analyzing columns: **{', '.join(matched_cols)}**")
                    st.dataframe(df.groupby(matched_cols[0])[num_cols[0]].describe(), use_container_width=True)
                else:
                    st.dataframe(df.describe().T, use_container_width=True)

        # Tab 9: FEATURE IMPORTANCE
        with tabs[8]:
            st.subheader(f"Predictive Driver Analysis for Target '{target_field}'")
            try:
                imp_df = train_risk_model(df, target_field)
                fig_imp = px.bar(imp_df.head(10), x='Importance', y='Feature', orientation='h', title="Top Driver Variables")
                fig_imp.update_layout(yaxis={'categoryorder': 'total ascending'}, height=420, template="plotly_white")
                st.plotly_chart(fig_imp, use_container_width=True)
            except Exception as e:
                st.error(f"Could not build predictive model: {e}")

        # PDF Report Download Trigger
        st.sidebar.markdown("---")
        if st.sidebar.button("📄 Generate PDF Executive Report"):
            with st.spinner("Creating PDF report with chart snapshots..."):
                pdf_bytes = generate_pdf_report(quality_info, dashboards)
                st.sidebar.download_button(
                    label="⬇️ Download PDF Report",
                    data=pdf_bytes,
                    file_name="Executive_Data_Report.pdf",
                    mime="application/pdf"
                )

else:
    st.info("👈 Upload a CSV or Excel file in the sidebar to activate all 4 dashboards, custom chart studio, and Q&A engines.")