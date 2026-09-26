"""
app.py - Data Analyzer AI Platform featuring 4 Visual Dashboards 
(with Integrated Custom Chart Builder inside Dashboard 1) and Interactive Q&A Engine.
"""

import io
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

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

# Custom Styling
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
    .custom-box {
        background-color: #ffffff;
        border: 1px solid #cbd5e1;
        border-radius: 10px;
        padding: 20px;
        margin-top: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
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

# Dataset Caching
@st.cache_data
def load_uploaded_file(file):
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file, engine='openpyxl')
        df.columns = df.columns.astype(str).str.strip()
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

# Profiler & QA Functions
def analyze_data_quality(df):
    return {
        "total_rows": len(df),
        "total_cols": len(df.columns),
        "total_missing": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum())
    }

def generate_analytical_questions(df):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
    
    questions = []
    if cat_cols and num_cols:
        questions.append({
            "category": "Segment Breakdown",
            "question": f"Which {cat_cols[0]} category yields the highest sum in {num_cols[0]}?",
            "purpose": f"Identifies volume leaders across {cat_cols[0]}."
        })
    if len(num_cols) >= 2:
        questions.append({
            "category": "Correlation Test",
            "question": f"How strong is the correlation between {num_cols[0]} and {num_cols[1]}?",
            "purpose": "Evaluates relational dependencies across numeric variables."
        })
    if num_cols:
        questions.append({
            "category": "Outlier Detection",
            "question": f"Are there extreme skewness or outliers in {num_cols[0]}?",
            "purpose": "Spotlights numerical distribution anomalies."
        })
    return questions

def answer_business_question(df, question_text):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
    q_lower = question_text.lower()
    
    if any(k in q_lower for k in ['top', 'best', 'highest', 'max', 'leading']):
        if cat_cols and num_cols:
            top_res = df.groupby(cat_cols[0])[num_cols[0]].sum().sort_values(ascending=False).head(5)
            return (f"🏆 **Top Performer Analysis:**\n\n"
                    f"The top-ranked **{cat_cols[0]}** is **'{top_res.index[0]}'** with total **{num_cols[0]}** of **{top_res.iloc[0]:,.2f}**.\n\n"
                    f"**Top 5 Leaderboard:**\n" + "\n".join([f"- **{k}**: {v:,.2f}" for k, v in top_res.items()]))
    
    elif any(k in q_lower for k in ['lowest', 'bottom', 'worst', 'min', 'risk']):
        if cat_cols and num_cols:
            low_res = df.groupby(cat_cols[0])[num_cols[0]].sum().sort_values(ascending=True).head(5)
            return (f"⚠️ **Lowest Performance Breakdown:**\n\n"
                    f"The lowest-ranked **{cat_cols[0]}** is **'{low_res.index[0]}'** with total **{num_cols[0]}** of **{low_res.iloc[0]:,.2f}**.\n\n"
                    f"**Bottom 5 Segments:**\n" + "\n".join([f"- **{k}**: {v:,.2f}" for k, v in low_res.items()]))

    elif any(k in q_lower for k in ['average', 'mean', 'avg']):
        if num_cols:
            return f"📊 **Average Benchmark:** Mean {num_cols[0]} is **{df[num_cols[0]].mean():,.2f}** (Median: {df[num_cols[0]].median():,.2f})."

    elif any(k in q_lower for k in ['total', 'sum', 'overall', 'volume']):
        if num_cols:
            return f"💰 **Total Aggregated Sum:** Cumulative {num_cols[0]} totals **{df[num_cols[0]].sum():,.2f}**."

    return "📈 **General Dataset Summary:** Select specific metric columns to evaluate categorical breakdowns."

# Robust ML Feature Importance Model
def train_risk_model(df, target_col):
    data = df.copy().dropna()
    
    # Drop identifier columns or high cardinality features
    for col in list(data.columns):
        if col != target_col and (col.lower().endswith('id') or data[col].nunique() > 100):
            data = data.drop(columns=[col])

    if data.empty or target_col not in data.columns or len(data.columns) <= 1:
        return pd.DataFrame({'Feature': ['Insufficient Data'], 'Importance': [0.0]})

    # Encode all non-numeric features
    X = data.drop(columns=[target_col])
    for col in X.columns:
        if not np.issubdtype(X[col].dtype, np.number):
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    y = data[target_col]
    
    # Target Type Check
    if np.issubdtype(y.dtype, np.number) and y.nunique() > 20:
        model = RandomForestRegressor(n_estimators=50, random_state=42)
    else:
        if not np.issubdtype(y.dtype, np.number):
            y = LabelEncoder().fit_transform(y.astype(str))
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        
    model.fit(X, y)
    return pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_}).sort_values(by='Importance', ascending=False)

# PDF Generation
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
        Paragraph("1. Dataset Summary", heading_style)
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

    story.append(Paragraph("2. Visual Dashboards", heading_style))
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

# Sidebar Navigation
st.sidebar.title("🧠 Data Analyzer AI")
uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel File", type=["csv", "xlsx"])

if uploaded_file is not None:
    with st.spinner("Processing data..."):
        df = load_uploaded_file(uploaded_file)

    if df is not None:
        target_field = st.sidebar.selectbox("Select Target Variable:", df.columns)
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
        quality_info = analyze_data_quality(df)
        questions_list = generate_analytical_questions(df)

        # Build Standard Dashboards
        dashboards = {}

        # Dashboard 1 Base Charts
        c1 = cat_cols[0] if cat_cols else df.columns[0]
        n1 = num_cols[0] if num_cols else df.columns[0]
        n2 = num_cols[1] if len(num_cols) > 1 else n1

        d1 = [
            ("Top Categorical Aggregation", px.bar(df.groupby(c1)[n1].sum().reset_index().head(10), x=c1, y=n1, color=c1, title=f"Top {c1} by {n1}")),
            ("Feature Correlation Scatter", px.scatter(df, x=n1, y=n2, title=f"Scatter: {n1} vs {n2}")),
            ("Proportion Share Donut Chart", px.pie(df, names=c1, title=f"Share: {c1}", hole=0.4))
        ]
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
        d4 = [
            ("Bivariate Feature Map", px.scatter(df, x=n1, y=n2, color=c1 if cat_cols else None, title="Multivariate Scatter Map")),
            ("Category Variance Boxplot", px.box(df, x=c1, y=n1, title=f"{n1} Variance Across {c1}")),
            ("Feature Density Breakdown", px.histogram(df, x=n1, color=c1 if cat_cols else None, title="Stacked Density Plot")),
            ("Target Column Focus", px.scatter(df, x=n1, y=target_field if target_field in num_cols else n1, title=f"Impact on Target ({target_field})"))
        ]
        dashboards["Dashboard 4: Advanced Relationships"] = d4

        # Main Layout Navigation
        st.title("Data Analyzer AI Platform")
        
        tabs = st.tabs([
            "📋 Dataset Overview", 
            "📊 Dashboard 1 (With Custom Chart)", 
            "📊 Dashboard 2", 
            "📊 Dashboard 3", 
            "📊 Dashboard 4", 
            "💬 Interactive Q&A Engine", 
            "🤖 Feature Importance"
        ])

        # Tab 1: Dataset Overview
        with tabs[0]:
            st.subheader("Dataset Overview Metrics")
            m1, m2, m3, m4 = st.columns(4)
            m1.markdown(f'<div class="metric-card"><div class="metric-title">Total Records</div><div class="metric-value">{quality_info["total_rows"]:,}</div></div>', unsafe_allow_html=True)
            m2.markdown(f'<div class="metric-card"><div class="metric-title">Total Columns</div><div class="metric-value">{quality_info["total_cols"]:,}</div></div>', unsafe_allow_html=True)
            m3.markdown(f'<div class="metric-card"><div class="metric-title">Numeric Columns</div><div class="metric-value">{len(num_cols)}</div></div>', unsafe_allow_html=True)
            m4.markdown(f'<div class="metric-card"><div class="metric-title">Categorical Columns</div><div class="metric-value">{len(cat_cols)}</div></div>', unsafe_allow_html=True)

            st.markdown("---")
            st.dataframe(df.head(10), width='stretch')

        # Tab 2: DASHBOARD 1 (INTEGRATED WITH CUSTOM CHART BUILDER)
        with tabs[1]:
            st.subheader("📊 Dashboard 1: Executive Overview & Custom Chart Studio")
            
            # Row 1: Standard Executive Visuals
            row1_col1, row1_col2 = st.columns(2)
            with row1_col1:
                st.plotly_chart(d1[0][1], width='stretch')
            with row1_col2:
                st.plotly_chart(d1[1][1], width='stretch')

            row2_col1, row2_col2 = st.columns(2)
            with row2_col1:
                st.plotly_chart(d1[2][1], width='stretch')

            # INTEGRATED CUSTOMIZABLE CHART PANEL INSIDE DASHBOARD 1
            with row2_col2:
                st.markdown("### 🎨 Integrated Custom Chart Builder")
                
                c_col1, c_col2 = st.columns(2)
                with c_col1:
                    chart_type = st.selectbox("Chart Type:", ["Bar Chart", "Line Chart", "Scatter Plot", "Histogram", "Box Plot", "Pie Chart"], key="d1_type")
                    x_col = st.selectbox("X-Axis Field:", df.columns, index=0, key="d1_x")
                with c_col2:
                    y_col = st.selectbox("Y-Axis Field:", [None] + list(df.columns), index=1 if len(df.columns) > 1 else 0, key="d1_y")
                    color_col = st.selectbox("Color / Group Field:", [None] + list(df.columns), key="d1_color")

                custom_title = st.text_input("Chart Title:", value=f"Customized {chart_type}: {x_col}", key="d1_title")

                # Generate Dynamic Customized Visualization
                try:
                    if chart_type == "Bar Chart":
                        if y_col:
                            agg_df = df.groupby(x_col)[y_col].sum().reset_index()
                            custom_fig = px.bar(agg_df, x=x_col, y=y_col, color=color_col if color_col else x_col, title=custom_title)
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
                    elif chart_type == "Pie Chart":
                        if y_col:
                            custom_fig = px.pie(df, names=x_col, values=y_col, title=custom_title, hole=0.3)
                        else:
                            custom_fig = px.pie(df, names=x_col, title=custom_title, hole=0.3)

                    custom_fig.update_layout(template="plotly_white", height=380)
                    st.plotly_chart(custom_fig, width='stretch')
                except Exception as e:
                    st.error(f"Error building custom chart: {e}")

        # Tabs 3, 4, 5: Dashboards 2, 3, and 4
        for idx, (dash_name, chart_list) in enumerate(list(dashboards.items())[1:], start=2):
            with tabs[idx]:
                st.subheader(f"📊 {dash_name}")
                row1_col1, row1_col2 = st.columns(2)
                with row1_col1:
                    st.plotly_chart(chart_list[0][1], width='stretch')
                with row1_col2:
                    st.plotly_chart(chart_list[1][1], width='stretch')

                row2_col1, row2_col2 = st.columns(2)
                with row2_col1:
                    st.plotly_chart(chart_list[2][1], width='stretch')
                with row2_col2:
                    st.plotly_chart(chart_list[3][1], width='stretch')

        # Tab 6: COMBINED INTERACTIVE Q&A ENGINE (BUSINESS + ANALYTICAL QA)
        with tabs[5]:
            st.subheader("💬 Interactive Data & Executive Q&A Engine")
            st.write("Ask operational questions or inspect automated data hypotheses.")

            qa_tab1, qa_tab2 = st.tabs(["💼 Business Q&A", "❓ Analytical & Statistical QA"])

            with qa_tab1:
                st.markdown("### Executive Business Query Assistant")
                preset_q = st.selectbox("Select a Preset Business Query:", [
                    "Custom Question",
                    "What is our top performing category?",
                    "Which category represents our highest risk / lowest performance?",
                    "What is the average metric benchmark?",
                    "What is the total overall volume?"
                ])

                if preset_q == "Custom Question":
                    biz_user_q = st.text_input("Type your business query (e.g., 'What is the top category?'):")
                else:
                    biz_user_q = preset_q

                if biz_user_q:
                    ans = answer_business_question(df, biz_user_q)
                    st.info(ans)

            with qa_tab2:
                st.markdown("### Pre-Generated Data Hypotheses")
                for q in questions_list:
                    st.markdown(f"""
                    <div class="q-card">
                        <div class="q-title">[{q['category']}] {q['question']}</div>
                        <p style="margin-top: 5px; color: #475569;"><b>Purpose:</b> {q['purpose']}</p>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("---")
                st.markdown("### 🔍 Specific Column Inspector Question")
                user_question = st.text_input("Ask about specific dataset columns or variables:")
                
                if user_question:
                    matched_cols = [c for c in df.columns if c.lower() in user_question.lower()]
                    if matched_cols and num_cols:
                        st.success(f"🔍 Analyzing columns: **{', '.join(matched_cols)}**")
                        st.dataframe(df.groupby(matched_cols[0])[num_cols[0]].describe(), width='stretch')
                    else:
                        st.dataframe(df.describe().T, width='stretch')

        # Tab 7: FEATURE IMPORTANCE
        with tabs[6]:
            st.subheader(f"Predictive Feature Drivers for Target Variable '{target_field}'")
            try:
                imp_df = train_risk_model(df, target_field)
                fig_imp = px.bar(imp_df.head(10), x='Importance', y='Feature', orientation='h', title="Top Drivers Importance")
                fig_imp.update_layout(yaxis={'categoryorder': 'total ascending'}, height=420, template="plotly_white")
                st.plotly_chart(fig_imp, width='stretch')
            except Exception as e:
                st.error(f"Could not calculate predictive drivers: {e}")

        # PDF Executive Report Download
        st.sidebar.markdown("---")
        if st.sidebar.button("📄 Export PDF Executive Report"):
            with st.spinner("Compiling PDF Executive Report with visual summaries..."):
                pdf_bytes = generate_pdf_report(quality_info, dashboards)
                st.sidebar.download_button(
                    label="⬇️ Download PDF Report",
                    data=pdf_bytes,
                    file_name="Executive_Data_Report.pdf",
                    mime="application/pdf"
                )

else:
    st.info("👈 Upload a CSV or Excel file using the sidebar to render the dashboards and integrated custom chart controls.")