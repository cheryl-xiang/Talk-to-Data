import streamlit as st
import duckdb
import pandas as pd
import plotly.express as px
import anthropic
import json
import re
import os

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Talk2Data · Olist BI",
    page_icon="📊",
    layout="wide",
)

# ── CSV sources ───────────────────────────────────────────────────────────────
# CSVs are loaded directly from the official Olist Kaggle mirror on GitHub
BASE_URL = "https://raw.githubusercontent.com/nicholasgasior/datascienceportfolio/main/olist"

# Fallback: well-known Kaggle mirror
KAGGLE_BASE = "https://raw.githubusercontent.com/dsrscientist/dataset1/master"

CSV_TABLES = {
    "orders":           "https://raw.githubusercontent.com/olist/work-at-olist-data/master/datasets/olist_orders_dataset.csv",
    "order_items":      "https://raw.githubusercontent.com/olist/work-at-olist-data/master/datasets/olist_order_items_dataset.csv",
    "order_payments":   "https://raw.githubusercontent.com/olist/work-at-olist-data/master/datasets/olist_order_payments_dataset.csv",
    "order_reviews":    "https://raw.githubusercontent.com/olist/work-at-olist-data/master/datasets/olist_order_reviews_dataset.csv",
    "customers":        "https://raw.githubusercontent.com/olist/work-at-olist-data/master/datasets/olist_customers_dataset.csv",
    "sellers":          "https://raw.githubusercontent.com/olist/work-at-olist-data/master/datasets/olist_sellers_dataset.csv",
    "products":         "https://raw.githubusercontent.com/olist/work-at-olist-data/master/datasets/olist_products_dataset.csv",
    "product_category_name_translation": "https://raw.githubusercontent.com/olist/work-at-olist-data/master/datasets/product_category_name_translation.csv",
}

SCHEMA_DESCRIPTION = """
You have access to an SQLite database with the following tables:

- orders (order_id, customer_id, order_status, order_purchase_timestamp, order_delivered_customer_date, order_estimated_delivery_date)
- order_items (order_id, order_item_id, product_id, seller_id, shipping_limit_date, price, freight_value)
- order_payments (order_id, payment_sequential, payment_type, payment_installments, payment_value)
- order_reviews (review_id, order_id, review_score, review_comment_title, review_comment_message, review_creation_date)
- customers (customer_id, customer_unique_id, customer_zip_code_prefix, customer_city, customer_state)
- sellers (seller_id, seller_zip_code_prefix, seller_city, seller_state)
- products (product_id, product_category_name, product_weight_g, product_length_cm, product_height_cm, product_width_cm)
- product_category_name_translation (product_category_name, product_category_name_english)
- geolocation (geolocation_zip_code_prefix, geolocation_lat, geolocation_lng, geolocation_city, geolocation_state)

Key relationships:
- orders → order_items via order_id
- orders → customers via customer_id
- order_items → products via product_id
- order_items → sellers via seller_id
- orders → order_payments via order_id
- orders → order_reviews via order_id
- products → product_category_name_translation via product_category_name
"""

TEXT_TO_SQL_SYSTEM = f"""You are a data analyst assistant. Given a natural language question,
generate a single valid DuckDB SQL query to answer it.

{SCHEMA_DESCRIPTION}

Rules:
- Return ONLY a JSON object with two keys: "sql" and "explanation"
- "sql": the SQL query string
- "explanation": a one-sentence plain English description of what the query does
- Use product_category_name_english for readable category names (join with product_category_name_translation)
- Limit results to 50 rows unless the user asks for more
- For date operations use strftime()
- Do not include markdown or code fences in your response
"""

INSIGHT_SYSTEM = """You are a senior business analyst. Given a summary of key metrics from 
an e-commerce dataset, write 3-4 sentences of sharp, actionable business insights. 
Be specific about numbers. Use plain English. No bullet points.
Do NOT use markdown formatting, bold, italics, or dollar signs ($). Write currency as R$ only within words, 
e.g. R$15 million. Output plain prose only."""

# ── Load CSVs into in-memory DuckDB ──────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_connection():
    conn = duckdb.connect()  # in-memory
    failed = []
    for table_name, url in CSV_TABLES.items():
        try:
            df = pd.read_csv(url)
            conn.register(table_name, df)
        except Exception as e:
            failed.append(f"{table_name}: {e}")
    if failed:
        st.warning("Some tables failed to load:\n" + "\n".join(failed))
    return conn

# ── Run SQL safely ────────────────────────────────────────────────────────────
def run_query(sql: str) -> pd.DataFrame:
    conn = get_connection()
    try:
        return conn.execute(sql).df()
    except Exception as e:
        st.error(f"Query error: {e}")
        return pd.DataFrame()

# ── KPI helper ────────────────────────────────────────────────────────────────
def metric_card(label, value, delta=None):
    st.metric(label=label, value=value, delta=delta)

# ── LLM helpers ───────────────────────────────────────────────────────────────
@st.cache_resource
def _get_client():
    api_key = st.secrets.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        st.error("Anthropic API key not found. Add it to `.streamlit/secrets.toml` or set the `ANTHROPIC_API_KEY` environment variable.")
        st.stop()
    return anthropic.Anthropic(api_key=api_key)

@st.cache_data(show_spinner=False)
def get_insights(summary_text: str) -> str:
    client = _get_client()
    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=300,
        system=INSIGHT_SYSTEM,
        messages=[{"role": "user", "content": summary_text}],
    )
    return msg.content[0].text

def _call_llm(messages: list, schema_override: str = None) -> dict:
    """Call the LLM and parse the JSON response."""
    client = _get_client()
    system = TEXT_TO_SQL_SYSTEM
    if schema_override:
        system = f"""You are a data analyst assistant. Given a natural language question,
generate a single valid DuckDB SQL query to answer it.
 
The user has uploaded a CSV file. It is available as a table called \"uploaded_table\" with these columns:
{schema_override}
 
Rules:
- Return ONLY a JSON object with two keys: "sql" and "explanation"
- "sql": the SQL query string  
- "explanation": a one-sentence plain English description of what the query does
- Limit results to 50 rows unless the user asks for more
- ALWAYS cast timestamp/date columns explicitly: CAST(col AS TIMESTAMP)
- Do not include markdown or code fences in your response
"""
    msg = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=500,
        system=system,
        messages=messages,
    )
    raw = msg.content[0].text.strip()
    raw = re.sub(r"```json|```", "", raw).strip()
    return json.loads(raw)

def text_to_sql(question: str, schema_override: str = None) -> dict:
    """Generate SQL from a natural language question, with one auto-retry on error."""
    messages = [{"role": "user", "content": question}]
    result = _call_llm(messages, schema_override=schema_override)
    sql = result.get("sql", "")
 
    # Try running the SQL — if it fails, send the error back to the LLM for a fix
    try:
        conn = get_connection()
        conn.execute(sql)  # dry run to check for errors
    except Exception as e:
        error_msg = str(e)
        # Add the failed attempt + error to the conversation and retry once
        messages += [
            {"role": "assistant", "content": f'''```json\n{json.dumps(result)}\n```'''},
            {"role": "user", "content": (
                f"That SQL failed with this error:\n\n{error_msg}\n\n"
                "Please fix the SQL and return a corrected JSON response. "
                "Remember to CAST any timestamp/date columns explicitly, e.g. CAST(col AS TIMESTAMP)."
            )},
        ]
        result = _call_llm(messages, schema_override=schema_override)
 
    return result

# ── Load KPI data ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_kpis():
    kpis = {}
    kpis["total_orders"] = run_query("SELECT COUNT(DISTINCT order_id) AS n FROM orders").iloc[0, 0]
    kpis["total_revenue"] = run_query("SELECT ROUND(SUM(payment_value),2) AS r FROM order_payments").iloc[0, 0]
    kpis["avg_order_value"] = run_query("""
        SELECT ROUND(AVG(order_total),2) FROM (
            SELECT order_id, SUM(payment_value) AS order_total FROM order_payments GROUP BY order_id
        )""").iloc[0, 0]
    kpis["avg_review"] = run_query("SELECT ROUND(AVG(review_score),2) FROM order_reviews").iloc[0, 0]
    kpis["total_customers"] = run_query("SELECT COUNT(DISTINCT customer_unique_id) FROM customers").iloc[0, 0]
    kpis["total_sellers"] = run_query("SELECT COUNT(DISTINCT seller_id) FROM sellers").iloc[0, 0]
    return kpis

@st.cache_data(show_spinner=False)
def load_charts():
    charts = {}

    # Revenue over time (monthly)
    charts["revenue_over_time"] = run_query("""
        SELECT strftime(CAST(o.order_purchase_timestamp AS TIMESTAMP), '%Y-%m') AS month,
               ROUND(SUM(p.payment_value), 2) AS revenue
        FROM orders o JOIN order_payments p ON o.order_id = p.order_id
        WHERE o.order_purchase_timestamp IS NOT NULL
        GROUP BY month ORDER BY month
    """)

    # Top 10 categories by revenue
    charts["top_categories"] = run_query("""
        SELECT COALESCE(t.product_category_name_english, pr.product_category_name) AS category,
               ROUND(SUM(oi.price), 2) AS revenue
        FROM order_items oi
        JOIN products pr ON oi.product_id = pr.product_id
        LEFT JOIN product_category_name_translation t ON pr.product_category_name = t.product_category_name
        GROUP BY category ORDER BY revenue DESC LIMIT 10
    """)

    # Orders by status
    charts["order_status"] = run_query("""
        SELECT order_status, COUNT(*) AS count FROM orders GROUP BY order_status ORDER BY count DESC
    """)

    # Revenue by state
    charts["revenue_by_state"] = run_query("""
        SELECT c.customer_state AS state, ROUND(SUM(p.payment_value),2) AS revenue
        FROM orders o
        JOIN customers c ON o.customer_id = c.customer_id
        JOIN order_payments p ON o.order_id = p.order_id
        GROUP BY state ORDER BY revenue DESC LIMIT 15
    """)

    # Review score distribution
    charts["review_dist"] = run_query("""
        SELECT review_score, COUNT(*) AS count FROM order_reviews GROUP BY review_score ORDER BY review_score
    """)

    return charts

# ── Auto-generated dashboard for uploaded CSVs ────────────────────────────────
def generate_uploaded_dashboard(df: pd.DataFrame, filename: str):
    """Use Claude to analyze an uploaded CSV and generate a dynamic dashboard."""
    client = _get_client()
 
    # Sample the data for the prompt
    sample = df.head(5).to_csv(index=False)
    col_info = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        n_unique = df[col].nunique()
        col_info.append(f"  - {col} ({dtype}, {n_unique} unique values)")
    col_summary = "\n".join(col_info)
 
    prompt = f"""You are a data analyst. A user uploaded a CSV called "{filename}".
Here are the columns:
{col_summary}
 
Sample rows:
{sample}
 
Return a JSON object with exactly these keys:
- "kpis": list of up to 4 objects, each with:
    - "label": display name
    - "sql": SQL query against table "uploaded_table" returning a single value
- "charts": list of up to 4 objects, each with:
    - "title": chart title
    - "type": one of "bar", "line", "pie"
    - "sql": SQL query against "uploaded_table" returning exactly 2 columns: a label column and a numeric column
- "insights_summary": a plain text paragraph (no markdown) summarising the dataset in 2 sentences for an AI insights prompt
 
Rules:
- Only use columns that exist in the table
- CAST any date/timestamp columns explicitly e.g. CAST(col AS TIMESTAMP)
- SQL must be valid DuckDB SQL
- Return ONLY the raw JSON, no markdown, no code fences
"""
    msg = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = msg.content[0].text.strip()
    raw = re.sub(r"```json|```", "", raw).strip()
    return json.loads(raw)

# ── Auto-generated example questions for uploaded CSVs ────────────────────────────────
@st.cache_data(show_spinner=False)
def generate_example_questions(columns: tuple, filename: str) -> list:
    """Ask Claude to suggest 5 example questions for an uploaded CSV."""
    client = _get_client()
    col_list = ", ".join(columns)
    msg = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=300,
        messages=[{"role": "user", "content": (
            f"A user uploaded a CSV called \"{filename}\" with these columns: {col_list}.\n\n"
            "Suggest exactly 5 short, specific natural language questions they could ask about this data. "
            "Each question should be answerable with a simple SQL query. "
            "Return ONLY a JSON array of 5 strings, no markdown, no explanation."
        )}],
    )
    raw = msg.content[0].text.strip()
    raw = re.sub(r"```json|```", "", raw).strip()
    return json.loads(raw)

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("📊 TalkToData")
st.caption("AI-powered business intelligence for the Olist Brazilian E-Commerce dataset")

conn = get_connection()
if conn is None:
    st.warning("""
    **Database not found.** 
    
    1. Download the SQLite version of the Olist dataset from Kaggle:  
       https://www.kaggle.com/datasets/terencicp/e-commerce-dataset-by-olist-as-an-sqlite-database
    2. Rename the file to `olist.db` and place it in the same folder as `app.py`
    3. Restart the app
    """)
    st.stop()

# ── CSV Upload (sidebar) ──────────────────────────────────────────────────────
with st.sidebar:
    st.header("📂 Upload your own CSV")
    st.caption("Upload a CSV to query it with natural language. This replaces the Olist dataset in the Ask the Data tab.")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv", key="csv_upload")
 
    if uploaded_file is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_file)
            # Store df in session_state so it survives reruns
            new_file = uploaded_file.name != st.session_state.get("uploaded_filename")
            st.session_state["uploaded_df"] = uploaded_df
            st.session_state["uploaded_table_name"] = "uploaded_table"
            st.session_state["uploaded_columns"] = list(uploaded_df.columns)
            st.session_state["uploaded_filename"] = uploaded_file.name
            if new_file:
                st.session_state["chat_history"] = []
                st.session_state.pop("_dashboard_cache_key", None)
            st.success(f"✅ Loaded **{uploaded_file.name}**")
            st.caption(f"{len(uploaded_df):,} rows · {len(uploaded_df.columns)} columns")
            with st.expander("Preview"):
                st.dataframe(uploaded_df.head(5), use_container_width=True)
        except Exception as e:
            st.error(f"Failed to load file: {e}")
 
    if st.session_state.get("uploaded_table_name"):
        if st.button("🗑️ Remove uploaded file", use_container_width=True):
            st.session_state.pop("uploaded_table_name", None)
            st.session_state.pop("uploaded_columns", None)
            st.session_state.pop("uploaded_filename", None)
            st.session_state.pop("uploaded_df", None)
            st.session_state.pop("_dashboard_cache_key", None)
            st.session_state.pop("_dashboard_spec", None)
            st.session_state["chat_history"] = []
            st.rerun()
 
    st.divider()
    st.caption("Using Olist dataset by default — 100K Brazilian e-commerce orders (2016–2018)")
 
# ── Dynamic caption based on active dataset ───────────────────────────────────
if st.session_state.get("uploaded_table_name"):
    st.caption(f"Querying: **{st.session_state['uploaded_filename']}**")
else:
    st.caption("AI-powered business intelligence · Olist Brazilian E-Commerce dataset")
 
# Re-register uploaded table on every rerun (survives Streamlit reruns)
if st.session_state.get("uploaded_df") is not None:
    try:
        get_connection().register("uploaded_table", st.session_state["uploaded_df"])
    except Exception:
        pass
 
# Trigger data load with a spinner
with st.spinner("Loading dataset... (first load may take ~30 seconds)"):
    get_connection()


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📈 Dashboard", "💬 Ask the Data", "🗂️ View Data"])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — Dashboard
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    # ── Uploaded CSV dashboard ────────────────────────────────────────────────
    if st.session_state.get("uploaded_table_name"):
        uploaded_df = get_connection().execute("SELECT * FROM uploaded_table LIMIT 1000").df()
        filename = st.session_state.get("uploaded_filename", "uploaded file")
 
        cache_key = f"dashboard_{filename}_{len(uploaded_df)}"
        if st.session_state.get("_dashboard_cache_key") != cache_key:
            with st.spinner("Analyzing your data and building dashboard..."):
                try:
                    dashboard_spec = generate_uploaded_dashboard(uploaded_df, filename)
                    st.session_state["_dashboard_spec"] = dashboard_spec
                    st.session_state["_dashboard_cache_key"] = cache_key
                except Exception as e:
                    st.error(f"Could not generate dashboard: {e}")
                    dashboard_spec = None
        else:
            dashboard_spec = st.session_state.get("_dashboard_spec")
 
        if dashboard_spec:
            # KPIs
            kpi_list = dashboard_spec.get("kpis", [])
            if kpi_list:
                kpi_cols = st.columns(len(kpi_list))
                for i, kpi in enumerate(kpi_list):
                    try:
                        val = run_query(kpi["sql"]).iloc[0, 0]
                        if isinstance(val, float):
                            val = f"{val:,.2f}"
                        elif isinstance(val, int):
                            val = f"{val:,}"
                        kpi_cols[i].metric(kpi["label"], val)
                    except Exception:
                        kpi_cols[i].metric(kpi["label"], "—")
 
            st.divider()
 
            # AI Insights
            with st.expander("🤖 AI Insights", expanded=True):
                summary = dashboard_spec.get("insights_summary", f"Dataset: {filename}")
                with st.spinner("Generating insights..."):
                    insights = get_insights(summary)
                insights_clean = re.sub(r"\*{1,2}([^*]+)\*{1,2}", r"", insights)
                insights_clean = insights_clean.replace("$", r"\$")
                st.info(insights_clean)
 
            st.divider()
 
            # Charts
            chart_list = dashboard_spec.get("charts", [])
            if chart_list:
                pairs = [chart_list[i:i+2] for i in range(0, len(chart_list), 2)]
                colors = ["#635BFF", "#00C9A7", "#FF6B6B", "#FFA94D"]
                for pair in pairs:
                    cols = st.columns(len(pair))
                    for i, chart_spec in enumerate(pair):
                        with cols[i]:
                            st.subheader(chart_spec["title"])
                            try:
                                df_chart = run_query(chart_spec["sql"])
                                if df_chart.empty:
                                    st.caption("No data")
                                    continue
                                xcol, ycol = df_chart.columns[0], df_chart.columns[1]
                                ctype = chart_spec.get("type", "bar")
                                color = colors[i % len(colors)]
                                if ctype == "line":
                                    fig = px.line(df_chart, x=xcol, y=ycol, color_discrete_sequence=[color])
                                elif ctype == "pie":
                                    fig = px.pie(df_chart, names=xcol, values=ycol,
                                                color_discrete_sequence=px.colors.qualitative.Pastel)
                                else:
                                    fig = px.bar(df_chart, x=xcol, y=ycol, color_discrete_sequence=[color])
                                fig.update_layout(margin=dict(t=10))
                                st.plotly_chart(fig, use_container_width=True)
                            except Exception as e:
                                st.caption(f"Could not render chart: {e}")
 
    # ── Default Olist dashboard ───────────────────────────────────────────────
    else:
        with st.spinner("Loading data..."):
            kpis = load_kpis()
            charts = load_charts()
 
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        with c1: metric_card("Total Orders", f"{kpis['total_orders']:,}")
        with c2: metric_card("Total Revenue", f"R${kpis['total_revenue']:,.0f}")
        with c3: metric_card("Avg Order Value", f"R${kpis['avg_order_value']:,.2f}")
        with c4: metric_card("Avg Review Score", f"⭐ {kpis['avg_review']}")
        with c5: metric_card("Customers", f"{kpis['total_customers']:,}")
        with c6: metric_card("Sellers", f"{kpis['total_sellers']:,}")
 
        st.divider()
 
        with st.expander("🤖 AI Insights", expanded=True):
            summary = f"""
            Olist e-commerce summary:
            - {kpis['total_orders']:,} total orders
            - R${kpis['total_revenue']:,.0f} total revenue
            - R${kpis['avg_order_value']:,.2f} average order value
            - {kpis['avg_review']} average review score out of 5
            - {kpis['total_customers']:,} unique customers
            - {kpis['total_sellers']:,} sellers on the platform
            Top categories: {', '.join(charts['top_categories']['category'].head(3).tolist()) if not charts['top_categories'].empty else 'N/A'}
            """
            with st.spinner("Generating insights..."):
                insights = get_insights(summary)
            insights_clean = re.sub(r"\*{1,2}([^*]+)\*{1,2}", r"\1", insights)
            insights_clean = insights_clean.replace("$", r"\$")
            st.info(insights_clean)
 
        st.divider()
 
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("Revenue Over Time")
            if not charts["revenue_over_time"].empty:
                fig = px.line(charts["revenue_over_time"], x="month", y="revenue",
                             labels={"month": "Month", "revenue": "Revenue (R$)"},
                             color_discrete_sequence=["#635BFF"])
                fig.update_layout(margin=dict(t=10))
                st.plotly_chart(fig, use_container_width=True)
 
        with col2:
            st.subheader("Order Status")
            if not charts["order_status"].empty:
                fig = px.pie(charts["order_status"], names="order_status", values="count",
                            color_discrete_sequence=px.colors.qualitative.Pastel)
                fig.update_layout(margin=dict(t=10), showlegend=True)
                st.plotly_chart(fig, use_container_width=True)
 
        col3, col4 = st.columns([1, 1])
        with col3:
            st.subheader("Top 10 Categories by Revenue")
            if not charts["top_categories"].empty:
                fig = px.bar(charts["top_categories"], x="revenue", y="category",
                            orientation="h",
                            labels={"revenue": "Revenue (R$)", "category": ""},
                            color_discrete_sequence=["#00C9A7"])
                fig.update_layout(margin=dict(t=10), yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig, use_container_width=True)
 
        with col4:
            st.subheader("Revenue by State")
            if not charts["revenue_by_state"].empty:
                fig = px.bar(charts["revenue_by_state"], x="state", y="revenue",
                            labels={"state": "State", "revenue": "Revenue (R$)"},
                            color_discrete_sequence=["#FF6B6B"])
                fig.update_layout(margin=dict(t=10))
                st.plotly_chart(fig, use_container_width=True)
 
        st.subheader("Review Score Distribution")
        if not charts["review_dist"].empty:
            fig = px.bar(charts["review_dist"], x="review_score", y="count",
                        labels={"review_score": "Score", "count": "Number of Reviews"},
                        color="review_score",
                        color_continuous_scale="RdYlGn")
            fig.update_layout(margin=dict(t=10), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — Chat
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    # Build SQL file from history
    sql_entries = [
        msg for msg in st.session_state.get("chat_history", [])
        if msg.get("role") == "assistant" and msg.get("sql")
    ]
    sql_export = "\n\n".join(
        f"-- Q: {msg.get('question', '')}\n{msg['sql']}"
        for msg in reversed(sql_entries)
    ) if sql_entries else ""
    header_col, dl_col, clear_col = st.columns([5, 1, 1])
    with header_col:
        st.subheader("💬 Ask the Data")
        if st.session_state.get("uploaded_filename"):
            st.caption(f"Querying **{st.session_state['uploaded_filename']}** · ask anything about your data")
        else:
            st.caption("Ask any business question in plain English. Examples:")
    with dl_col:
        if sql_entries:
            sql_export = "\n\n".join(
                f"-- Q: {msg.get('question', '')}\n{msg['sql']}"
                for msg in reversed(sql_entries)  # oldest first
            )
            st.download_button(
                label="⬇️ SQL",
                data=sql_export.encode("utf-8"),
                file_name="queries.sql",
                mime="text/plain",
                use_container_width=True,
                key="download_sql",
            )
        else:
            st.button("⬇️ SQL", disabled=True, use_container_width=True, key="download_sql_disabled")
    with clear_col:
        st.write("")  # spacer to align button vertically
        if st.button("🗑️ Clear", use_container_width=True, key="clear_chat"):
            st.session_state.chat_history = []
            st.rerun()
 
    if st.session_state.get("uploaded_table_name"):
        uploaded_cols = tuple(st.session_state.get("uploaded_columns", []))
        uploaded_filename = st.session_state.get("uploaded_filename", "file")
        try:
            with st.spinner("Generating example questions..."):
                examples = generate_example_questions(uploaded_cols, uploaded_filename)
        except Exception:
            examples = [f"How many rows are in the dataset?", "Show me the first 10 rows"]
    else:
        examples = [
            "Which 5 product categories have the highest average review score?",
            "Show monthly revenue for 2018",
            "Which states have the most customers?",
            "What is the most common payment type?",
            "Show the top 10 sellers by total revenue",
        ]
 
    cols = st.columns(len(examples))
    for i, ex in enumerate(examples):
        if cols[i].button(ex, key=f"ex_{i}", use_container_width=True):
            st.session_state["prefill"] = ex
 
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
 
    # ── Input box at the top ──────────────────────────────────────────────────
    prefill = st.session_state.pop("prefill", "")
    question = st.chat_input("Ask a question about the data...", key="chat_input")
    if prefill and not question:
        question = prefill
 
    if question:
        with st.spinner("Thinking..."):
            try:
                schema_override = None
                if st.session_state.get("uploaded_table_name"):
                    cols = st.session_state.get("uploaded_columns", [])
                    schema_override = ", ".join(cols)
                result = text_to_sql(question, schema_override=schema_override)
                sql = result.get("sql", "")
                explanation = result.get("explanation", "")
 
                try:
                    df = run_query(sql)
                except RuntimeError as qe:
                    df = pd.DataFrame()
                    explanation = f"Query error: {qe}"
 
                fig = None
                if not df.empty and len(df.columns) == 2 and pd.api.types.is_numeric_dtype(df.iloc[:, 1]):
                    fig = px.bar(df, x=df.columns[0], y=df.columns[1],
                                color_discrete_sequence=["#635BFF"])
 
                # Prepend to history so newest is always at the top
                st.session_state.chat_history.insert(0, {
                    "role": "assistant",
                    "content": f"_{explanation}_\n\n```sql\n{sql}\n```",
                    "df": df,
                    "fig": fig,
                })
                st.session_state.chat_history.insert(0, {
                    "role": "user",
                    "content": question,
                })
 
            except json.JSONDecodeError:
                st.session_state.chat_history.insert(0, {
                    "role": "assistant",
                    "content": "Sorry, I couldn't parse that into a query. Try rephrasing.",
                })
                st.session_state.chat_history.insert(0, {"role": "user", "content": question})
            except Exception as e:
                st.session_state.chat_history.insert(0, {
                    "role": "assistant",
                    "content": f"Something went wrong: {e}",
                })
                st.session_state.chat_history.insert(0, {"role": "user", "content": question})
 
        st.rerun()
 
    # ── Render history newest-first ───────────────────────────────────────────
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "df" in msg and msg["df"] is not None and not msg["df"].empty:
                st.dataframe(msg["df"], use_container_width=True)
            if "fig" in msg and msg["fig"] is not None:
                st.plotly_chart(msg["fig"], use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — View Data
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    if st.session_state.get("uploaded_df") is not None:
        df = st.session_state["uploaded_df"]
        filename = st.session_state.get("uploaded_filename", "uploaded file")
    else:
        # Load the main Olist orders table as the default view
        with st.spinner("Loading Olist orders..."):
            df = run_query("SELECT * FROM orders LIMIT 5000")
        filename = "olist_orders_dataset.csv"
        st.caption("Showing the Olist orders table (first 5,000 rows). Upload your own CSV to explore different data.")
 
    if df is not None:
        st.subheader(f"📄 {filename}")
        st.caption(f"{len(df):,} rows · {len(df.columns)} columns")

        # ── Summary stats ─────────────────────────────────────────────
        with st.expander("📊 Column Summary", expanded=True):
            summary_rows = []
            for col in df.columns:
                dtype = str(df[col].dtype)
                n_null = int(df[col].isnull().sum())
                n_unique = int(df[col].nunique())
                pct_null = f"{n_null / len(df) * 100:.1f}%" if len(df) > 0 else "0%"
                if pd.api.types.is_numeric_dtype(df[col]):
                    summary_rows.append({
                        "Column": col,
                        "Type": dtype,
                        "Unique": n_unique,
                        "Nulls": f"{n_null} ({pct_null})",
                        "Min": f"{df[col].min():,.2f}",
                        "Max": f"{df[col].max():,.2f}",
                        "Mean": f"{df[col].mean():,.2f}",
                    })
                else:
                    sample_vals = ", ".join(str(v) for v in df[col].dropna().unique()[:3])
                    summary_rows.append({
                        "Column": col,
                        "Type": dtype,
                        "Unique": n_unique,
                        "Nulls": f"{n_null} ({pct_null})",
                        "Min": "—",
                        "Max": "—",
                        "Mean": f"e.g. {sample_vals}",
                    })
            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
 
        st.divider()

        # ── Raw data with search ──────────────────────────────────────
        st.subheader("Raw Data")
        search = st.text_input("🔍 Filter rows (searches all columns)", placeholder="Type to filter...")
        if search:
            mask = df.apply(lambda col: col.astype(str).str.contains(search, case=False, na=False)).any(axis=1)
            filtered = df[mask]
            st.caption(f"{len(filtered):,} rows match")
        else:
            filtered = df
 
        st.dataframe(filtered, use_container_width=True, height=500)