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
    page_title="TalkToData · Olist BI",
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
Be specific about numbers. Use plain English. No bullet points."""

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

def _call_llm(messages: list) -> dict:
    """Call the LLM and parse the JSON response."""
    client = _get_client()
    msg = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=500,
        system=TEXT_TO_SQL_SYSTEM,
        messages=messages,
    )
    raw = msg.content[0].text.strip()
    raw = re.sub(r"```json|```", "", raw).strip()
    return json.loads(raw)

def text_to_sql(question: str) -> dict:
    """Generate SQL from a natural language question, with one auto-retry on error."""
    messages = [{"role": "user", "content": question}]
    result = _call_llm(messages)
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
        result = _call_llm(messages)
 
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

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["📈 Dashboard", "💬 Ask the Data"])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — Dashboard
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    with st.spinner("Loading data..."):
        kpis = load_kpis()
        charts = load_charts()

    # KPI row
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1: metric_card("Total Orders", f"{kpis['total_orders']:,}")
    with c2: metric_card("Total Revenue", f"R${kpis['total_revenue']:,.0f}")
    with c3: metric_card("Avg Order Value", f"R${kpis['avg_order_value']:,.2f}")
    with c4: metric_card("Avg Review Score", f"⭐ {kpis['avg_review']}")
    with c5: metric_card("Customers", f"{kpis['total_customers']:,}")
    with c6: metric_card("Sellers", f"{kpis['total_sellers']:,}")

    st.divider()

    # AI Insights box
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
        st.info(insights)

    st.divider()

    # Charts row 1
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

    # Charts row 2
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

    # Review distribution
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
    st.subheader("💬 Ask the Data")
    st.caption("Ask any business question in plain English. Examples:")
    
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

    # Render chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "df" in msg and msg["df"] is not None and not msg["df"].empty:
                st.dataframe(msg["df"], use_container_width=True)
            if "fig" in msg and msg["fig"] is not None:
                st.plotly_chart(msg["fig"], use_container_width=True)

    # Input
    prefill = st.session_state.pop("prefill", "")
    question = st.chat_input("Ask a question about the data...", key="chat_input")
    if prefill and not question:
        question = prefill

    if question:
        # Show user message
        with st.chat_message("user"):
            st.markdown(question)
        st.session_state.chat_history.append({"role": "user", "content": question})

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = text_to_sql(question)
                    sql = result.get("sql", "")
                    explanation = result.get("explanation", "")

                    st.markdown(f"_{explanation}_")
                    with st.expander("View SQL"):
                        st.code(sql, language="sql")

                    df = run_query(sql)
                    fig = None

                    if not df.empty:
                        st.dataframe(df, use_container_width=True)

                        # Auto-chart: if 2 cols and second is numeric, try a bar chart
                        if len(df.columns) == 2 and pd.api.types.is_numeric_dtype(df.iloc[:, 1]):
                            fig = px.bar(df, x=df.columns[0], y=df.columns[1],
                                        color_discrete_sequence=["#635BFF"])
                            st.plotly_chart(fig, use_container_width=True)

                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": f"_{explanation}_\n\n```sql\n{sql}\n```",
                        "df": df,
                        "fig": fig,
                    })

                except json.JSONDecodeError:
                    msg = "Sorry, I couldn't parse that question into a query. Try rephrasing."
                    st.warning(msg)
                    st.session_state.chat_history.append({"role": "assistant", "content": msg})
                except Exception as e:
                    msg = f"Something went wrong: {e}"
                    st.error(msg)
                    st.session_state.chat_history.append({"role": "assistant", "content": msg})
