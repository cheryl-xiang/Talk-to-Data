# 📊 TalkToData

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B?logo=streamlit)
![Powered by Claude](https://img.shields.io/badge/Powered%20by-Claude%20Sonnet%204.6-blueviolet)
![License](https://img.shields.io/badge/License-MIT-green)

An AI-powered business intelligence dashboard that lets you explore the [Olist Brazilian E-Commerce dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) using natural language. Ask questions in plain English, get SQL, results, and charts — no SQL knowledge required.

**No database setup needed** — data loads automatically from the Olist public dataset on startup.

---

## ✨ Features

- **📈 BI Dashboard** — KPIs, revenue trends, top product categories, order status breakdown, revenue by state, and review score distribution
- **🤖 AI Insights** — Claude automatically generates a plain-English business summary of the data on load
- **💬 Natural Language Chat** — type any question about the data and get back the SQL query, a results table, and an auto-generated chart
- **⚡ In-memory queries** — powered by DuckDB, no local database file required

---

## 🖥️ Demo

> **Live demo:** https://talk-to-data-gappv5jsmdhuejeirtr6n3y.streamlit.app/

Example questions you can ask:
- *"Which 5 product categories have the highest average review score?"*
- *"Show monthly revenue for 2018"*
- *"What percentage of orders were delivered late?"*
- *"Which states have the most customers?"*
- *"Show the top 10 sellers by total revenue"*

---

## 🛠️ Tech Stack

| Layer | Tool |
|---|---|
| UI & app framework | [Streamlit](https://streamlit.io) |
| Query engine | [DuckDB](https://duckdb.org) (in-memory) |
| Charts | [Plotly](https://plotly.com/python/) |
| LLM (text-to-SQL + insights) | [Claude Sonnet 4.6 via Anthropic API](https://www.anthropic.com) |
| Data | [Olist Brazilian E-Commerce](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) |

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/talktodata.git
cd talktodata
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add your Anthropic API key

Copy the example secrets file:
```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

Then edit `.streamlit/secrets.toml` and add your key:
```toml
ANTHROPIC_API_KEY = "your_key_here"
```

Get a free API key at [console.anthropic.com](https://console.anthropic.com).

### 4. Run the app
```bash
streamlit run app.py
```

> The dataset loads automatically on first run (~20-30 seconds). No database download needed.

---

## ☁️ Deploy to Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your repo
3. Set your main file path to `app.py`
4. Before deploying, click **Advanced settings → Secrets** and add:
```toml
ANTHROPIC_API_KEY = "your_key_here"
```
5. Click **Deploy** — that's it

---

## 📁 Project Structure

```
talktodata/
├── app.py                          # main Streamlit app
├── requirements.txt                # Python dependencies
├── .gitignore
├── .streamlit/
│   └── secrets.toml.example        # template for API key (safe to commit)
└── README.md
```

---

## 📊 Dataset

This project uses the [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce), containing ~100,000 orders from 2016–2018 across multiple Brazilian marketplaces. The app loads 8 CSV tables at startup:

- `orders` — core order records
- `order_items` — line items, price, freight
- `order_payments` — payment type and value
- `order_reviews` — customer review scores and comments
- `customers` — customer location data
- `sellers` — seller location data
- `products` — product dimensions and category
- `product_category_name_translation` — Portuguese to English category names

The dataset is licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).

---

## 📄 License

MIT