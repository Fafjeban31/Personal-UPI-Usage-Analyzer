
import gradio as gr
import fitz  # PyMuPDF
import re
import json
import pandas as pd
import plotly.express as px
import markdown2
from openai import OpenAI
from dotenv import load_dotenv

# === Load .env ===
import os
api_key = os.environ["OPENAI_API_KEY"]  # Hugging Face secret

# === OpenAI Client Setup ===
os.environ["OPENAI_API_KEY"] = api_key
client = OpenAI(base_url="https://openrouter.ai/api/v1")

# === Helper to clean LLM JSON ===
def clean_llm_json(text):
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()

# === Extract Cleaned Text from PDF ===
def extract_clean_transaction_text(pdf_path):
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        return f"❌ Error opening PDF: {e}", None

    lines = []
    for page in doc:
        lines += page.get_text().split("\n")

    cleaned = []
    for i, line in enumerate(lines):
        line = line.strip()
        if not line or i < 5:
            continue
        if any(re.search(p, line.lower()) for p in [
            r"(upi|statement|phonepe|paytm|gpay|name|account|bank)",
            r"\b[789]\d{9}\b", r"(utr|transaction id|ref)",
            r"[xX*]{2,}\d{2,}", r"(credited to|paid by)", 
            "page", "system generated", "support", "note:"
        ]):
            continue
        cleaned.append(line)

    result = "\n".join(cleaned)
    return result, result if cleaned else None

# === AI Financial Advice ===
def get_financial_advice(cleaned_text):
    prompt = (
        "You are a brilliant financial advisor. Analyze the following transaction text and give:\n"
        "1. Spending summary\n2. Poor spending habits\n3. Budget plan\n4. Category spend breakdown\n"
        "5. Investment suggestions\n6. Ways to save ₹5,000/month\n\nUse markdown with headings and bullets.\n\n"
        f"{cleaned_text}"
    )
    response = client.chat.completions.create(
        model="deepseek/deepseek-chat",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# === Save Advice to HTML ===
def save_llm_output_as_html(llm_output, output_path="Financial_Report.html"):
    html = markdown2.markdown(llm_output)
    styled = f"""
    <html>
    <head>
    <style>
        body {{ font-family: Arial; padding: 20px; line-height: 1.6; }}
        h1, h2 {{ color: #2E86C1; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ border: 1px solid #ccc; padding: 8px; }}
        th {{ background-color: #f2f2f2; }}
    </style>
    </head>
    <body>{html}</body></html>
    """
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(styled)
    return output_path

# === Chart Generation ===
def generate_charts_from_cleaned(cleaned_text):
    chart_prompt = f"""
    You are a financial data analyst. From this cleaned transaction text, return:
    {{
      "category_spending": [{{"category": "...", "amount": ...}}],
      "top_merchants": [{{"merchant": "...", "amount": ...}}],
      "monthly_spending": [{{"month": "...", "amount": ...}}],
      "credit_vs_debit": {{"total_credit": ..., "total_debit": ...}},
      "daily_spending": [{{"date": "YYYY-MM-DD", "debit": ..., "credit": ...}}],
      "essentials_vs_discretionary": [{{"type": "Essential", "amount": ...}}, {{"type": "Discretionary", "amount": ...}}],
      "cumulative_spending": [{{"date": "YYYY-MM-DD", "cumulative_debit": ...}}],
      "weekday_spending": [{{"weekday": "Monday", "amount": ...}}],
      "time_of_day_spending": [{{"period": "Morning", "amount": ...}}],
      "income_vs_spend_trend": [{{"date": "YYYY-MM-DD", "debit": ..., "credit": ...}}]
    }}
    Only return valid JSON. If any data missing, guess it.
    {cleaned_text}
    """
    response = client.chat.completions.create(
        model="deepseek/deepseek-chat",
        messages=[{"role": "user", "content": chart_prompt}]
    )
    raw = clean_llm_json(response.choices[0].message.content)
    parsed = json.loads(raw)

    figs = {
        "Category-wise Spending": px.pie(pd.DataFrame(parsed["category_spending"]), names="category", values="amount"),
        "Top Merchants": px.bar(pd.DataFrame(parsed["top_merchants"]), x="amount", y="merchant", orientation="h"),
        "Monthly Spending": px.bar(pd.DataFrame(parsed["monthly_spending"]), x="month", y="amount"),
        "Credit vs Debit": px.pie(pd.DataFrame([
            {"type": "Credit", "amount": parsed["credit_vs_debit"]["total_credit"]},
            {"type": "Debit", "amount": parsed["credit_vs_debit"]["total_debit"]}
        ]), names="type", values="amount"),
        "Daily Debit vs Credit": px.line(pd.DataFrame(parsed["daily_spending"]), x="date", y=["debit", "credit"]),
        "Essentials vs Discretionary": px.pie(pd.DataFrame(parsed["essentials_vs_discretionary"]), names="type", values="amount"),
        "Cumulative Spending": px.line(pd.DataFrame(parsed["cumulative_spending"]), x="date", y="cumulative_debit"),
        "Weekday Spending": px.bar(pd.DataFrame(parsed["weekday_spending"]), x="weekday", y="amount"),
        "Time of Day Spending": px.bar(pd.DataFrame(parsed["time_of_day_spending"]), x="period", y="amount"),
        "Income vs Spend Trend": px.line(pd.DataFrame(parsed["income_vs_spend_trend"]), x="date", y=["debit", "credit"]),
    }

    return list(figs.values())

# === Gradio Interface ===
with gr.Blocks(theme="huggingface") as app:
    gr.Markdown("## 💸 Smart UPI Financial Analyzer")

    with gr.Tabs():
        with gr.Tab("📄 Advisor + PDF"):
            file_input = gr.File(label="Upload PDF")
            btn = gr.Button("Analyze PDF")
            cleaned_output = gr.Textbox(label="🧹 Cleaned Text", lines=10)
            advice_output = gr.Textbox(label="📋 Financial Advice", lines=20)
            html_file = gr.File(label="📄 Download HTML Report")

            def handle_advice(file):
                cleaned, raw = extract_clean_transaction_text(file.name)
                if not raw:
                    return cleaned, "", None
                advice = get_financial_advice(raw)
                html_path = save_llm_output_as_html(advice)
                return cleaned, advice, html_path

            btn.click(fn=handle_advice, inputs=file_input, outputs=[cleaned_output, advice_output, html_file])

        with gr.Tab("📊 Charts"):
            gr.Markdown("Paste cleaned transaction text below:")
            chart_input = gr.Textbox(lines=10, label="Cleaned Text")
            chart_btn = gr.Button("Generate Charts")
            chart_outputs = [gr.Plot(label=label) for label in [
                "Category-wise Spending", "Top Merchants", "Monthly Spending", "Credit vs Debit",
                "Daily Debit vs Credit", "Essentials vs Discretionary", "Cumulative Spending",
                "Weekday Spending", "Time of Day Spending", "Income vs Spend Trend"
            ]]
            chart_btn.click(fn=generate_charts_from_cleaned, inputs=chart_input, outputs=chart_outputs)

app.launch(share=True)
