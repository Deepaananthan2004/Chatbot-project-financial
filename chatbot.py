from flask import Flask, request, render_template_string
from groq import Groq
import os
import pandas as pd
from io import StringIO

# Initialize Groq client
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable not set")

client = Groq(api_key=GROQ_API_KEY)

app = Flask(__name__)

HTML_TEMPLATE = '''
<!doctype html>
<title>Financial Chatbot</title>
<h2>Financial Chatbot (Powered by Groq API)</h2>

<form method="POST" enctype="multipart/form-data">
    <label>Upload Financial CSV:</label>
    <input type="file" name="file" accept=".csv" required><br><br>
    <label>Ask a financial question:</label>
    <input name="query" style="width: 400px;" placeholder="Example: What is the revenue for 2024?" required>
    <input type="submit" value="Ask">
</form>

{% if response %}
    <p><strong>Bot:</strong> {{ response }}</p>
{% endif %}

{% if dataframe %}
    <h3>Uploaded Dataset Preview:</h3>
    {{ dataframe|safe }}
{% endif %}
'''

# Function to query Groq
def ask_groq(prompt, context_text):
    try:
        response = client.chat.completions.create(
            model="grok-1",
            messages=[
                {"role": "system", "content": "You are a helpful financial assistant. Answer queries based on the provided financial data."},
                {"role": "user", "content": f"Data: {context_text}\n\nQuestion: {prompt}"}
            ],
            temperature=0.5,
            max_tokens=300
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {e}"

@app.route("/", methods=["GET", "POST"])
def home():
    response = ""
    dataframe_html = ""
    context_text = ""

    if request.method == "POST":
        # Get uploaded file
        uploaded_file = request.files.get("file")
        user_query = request.form.get("query")

        if uploaded_file:
            try:
                # Read CSV into Pandas
                df = pd.read_csv(uploaded_file)
                dataframe_html = df.head(10).to_html(classes="data", header="true")  # preview first 10 rows
                # Convert CSV to string context for the chatbot
                context_text = df.to_csv(index=False)
            except Exception as e:
                response = f"Error reading CSV: {e}"

        if user_query and context_text:
            response = ask_groq(user_query, context_text)

    return render_template_string(HTML_TEMPLATE, response=response, dataframe=dataframe_html)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)