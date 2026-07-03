
from flask import Flask, request, render_template_string
from scoring import score_comic

app = Flask(__name__)

HTML = """
<!doctype html>
<html>
  <head>
    <title>Comic Signal (v1)</title>
  </head>
  <body style="font-family: Arial, sans-serif; margin: 40px;">
    <h2>Comic Signal (v1)</h2>
    <p>Type a comic name or keywords. This returns a <b>proxy</b> signal based on clustering.</p>

    <form method="post">
      <input name="query" style="width:520px; padding:8px;" placeholder="e.g., Spider-Man, X-Men, Wolverine" value="{{query|default('')}}">
      <button type="submit" style="padding:8px 12px;">Search</button>
    </form>

    {% if result %}
      <hr>
      <h3>Result</h3>
      <p><b>Match:</b> {{ result["comic_name"] }} — {{ result["issue_title"] }}</p>
      <p><b>Publish date:</b> {{ result["publish_date"] }}</p>
      <p><b>Cluster:</b> {{ result["cluster"] }} ({{ result["cluster_name"] }})</p>
      <p><b>Signal:</b> {{ result["label"] }}</p>
    {% elif query %}
      <hr>
      <p><b>No match found.</b> Try fewer words or a different spelling.</p>
    {% endif %}
  </body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    query = ""
    result = None

    if request.method == "POST":
        query = request.form.get("query", "")
        result = score_comic(query)

    return render_template_string(HTML, query=query, result=result)

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)