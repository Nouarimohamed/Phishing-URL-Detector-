from flask import Flask, render_template, request

from src.predict import PHISHING, predict_url

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    url = ""
    result = None
    if request.method == "POST":
        url = request.form["url"].strip()
        result = predict_url(url)
    return render_template("index.html", url=url, result=result, is_phishing=(result == PHISHING))


if __name__ == "__main__":
    app.run(debug=True)
