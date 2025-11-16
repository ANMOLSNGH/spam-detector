from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load model and vectorizer
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html", title="Home")

@app.route("/about")
def about():
    return render_template("about.html", title="About")

@app.route("/contact")
def contact():
    return render_template("contact.html", title="Contact")

@app.route("/predict", methods=["POST"])
def predict():
    msg = request.form["message"]

    # Vectorize
    transformed = vectorizer.transform([msg])
    result = model.predict(transformed)[0]

    label = "SPAM" if result == 1 else "NOT SPAM"
    color = "red" if result == 1 else "lightgreen"

    return render_template("result.html",
                           title="Result",
                           message=msg,
                           result=label,
                           color=color)

if __name__ == "__main__":
    app.run(debug=True)
