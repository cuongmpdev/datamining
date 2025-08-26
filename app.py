from flask import Flask, render_template
from pages.reduct.reduct import bp as reduct_bp

app = Flask(__name__)

app.register_blueprint(reduct_bp)

@app.route("/")
def home():
    return render_template("home.html")

if __name__ == "__main__":
    app.run(debug=True)
