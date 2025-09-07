from flask import Flask, render_template
from pages.reduct.reduct import bp as reduct_bp
from pages.decision_tree.decision_tree import bp as decision_tree_bp

app = Flask(__name__)

blueprints = [reduct_bp, decision_tree_bp]

for bp in blueprints:
    app.register_blueprint(bp)


@app.route("/")
def home():
    return render_template("home.html")


if __name__ == "__main__":
    app.run(debug=True)
