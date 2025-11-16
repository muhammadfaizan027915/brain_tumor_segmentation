from flask import render_template, Blueprint

results_bp = Blueprint("Results", __name__)

@results_bp.route("/results", methods=["GET"])
def render_results_page():
    return render_template("results.html", active="results")