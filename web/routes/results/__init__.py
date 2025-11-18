from quart import render_template, Blueprint

results_bp = Blueprint("Results", __name__)

@results_bp.route("/results", methods=["GET"])
async def render_results_page():
    return await render_template("results.html", active="results")