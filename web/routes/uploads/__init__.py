from os import path
from flask import render_template, Blueprint, request, redirect, url_for
from config import UPLOAD_FOLDER

uploads_bp = Blueprint("Uploads", __name__)

@uploads_bp.route("/", methods=["GET"])
def render_uploads_page():
    return render_template("uploads.html", active="uploads")

@uploads_bp.route("/upload_file", methods=["POST"])
def upload_file():
    file = request.files["file"]
    
    filepath = path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)
    
    return redirect(url_for("Results.render_results_page"))


