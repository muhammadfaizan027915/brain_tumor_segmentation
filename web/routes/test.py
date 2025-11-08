from flask import Blueprint, render_template

test_bp = Blueprint('test', __name__)

@test_bp.get("/test")

def render_test_template():
    return render_template("test.html")