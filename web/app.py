from routes import results
from routes import uploads
from flask import Flask

app = Flask(__name__)
app.register_blueprint(uploads.uploads_bp)
app.register_blueprint(results.results_bp)

if __name__ == "__main__":
    app.run(debug=True)