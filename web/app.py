from config import MAX_FILE_SIZE
from routes import results
from routes import uploads
from quart import Quart

app = Quart(__name__)
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Registering the blueprints
app.register_blueprint(uploads.uploads_bp)
app.register_blueprint(results.results_bp)

if __name__ == "__main__":
    app.run(debug=True)