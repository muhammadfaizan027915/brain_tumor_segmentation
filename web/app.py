from routes import test
from flask import Flask, request

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home ():
    print(request.method)
    return "Hello, World!"


if __name__ == "__main__":
    app.register_blueprint(test.test_bp)
    app.run(debug=True)