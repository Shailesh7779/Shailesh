from flask import Flask, jsonify

app = Flask(__name__)


@app.route("/")
def index():
    """
    this is a root dir of my server
    :return: str
    """
    return "This is root!!!!"



if __name__ == '__main__':
    app.run()