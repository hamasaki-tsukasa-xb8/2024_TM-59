from flask import Flask, render_template
from rl_model import get_recommendation

app = Flask(__name__)

@app.route('/')
def index():
    recommendation = get_recommendation()
    return render_template('index.html', recommendation=recommendation)

if __name__ == '__main__':
    app.run(debug=True)