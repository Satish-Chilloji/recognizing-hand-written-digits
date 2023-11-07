from flask import Flask, request

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/sum/<x>/<y>")
def sum(x,y):
    sum_=int(x)+int(y)
    return str(sum_) 

@app.route('/inference', methods=['POST'])
def inference():
    js=request.get_json()
    x = js['x']
    y = js['y']
    return x+y

