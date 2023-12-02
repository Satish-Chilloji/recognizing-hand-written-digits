from flask import Flask, request

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/",methods=["POST"])
def hell_world_post():
    return {"op": "Hello, World POST "+ request.json["suffix"]}

@app.route("/sum/<x>/<y>")
def sum(x,y):
    sum_=int(x)+int(y)
    return str(sum_) 

@app.route('/inference', methods=['GET'])
def inference():
    # js=request.get_json()
    # x = js['x']
    # y = js['y']
    a = request.args.get('a', type=int)
    b = request.args.get('b', type=int)
    # if a is None or b is None:
    #     return 'Please provide values for both "a" and "b" as query parameters.'

    # Calculate the sum
    result = a + b

    # Return the result as a string
    #return f'The sum of {a} and {b} is: {result}'
    return str(result)

    #return x+y

