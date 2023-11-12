from flask import Flask, request,jsonify
from joblib import dump, load
import numpy as np

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

def compare_images_here(image1, image2):
    best_model_path="models/tree_max_depth:100.joblib"
    image1=np.array(image1).reshape(1,-1)
    image2=np.array(image2).reshape(1,-1)
    # Load the model
    best_model = load(best_model_path)
    predicted_image1 = best_model.predict(image1)
    predicted_image2 = best_model.predict(image2)

    if predicted_image1 is None or predicted_image2 is None:
        return False
    
    if predicted_image1==predicted_image2:
        return True
    else:
        return False

@app.route('/compare_images', methods=['POST'])
def compare_images():
    # if 'image1' not in request.files or 'image2' not in request.files:
    #     return ({'error': 'Both images are required'}), 400

    # image1 = request.files['image1']
    # image2 = request.files['image2']

    # result = compare_images(image1, image2)
    # return ({'are_images_same': result})

    data = request.get_json()
    if 'image1' in data and 'image2' in data:
        image1 = data['image1']
        image2 = data['image2']
        result = compare_images_here(image1, image2)
        return ({'are_images_same': result})

if __name__ == '__main__':
    best_model_path="models/tree_max_depth:100.joblib"
    app.run(host='0.0.0.0', port=5000)