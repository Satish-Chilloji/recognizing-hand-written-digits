from flask import Flask, request
from joblib import dump, load

app = Flask(__name__)

def compare_images(image1, image2):
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
def compare_images_route():
    if 'image1' not in request.files or 'image2' not in request.files:
        return ({'error': 'Both images are required'}), 400

    image1 = request.files['image1']
    image2 = request.files['image2']

    result = compare_images(image1, image2)

    return ({'are_images_same': result})

if __name__ == '__main__':
    best_model_path="models/tree_max_depth:100.joblib"
    app.run(host='0.0.0.0', port=5000)