from flask import Flask, render_template, request
import os
import cv2
import numpy as np
import joblib
import base64

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploaded'
model = joblib.load('model/decision_tree_model.pkl')

def extract_features_from_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (100, 100))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean = np.mean(img_gray)
    std = np.std(img_gray)
    return mean, std

solusi_dict = {
    "acne": "Gunakan pembersih wajah non-komedogenik.",
    "bags": "Tidur cukup dan minum air putih.",
    "redness": "Gunakan pelembap dan hindari iritan.",
}

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    solusi = None

    if request.method == 'POST':
        image_data = request.form.get('image_data')
        if image_data:
            img_bytes = base64.b64decode(image_data.split(',')[1])
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], 'capture.jpg')
            with open(img_path, 'wb') as f:
                f.write(img_bytes)

            mean, std = extract_features_from_image(img_path)
            features = np.array([[mean, std, mean, std, mean, std]])
            pred = model.predict(features)[0]
            prediction = pred
            solusi = solusi_dict.get(pred, "Solusi belum tersedia.")

    return render_template('index.html', prediction=prediction, solusi=solusi)

if __name__ == '__main__':
    app.run(debug=True)