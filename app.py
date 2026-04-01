from flask import Flask, render_template, request
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from io import BytesIO
import os
import tensorflow as tf
import base64
import gdown

app = Flask(__name__)

# ------------------ MODEL ------------------
MODEL_PATH = "vgg16_best.h5"
MODEL_URL = "https://drive.google.com/uc?id=1sq-Cz_Jvtyns3bxx8_kqdt8dfZDInZMr"

if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

print("Loading model...")
model = load_model(MODEL_PATH)

class_labels = [
    'adenocarcinoma',
    'large.cell.carcinoma',
    'normal',
    'squamous.cell.carcinoma'
]

# ------------------ GRAD-CAM ------------------
def get_gradcam(img_array):
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer('block5_conv3').output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)

        if isinstance(predictions, list):
            predictions = predictions[0]

        pred_index = tf.argmax(predictions[0])
        loss = predictions[:, pred_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-10)

    return heatmap.numpy()

# ------------------ STAGE CALCULATION ------------------
def calculate_stage(heatmap):
    heatmap = heatmap / (np.max(heatmap) + 1e-8)

    tumor_pixels = np.sum(heatmap > 0.5)
    total_pixels = heatmap.size

    coverage = (tumor_pixels / total_pixels) * 100

    if coverage <= 10:
        stage = "Stage I"
    elif coverage <= 25:
        stage = "Stage II"
    elif coverage <= 45:
        stage = "Stage III"
    else:
        stage = "Stage IV"

    return round(coverage, 2), stage

# ------------------ ROUTES ------------------

@app.route('/')
def home():
    return render_template('welcome.html')

@app.route('/analyze')
def analyze():
    return render_template('analyze.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['scan']

    patient_name = request.form.get('patient_name')
    age = request.form.get('age')
    gender = request.form.get('gender')
    smoking = request.form.get('smoking')

    # ---------- Image processing ----------
    img_bytes = file.read()
    img = image.load_img(BytesIO(img_bytes), target_size=(224,224))
    arr = image.img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)

    # ---------- Prediction ----------
    preds = model.predict(arr)[0]
    idx = np.argmax(preds)

    label = class_labels[idx]
    confidence = float(np.max(preds) * 100)

    # Class-wise confidence
    confidences = {
        class_labels[i]: float(preds[i])
        for i in range(len(class_labels))
    }

    # ---------- Grad-CAM ----------
    heatmap = get_gradcam(arr)

    orig = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    heatmap = cv2.resize(heatmap, (orig.shape[1], orig.shape[0]))

    # ✅ Calculate stage BEFORE modifying heatmap
    coverage, stage = calculate_stage(heatmap)

    # ---------- Visualization ----------
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    superimposed = cv2.addWeighted(orig, 0.6, heatmap_color, 0.4, 0)

    # ---------- Convert to base64 ----------
    _, orig_buf = cv2.imencode('.png', orig)
    _, heat_buf = cv2.imencode('.png', superimposed)

    original_base64 = base64.b64encode(orig_buf).decode('utf-8')
    gradcam_base64 = base64.b64encode(heat_buf).decode('utf-8')

    # ---------- Send to UI ----------
    return render_template('results.html',
        prediction=label,
        confidence=f"{confidence:.2f}%",
        original=original_base64,
        gradcam=gradcam_base64,
        patient_name=patient_name,
        age=age,
        gender=gender,
        smoking=smoking,
        confidences=confidences,
        coverage=f"{coverage}%",
        stage=stage
    )

# ------------------ RUN ------------------
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)