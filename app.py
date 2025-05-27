# app.py
import os
import uuid
import time
import torch
import pickle
import csv
import cv2
import numpy as np
import pandas as pd
from flask import Flask, request, render_template, jsonify, send_file
from datetime import datetime, timedelta
from threading import Thread

from sympy.series.limits import heuristics
from ultralytics import YOLO
from torchvision.ops import nms
from dotenv import load_dotenv
import logging
import socket
import xgboost as xgb
import joblib
import random
# ==============================
# Load environment variables
# ==============================
load_dotenv()


# ==============================
# Configuration & Constants
# ==============================

# Paths for uploads and outputs
UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', 'static/uploads')
OUTPUT_FOLDER = os.getenv('OUTPUT_FOLDER', 'static/outputs')


MODEL_PATHS = {
    'TPE': {
        'model': 'models/cd_xgboost_v1.json',
        'scaler': 'models/cd_scaler.pkl',
        'y_scaler': 'models/cd_y_scaler.pkl'
    },

    'NLE': {
        'model': 'models/hg_xgboost_v2.json',
        'scaler': 'models/hg_scaler_v2.pkl',
        'y_scaler': 'models/hg_y_scaler_v2.pkl'
    }
}
## Thresholds for object detection
IOU_THRESHOLD = 0.5
CONFIDENCE_THRESHOLD = 0.4
#FIXED_HEIGHT = 200  # Fixed ROI height
SEED = 42  # Random seed for reproducibility

random.seed(SEED)
np.random.seed(SEED)
xgb.set_config(verbosity=0)  # Suppress warnings
torch.manual_seed(SEED)






# ==============================
# Load Machine Learning Models
# ==============================

# Preload models into memory to improve performance
MODEL_OBJECTS = {}
for nanosensor_type, model_path in MODEL_PATHS.items():
    model_path = MODEL_PATHS.get(nanosensor_type)['model']
    scaler_path = MODEL_PATHS.get(nanosensor_type)['scaler']
    y_scaler_path = MODEL_PATHS.get(nanosensor_type)['y_scaler']


    MODEL_OBJECTS[nanosensor_type] = {
        'model': xgb.Booster(),
        'scaler': joblib.load(scaler_path) if scaler_path else None,
        'y_scaler': joblib.load(y_scaler_path) if y_scaler_path else None,
    }

    MODEL_OBJECTS[nanosensor_type]['model'].load_model(model_path)


# Initialize YOLO Model for ROI extraction
model = YOLO('models/yolov8n_roi_extraction_v2.pt')

# ==============================
# Flask App Initialization
# ==============================n
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Ensure upload and output directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ==============================
# Logging Configuration
# ==============================
logging.basicConfig(
    filename='experiment.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Track file timestamps for cleanup
file_timestamps = {}

# ==============================
# Cleanup Function for Old Files
# ==============================
def cleanup_files():
    """
      Periodically deletes files older than 5 minutes from upload and output directories.
      Runs every 5 minutes in a separate thread.
      """
    while True:
        now = datetime.now()
        for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER]:
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                if os.path.isfile(file_path):
                    file_age = now - datetime.fromtimestamp(os.path.getmtime(file_path))
                    if file_age > timedelta(minutes=5):
                        try:
                            os.remove(file_path)
                            logging.info(f"Deleted old file: {file_path}")
                        except Exception as e:
                            logging.error(f"Error deleting {file_path}: {e}")
        time.sleep(300)  # Run every 5 minutes


# Start cleanup thread
cleanup_thread = Thread(target=cleanup_files, daemon=True)
cleanup_thread.start()

# ==============================
# Function: Run YOLO Model
# ==============================
def run_yolo_model(image_path: str):
    """
    Run the YOLO model on an image to extract regions of interest (ROIs) and annotate the image.

    Parameters
    ----------
    image_path : str
        The path to the input image.

    Returns
    -------
    tuple
        A list of sorted ROIs and the annotated image.
    """
    image = cv2.imread(image_path)
    results = model(image)
    if not results:
        return [], image

    rois_rgb = []
    h, w, _ = image.shape
    for result in results:
        if not hasattr(result, 'boxes') or result.boxes is None:
            continue

        boxes = result.boxes.xyxy.clone().detach()
        confs = result.boxes.conf.clone().detach()
        classes = result.boxes.cls.clone().detach()

        # Apply Non-Maximum Suppression (NMS)
        keep_indices = nms(boxes, confs, iou_threshold=IOU_THRESHOLD)
        boxes, confs, classes = boxes[keep_indices], confs[keep_indices], classes[keep_indices]

        # Filter boxes by confidence threshold
        valid_indices = confs > CONFIDENCE_THRESHOLD
        boxes, confs, classes = boxes[valid_indices], confs[valid_indices], classes[valid_indices]

        if boxes.size(0) == 0:
            return [], image  # No ROIs detected

        max_y2 = max(int(box[3]) for box in boxes)

        # Extract and annotate ROIs
        for box, conf, cls in zip(boxes, confs, classes):
            x1, y1, x2, y2 = map(int, box.tolist())

            height = y2 - y1
            y2 = max_y2-round(height*0.5)

            y1 = min(h, y2-round(height*0.55))
            roi = image[y1:y2, x1:x2]
            mean_rgb = cv2.mean(roi)[:3]
            rois_rgb.append((mean_rgb[2], mean_rgb[1], mean_rgb[0], cls.item(), x1, y1, x2, y2))
            #IT takes BGR value!!

    rois_rgb_sorted = sorted(rois_rgb, key=lambda roi: roi[4])
    for i, roi in enumerate(rois_rgb_sorted, start=1):
        x1, y1, x2, y2 = roi[4:8]
        cv2.rectangle(
            image, (x1, y1), (x2, y2),
            color=(0, 255, 0),
            thickness=3,
            lineType=cv2.LINE_AA  # Anti-aliased lines
        )

        cv2.putText(
            image, f'T{i}', (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.7,
            color=(255, 255, 255),
            thickness=2,
            lineType=cv2.LINE_AA
        )

    print(rois_rgb_sorted)
    return rois_rgb_sorted, image
# ==============================
# Function: Calculate Intensity Change
# ==============================
def calculate_intensity(rois_rgb_sorted, has_second_sample=False):
    """
    Calculate the percentage change in intensity for each RGB channel.

    Parameters
    ----------
    rois_rgb_sorted : list
        Sorted list of ROIs with RGB values.
    has_second_sample : bool
        Whether Second Sample data is available.

    Returns
    -------
    tuple
        Percentage changes for first and second samples.
    """
    r_values = [roi[0] for roi in rois_rgb_sorted]
    g_values = [roi[1] for roi in rois_rgb_sorted]
    b_values = [roi[2] for roi in rois_rgb_sorted]

    baseline_intensity = (r_values[0], g_values[0], b_values[0])
    print(baseline_intensity)
    print(f'Baseline Intensity:{baseline_intensity}')
    first_sample_intensity = (r_values[1], g_values[1], b_values[1])
    print(f'First Sample Intensity:{first_sample_intensity}')
    second_sample_intensity = None
    # if has_second_sample:
    #     second_sample_intensity = (r_values[2], g_values[2], b_values[2])
    #     print(second_sample_intensity)

    if has_second_sample:
        second_sample_intensity = (r_values[2], g_values[2], b_values[2])
        print(f'Second Sample Intensity:{second_sample_intensity}')
        second_sample_list = list(second_sample_intensity)
    else:
        second_sample_list = None
    return list(baseline_intensity), list(first_sample_intensity), second_sample_list

    # return list(baseline_intensity),list(first_sample_intensity),list(second_sample_intensity)

# ==============================
# Function: Get HMI Description
# ==============================
def get_detected_hmi(nanosensor_type: str) -> str:
    if nanosensor_type == 'TPE':
        return "HMI Detected: Cadmium"
    elif nanosensor_type == 'NLE':
        return "HMI Detected: Mercury"
    return "HMI Detected: Unknown"


# ==============================
# Function: Load Model and Predict
# ==============================


def predict_concentration_xgb(nanosensor_type,baseline_rgb,  first_sample_rgb):
    known_concentration = 0.00000000001
    try:
        model = MODEL_OBJECTS[nanosensor_type]['model']
        if not model:
            raise FileNotFoundError(f"Model for {nanosensor_type} not found.")
        scaler= MODEL_OBJECTS[nanosensor_type]['scaler']
        if not scaler:
            raise FileNotFoundError(f"Scaler for {nanosensor_type} not found.")
        y_scaler =MODEL_OBJECTS[nanosensor_type]['y_scaler']
        if not y_scaler:
            raise FileNotFoundError(f"Y_Scaler for {nanosensor_type} not found.")
            # trained_xgb_model = xgb.Booster()
            # trained_xgb_model.load_model(model)
            scaler = joblib.load(scaler_path)
            y_scaler = joblib.load(y_scaler_path)

        baseline_rgb, first_sample_rgb = np.array(baseline_rgb, dtype=np.float32), np.array(first_sample_rgb, dtype=np.float32)
        baseline_rgb = np.where(baseline_rgb == 0, 1, baseline_rgb)
        known_concentration = max(known_concentration, 1e-6)

        pct_change = (first_sample_rgb - baseline_rgb) / baseline_rgb * 100
        features = pd.DataFrame([list(pct_change) + [pct_change.mean(), np.abs(pct_change).mean()]],
                                columns=['R', 'G', 'B', 'avg_pct_change', 'abs_avg_pct_change'])

        features_scaled = scaler.transform(features)
        dfeatures = xgb.DMatrix(features_scaled)
        concentration_difference_pred = model.predict(dfeatures)[0]
        concentration_difference_pred = y_scaler.inverse_transform([[concentration_difference_pred]])[0][0]

        return max(0, known_concentration + concentration_difference_pred)
    except FileNotFoundError as e:
        logging.error(f"Model loading error: {e}")
        raise
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        raise
# ==============================
# Flask Routes
# ==============================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles image upload, runs YOLO model, calculates intensity changes,
    and predicts concentration.    """


    file = request.files['image']
    nanosensor_type = request.form['nanosensor_type']
    has_second_sample = request.form.get('has_second_sample', 'off') == 'on'
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 5MB limit

    if not file.mimetype.startswith('image/'):
        return jsonify({'error': 'Invalid file type, only images are allowed'}), 400

    if request.content_length and request.content_length > MAX_FILE_SIZE:

        return jsonify({'error': 'File size is too large. The maximum size allowed is 10MB.'}), 400

    if not file:
        return jsonify({'error': 'No file uploaded'}), 400
    # Ensure filenames are safe and prevent path traversal
    input_filename = os.path.basename(f"{uuid.uuid4().hex}_{file.filename}")
    input_filepath = os.path.join(app.config['UPLOAD_FOLDER'], input_filename)


    file.save(input_filepath)

    output_filename = f"{uuid.uuid4().hex}_output.jpg"
    output_filepath = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)

    # Run YOLO model to get ROI information
    try:
        rois_rgb_sorted, processed_image = run_yolo_model(input_filepath)
    except Exception as e:
        logging.error(f"YOLO model error: {e}")
        return render_template('error.html',
                               error_message="Error processing the image. Please try again with a valid image."), 500

    if len(rois_rgb_sorted) < 2:
        return render_template('error.html',
                               error_message="The uploaded image does not contain enough ROIs (at least 2 are required). Please upload a valid image."), 400

    if len(rois_rgb_sorted) > 3:
        return render_template('error.html',
                               error_message="The uploaded image contains too many ROIs (maximum allowed is 3). Please upload a valid image."), 400
    # Check for second_sample data if the checkbox is selected
    if has_second_sample and len(rois_rgb_sorted) < 3:
        return render_template('error.html',
                               error_message='The "More Than One Sample" checkbox is selected, but the Another Concentration data (third ROI) is missing. Please provide valid input and try again.'), 400



    # Calculate intensity changes

    baseline, first_sample, second_sample = calculate_intensity(rois_rgb_sorted, has_second_sample)

    try:
        first_sample_concentration = predict_concentration_xgb(
            nanosensor_type, baseline_rgb=baseline, first_sample_rgb=first_sample
        )

        second_sample_concentration = None
        if has_second_sample:

            second_sample_concentration = predict_concentration_xgb(
                nanosensor_type, baseline_rgb=baseline, first_sample_rgb=second_sample
            )

        # Save processed image
        cv2.imwrite(output_filepath, processed_image)

        # Log the file paths for cleanup
        file_timestamps[input_filepath] = datetime.now()
        file_timestamps[output_filepath] = datetime.now()

        detected_hmi = get_detected_hmi(nanosensor_type)

        # Inside `render_template('result.html')`
        return render_template('result.html', response={
            'image_path': f'/{output_filepath}',
            'first_sample_concentration': f"{first_sample_concentration:.4f} nM",
            'second_sample_concentration': f"{second_sample_concentration:.4f} nM" if second_sample_concentration else None,
            'detected_hmi': detected_hmi,
            'nanosensor_type': nanosensor_type

        })


    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return render_template('error.html', error_message="An unexpected error occurred. Please try again later."), 500

@app.route('/save-results', methods=['POST'])
def save_results():
    data = request.json
    first_sample_concentration = data.get('first_sample_concentration')
    second_sample_concentration = data.get('second_sample_concentration')
    # Debug print to check what values we are receiving
    print(f"Test Sample Concentration: {first_sample_concentration}")
    print(f"Standard Concentration: {second_sample_concentration}")
    # Function to extract concentration value and unit
    def extract_concentration(concentration):
        if concentration and isinstance(concentration, str):
            parts = concentration.split()  # Split into value and unit
            if len(parts) == 2:
                value, unit = parts
                try:
                    # Validate value
                    value = float(value)
                    if value < 0:
                        raise ValueError("Concentration cannot be negative.")
                    return value, unit
                except ValueError:
                    return None, None
        return None, None

    # Extract concentration values
    first_sample_value, first_sample_unit = extract_concentration(first_sample_concentration)
    second_sample_value, second_sample_unit = extract_concentration(second_sample_concentration)
    # Check and debug the extracted values
    print(f"Extracted First Sample Value: {first_sample_value}, Unit: {first_sample_unit}")
    print(f"Extracted Second Sample Value: {second_sample_value}, Unit: {second_sample_unit}")
    # Prepare the data for CSV
    first_sample_conc_str = f"{first_sample_value} {first_sample_unit}" if first_sample_value else 'N/A'
    second_sample_conc_str = f"{second_sample_value} {second_sample_unit}" if second_sample_value else 'N/A'
    # Generate a unique name for the CSV file
    csv_filename = f"{uuid.uuid4().hex}_result.csv"
    file_path = os.path.join(app.config['OUTPUT_FOLDER'], csv_filename)
    # Save results to CSV
    try:
        with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            # Write the headers (including RGB Channel and RGB Change columns)
            writer.writerow(['First_Sample_Concentration', 'Second_Sample_Concentration'])
            # Write the concentration values in the first row
            writer.writerow([first_sample_conc_str, second_sample_conc_str])
             # Format the change to 2 decimal places
            file_timestamps[file_path] = datetime.now()
            # Return the CSV file
            return send_file(file_path, as_attachment=True, download_name='result.csv', mimetype='text/csv')

    except Exception as e:
        logging.error(f"Csv File Processing Error: {e}")
        return render_template('error.html',
                               error_message="Error processing the CSV file."), 500


# ==============================
# Run Flask App
# ==============================



import os
import socket

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)

    cert_path = 'certs/cert.pem'
    key_path = 'certs/key.pem'

    if os.path.exists(cert_path) and os.path.exists(key_path):
        print(f"Server is running securely at: https://{local_ip}:{port}")
        print(f"Accessible via: https://127.0.0.1:{port}")
        ssl_context = (cert_path, key_path)
        app.run(debug=False, host='0.0.0.0', port=port, ssl_context=ssl_context)
    else:
        print(f"No SSL certificate found. Running on HTTP at http://{local_ip}:{port}")
        print(f"Accessible via: http://127.0.0.1:{port}")
        app.run(debug=False, host='0.0.0.0', port=port)
