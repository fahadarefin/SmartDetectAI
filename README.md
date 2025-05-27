

# Smart Nanosensor Image Processing and Concentration Prediction System

## Overview

This project implements a complete pipeline for detecting regions of interest (ROIs) in nanosensor images, extracting color-based features, and predicting heavy metal ion (HMI) concentrations using advanced machine learning techniques. The system integrates:

* **YOLOv8** for ROI detection
* Dataset augmentation for robustness
* **XGBoost** regression models with Bayesian hyperparameter tuning for accurate prediction
* A secure, user-friendly **Flask** web interface for image upload, processing, and result visualization

Designed for robustness, scalability, and ease of use by researchers and practitioners in environmental heavy metal detection.

---

## Table of Contents

* [Overview](#overview)
* [Project Structure](#project-structure)
* [Setup and Installation](#setup-and-installation)
* [Generating Self-Signed SSL Certificates](#generating-self-signed-ssl-certificates)
* [Sample Preparation](#sample-preparation)
* [How to Take Images for Analysis](#how-to-take-images-for-analysis)
* [Example Images](#example-images)
* [File Descriptions](#file-descriptions)
* [Usage](#usage)
* [Contributing](#contributing)
* [References](#references)
* [Licensing](#licensing)

---

## Project Structure

```plaintext
.
├── app.py
├── requirements.txt
├── models/
│   ├── cd_xgboost_v1.json
│   ├── cd_scaler.pkl
│   ├── cd_y_scaler.pkl
│   ├── hg_xgboost_v2.json
│   ├── hg_scaler_v2.pkl
│   ├── hg_y_scaler_v2.pkl
│   └── yolov8n_roi_extraction_v2.pt
├── static/
│   ├── uploads/
│   └── outputs/
├── templates/
│   ├── index.html
│   ├── result.html
│   └── error.html
├── certs/
│   ├── cert.pem
│   └── key.pem
Training/
├── RGBTraining/
│   ├── augment_dataset.py
│   ├── cd_augmented_dataset_v1.csv
│   ├── XG_BOOST.py
│   └── hg_augmented_dataset_v2.csv
├── YoloTraining/
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   ├── labels/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── roi.yaml
├── experiment.log
└── README.md
```

---

## Setup and Installation

### 1. Prerequisites

* Python 3.8 or higher
* Git
* Virtual environment tool (`venv`)
* Recommended: PyCharm IDE for Windows users

### 2. Install and Run Locally

```bash
git clone https://github.com/fahadarefin/smartdetectai.git
cd smartdetectai
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate
pip install -r requirements.txt
python app.py
```

Access the app at:

```
http://127.0.0.1:5000
```

### 3. Accessing on Local Network

* Find your IPv4 address via `ipconfig` (Windows) or `ifconfig` (Linux/macOS).
* Run the Flask app (same as above).
* Access from other devices on the same network via:

```
http://<your-local-ip>:5000
```

---

## Deploying to a Web Server

### Option 1: Using Gunicorn & Nginx (Production)

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

**Nginx configuration example:**

```nginx
server {
    listen 80;
    server_name yourdomain.com;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

Restart Nginx:

```bash
sudo systemctl restart nginx
```

### Option 2: Using Docker

```bash
docker build -t smartdetectai .
docker run -p 5000:5000 smartdetectai
```

---

## Generating Self-Signed SSL Certificates

For HTTPS support during development or internal testing:

```bash
openssl req -x509 -newkey rsa:4096 -sha256 -days 365 -nodes \
  -keyout certs/key.pem -out certs/cert.pem \
  -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost"
```

*Note:* Browsers will warn for self-signed certs; use Let's Encrypt or other CA for production.

---

## Sample Preparation

Before capturing images for analysis, prepare your samples carefully to ensure accurate results:

1. Place your sample in one test tubes (or two test tubes if you are testing multiple samples for the same target heavy metal ion).
2. Add the appropriate **AgNS** (Silver Nanosensor) to each test tube. Choose the nanosensor type based on the target heavy metal ion you want to detect.
3. Prepare a control test tube by adding pure water along with the same type of **AgNS** nanosensor.
4. Arrange the test tubes **inside the SmartDetect chamber**, positioning the control tube with pure water on the **left** side and the sample tubes on the **right** side.
5. Allow the colorimetric reaction to develop for approximately **20 to 30 minutes** before proceeding to image capture.

---

## How to Take Images for Analysis

High-quality images are essential for accurate concentration prediction.

### Recommendations

* Use **consistent, good lighting**; avoid shadows and glare.
* Position the camera **directly in front of** the nanosensor sample for a flat view.
* Keep the image **sharp and in focus** (use tripod if necessary).
* Ensure the sample region occupies a significant part of the frame.
* Use **high resolution** and avoid heavy compression artifacts.

### Supported Formats and Limits

* Accepted formats: PNG, JPG, JPEG
* Max file size: 10 MB

### Using the Web Interface

* Upload images by drag-and-drop or **Select a File** option.
* Use the **Take a Photo** button (mobile devices preferred) to capture live images.
* Adjust contrast, brightness, and threshold sliders if needed.
* Preview images before submission.

> **Important:** Modern smartphones require HTTPS-secured web pages to grant access to the device camera. To use the **Take Photo** feature within the web interface, ensure you are accessing the app via **HTTPS** (secure connection). Self-signed certificates or valid SSL certificates are needed to enable camera access.

---

## Example Images

### Web Interface Preview

<img src="/.docs/web_ui.png" alt="Web UI" style="max-width:600px; width:50%; height:auto;" />
<p><em>Screenshot of the upload interface showing drag-and-drop and camera capture button.</em></p>

### Sample Nanosensor Image

<img src="/.docs/sample_image.png" alt="Sample Image" style="max-width:600px; width:50%; height:auto;" />
<p><em>Example of a well-captured nanosensor test tube image used for analysis.</em></p>

### Result Page

<img src="/.docs/result_page.png" alt="Result Page" style="max-width:600px; width:50%; height:auto;" />
<p><em>Screenshot of the result page displaying the concentration prediction output.</em></p>


---

## File Descriptions

| File/Folder          | Purpose                                                                                         |
| -------------------- | ----------------------------------------------------------------------------------------------- |
| `app.py`             | Flask backend: handles uploads, ROI detection, feature extraction, prediction, and UI rendering |
| `XG_BOOST.py`        | Model training script using XGBoost with Bayesian hyperparameter tuning                         |
| `augment_dataset.py` | Dataset augmentation script to improve model generalization                                     |
| `YOLO.py`            | Script for training YOLOv8 model on ROI dataset                                                 |
| `requirements.txt`   | Lists all Python dependencies                                                                   |
| `models/`            | Stores pretrained YOLO and XGBoost models and scalers                                           |
| `static/uploads/`    | Temporary storage for user-uploaded images                                                      |
| `static/outputs/`    | Stores processed result images                                                                  |
| `templates/`         | HTML templates for Flask web UI                                                                 |
| `certs/`             | SSL certificates for HTTPS support                                                              |

---

## Usage

### Running the Flask App

```bash
python app.py
```

* Runs with HTTPS if certificates are present in `certs/`.
* Otherwise, runs HTTP on port 5000.
* Access via `https://localhost:5000` or `http://localhost:5000`.

### Training YOLO Model

```bash
python YOLO.py
```

* Trains ROI detection model on custom dataset.

### Training XGBoost Models

```bash
python XG_BOOST.py
```

* Trains regression models for heavy metal concentration prediction.

### Augmenting Dataset

```bash
python augment_dataset.py
```

* Performs dataset augmentation to improve model robustness.

---

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a new feature branch.
3. Implement your changes adhering to PEP 8 style.
4. Add tests where applicable.
5. Submit a pull request with detailed description.

---

## References

* [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/)
* [scikit-optimize (Bayesian Optimization)](https://scikit-optimize.github.io/)
* [Flask Web Framework](https://flask.palletsprojects.com/)
* [OpenSSL Documentation](https://www.openssl.org/docs/)

---

## Licensing

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT), allowing free use, modification, and distribution.

> **Note:** All intellectual property rights related to research data, experimental results, and nanosensor technology belong to the Microsystems and Nano Engineering Group, University of Dhaka. Use for academic and research purposes is encouraged with proper citation. Commercial use requires prior permission from the group.

---

*This system aims to advance nanosensor data analysis by combining cutting-edge image processing and machine learning, making environmental heavy metal detection more accessible and interpretable.*

