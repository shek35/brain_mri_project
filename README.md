### `README.md`

```markdown
# Brain MRI Tumor Detection

This project is a deep learning solution for classifying Brain MRI images as **Tumor** or **No Tumor** using TensorFlow and Keras. It includes the full pipeline for data preparation, model training, and deployment via a REST API built with Flask.

## Project Overview

- **Goal**: Use a Convolutional Neural Network (CNN) to classify Brain MRI images for tumor detection.
- **Tools Used**: Python, TensorFlow, Keras, OpenCV, Flask, Postman.
- **Deployment**: The trained model is deployed as a REST API using Flask, enabling easy integration into other systems.

## Dataset

- **Source**: The dataset used for this project is the [Brain MRI Images for Brain Tumor Detection Dataset](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection) available on Kaggle.
- **Data Structure**: The dataset consists of two categories of images — `yes` (with tumor) and `no` (without tumor).

## Installation

### Prerequisites

- Python 3.8 or higher
- Virtual Environment (venv)
- Required Python packages listed in `requirements.txt`

### Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/brain_mri_project.git
   cd brain_mri_project
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Linux/Mac
   .\venv\Scripts\activate  # Windows
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download and place the dataset**:
   - Download the dataset from Kaggle and place it in the `data/` directory.

## Running the Project

### Data Preparation
Run the following command to prepare the data:
```bash
python prepare_data.py
```
This script will load, resize, and normalize the MRI images, saving them in a format suitable for model training.

### Model Training
Train the CNN model using the following command:
```bash
python train_model.py
```
The trained model will be saved as `brain_mri_model.h5`.

### Running the Flask App
To serve the model as a REST API, run:
```bash
python app.py
```
The Flask server will be available at `http://127.0.0.1:5000`.

### Testing the API
You can test the `/predict` endpoint using Postman or cURL. Example with Postman:
- Send a **POST** request to `http://127.0.0.1:5000/predict` with an MRI image file as form-data under the key `file`.
- The response will contain the prediction (`"Tumor"` or `"No Tumor"`).

## License
This project is open-source and available under the MIT License.
```

This `README.md` provides all the essential information in a concise format and is ready for easy copy-pasting. Let me know if you need any modifications or additional details!
