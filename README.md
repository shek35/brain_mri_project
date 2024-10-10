```markdown
# Brain Tumor Detection using TensorFlow

## Overview
This project aims to detect brain tumors using MRI images with a Convolutional Neural Network (CNN) built using TensorFlow. The model classifies MRI scans as either containing a tumor or being tumor-free.

## Dataset
- **Source**: [Kaggle - Brain MRI Images for Brain Tumor Detection](https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection)
- Contains MRI images divided into `tumor` and `no_tumor` categories.

## Installation
1. Clone the repository.
2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # For Windows: .\venv\Scripts\activate
   pip install -r requirements.txt
   ```

## Usage
1. **Training the Model**:
   - Preprocesses the data and trains a CNN model.
   - Run:
     ```bash
     python scripts/train_model.py
     ```

2. **Deploying with Flask**:
   - Start the Flask server:
     ```bash
     python app.py
     ```
   - Use Postman to send a POST request with an MRI image for prediction.

## Results
- The model outputs the probability of the presence of a tumor.
- Visualize training accuracy and loss using `matplotlib`.

## Acknowledgments
- Data sourced from [Kaggle - Brain MRI Images for Brain Tumor Detection](https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection).
```
