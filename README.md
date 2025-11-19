# DMT Network Anomaly Detection

This project implements a **Deep Metric Learning (DMT)** approach for **Network Anomaly Detection**. The system detects abnormal network traffic patterns by learning deep feature representations, which can help in identifying potential security threats such as DDoS attacks, data breaches, and other malicious network behaviors.

**GitHub Repository:** [DMT Network Anomaly Detection](https://github.com/prajwaldevaraj-2001/DMT-Network-Anamoly-Detection)

---

## Overview

The project uses **Deep Metric Learning (DMT)** to detect anomalies in network traffic based on a dataset that contains normal and abnormal network behavior.</br> 
It uses deep learning techniques to learn embeddings and distances between data points, and based on these distances, identifies outliers that represent anomalies.

### Key Features:
- **Anomaly Detection**: Identifies abnormal patterns in network traffic data.
- **Deep Metric Learning**: Uses advanced deep learning techniques to learn feature embeddings that distinguish normal and anomalous behavior.
- **Performance Evaluation**: Evaluates the performance of the anomaly detection system using metrics such as precision, recall, F1-score, and ROC curves.
- **Real-time Detection**: Potential for real-time anomaly detection and alerts.
  
---

## Technologies Used

- **Deep Metric Learning (DMT)**: For learning embeddings of network traffic.
- **Python**: Primary language for implementation.
- **TensorFlow/Keras**: Deep learning frameworks used to build and train models.
- **Scikit-learn**: For data preprocessing and evaluation metrics.
- **Matplotlib/Seaborn**: For visualizing the data and evaluation results.

---

Installation & Setup
1. Clone the Repository</br>
git clone https://github.com/prajwaldevaraj-2001/DMT-Network-Anamoly-Detection.git</br>
cd DMT-Network-Anamoly-Detection</br>

2. Install Dependencies</br>
Install the required libraries:</br>
pip install -r requirements.txt</br>

3. Download the Dataset</br>
Ensure you have access to a network traffic dataset. You can use publicly available datasets like the CICIDS 2017 dataset or any network traffic logs containing both normal and anomalous behavior. Place the dataset in the data/ folder.</br>

4. Train the Model</br>
To train the model, run the training script:</br>
python models/train.py</br>

5. Evaluate the Model</br>
After training, you can evaluate the model's performance:</br>
python evaluation/results.py</br>

## Usage
1. Data Preprocessing:
- Use the scripts/preprocess.py file to preprocess the network traffic data (normalize features, handle missing values, etc.).
- The dataset should have labels indicating whether the traffic is normal or anomalous.

2. Model Training:
- The training script models/train.py builds and trains a Deep Metric Learning model on the preprocessed data.
- The model learns to map network traffic data to feature embeddings that separate normal and anomalous traffic.

3. Anomaly Detection:
- After training the model, the learned embeddings can be used to detect anomalies in new data.
- The model computes the distance between the embeddings of a test sample and a set of reference points (normal traffic).
- If the distance is above a certain threshold, the sample is flagged as anomalous.

4. Performance Evaluation:
- Evaluate the model using metrics like accuracy, precision, recall, F1-score, and ROC-AUC to assess the anomaly detection performance.

## Example Usage
After setting up the environment and training the model, you can:</br>

Preprocess the Data:</br>
Run the preprocessing script to prepare the data for training:</br>
python scripts/preprocess.py</br>

Train the Model:</br>
Train the DMT model:</br>
python models/train.py</br>

Evaluate the Results:</br>
Once the model is trained, evaluate its performance:</br>
python evaluation/results.py</br>

## Future Improvements
- Real-Time Anomaly Detection: Implement real-time network anomaly detection using a continuous stream of data.
- Advanced Metrics: Integrate advanced metrics for anomaly detection, such as confusion matrix and precision-recall curve.
- Visualization: Visualize the learned feature embeddings using tools like t-SNE or PCA for better understanding of the decision boundaries.

## Project Structure

```
DMT-Network-Anomaly-Detection/
│
├── data/                          # Dataset and related files
│   ├── network_traffic.csv        # Dataset containing network traffic logs
│   └── anomaly_labels.csv         # Labels for identifying anomalies in the data
│
├── models/                        # Model-related files
│   ├── dmt_model.py               # Deep Metric Learning model implementation
│   └── train.py                   # Script to train the DMT model
│
├── scripts/                       # Additional utility scripts
│   └── preprocess.py              # Data preprocessing script
│
├── evaluation/                    # Evaluation results and metrics
│   └── results.py                 # Script for evaluating model performance
│
├── requirements.txt               # Python dependencies
├── README.md                      # Documentation
└── main.py                         # Entry point to run the system

```

### Developed by

**Prajwal Devaraj**
