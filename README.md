# FACIAL-MOOD-DETECTION-SYSTEM-Python-OpenCV-CNN
📌 Project Overview

This project implements a facial mood detection system that identifies human emotions from facial images using Computer Vision and Deep Learning techniques. The system detects faces using OpenCV and classifies facial expressions into seven distinct emotion categories using a Convolutional Neural Network (CNN) trained on the CK+ dataset.

🎯 Problem Statement

Automated recognition of human emotions from facial expressions has applications in human–computer interaction, mental health analysis, and intelligent systems. The goal of this project is to build an efficient and accurate facial emotion classifier using CNNs combined with OpenCV-based face detection.

🧠 Emotions Classified

Happy

Sad

Angry

Fear

Disgust

Surprise

Neutral

🛠️ Technologies Used

Python

OpenCV

Convolutional Neural Networks (CNN)

NumPy

CK+ (Extended Cohn-Kanade) Dataset

⚙️ Methodology

Data Collection & Preprocessing

Loaded facial expression images from the CK+ dataset (https://www.kaggle.com/datasets/shuvoalok/ck-dataset)

Converted images to grayscale and resized them for consistency

Face Detection

Used OpenCV’s Haar Cascade Classifier to detect and extract facial regions of interest (ROI)

Model Training

Trained a CNN model on processed facial ROIs to learn emotion-specific features

Evaluation

Evaluated model performance by measuring classification accuracy across different emotion classes

📊 Dataset

Dataset Used: Extended Cohn-Kanade (CK+) Dataset

Contains labeled facial expression images for emotion recognition tasks

🚀 Key Features

Real-time face detection using OpenCV

CNN-based deep learning model for emotion classification

End-to-end pipeline from preprocessing to prediction

Focus on fundamental computer vision and deep learning concepts

📂 Project Structure
├── dataset/
│   └── CK+/
├── haarcascade_frontalface_default.xml
├── model/
│   └── cnn_model.py
├── train.py
├── preprocess.py
├── predict.py
├── requirements.txt
└── README.md


📈 Results

The CNN model successfully learned facial expression patterns and achieved reliable performance in classifying multiple emotion categories from facial images.

🔮 Future Enhancements

Improve accuracy using data augmentation

Add real-time webcam-based emotion detection

Deploy the model using Flask or Streamlit

Extend support for larger and more diverse datasets
