# Network IDS using ML

## Overview
Project implements a ml based IDS to detect malicious network traffic

## Problem Statement
Modern networks face threats such as DoS, probing, and brute attacks. Manual detection is slow

## Technologies Used
- Python
- Scikit-learn
- Random Forest
- NSL-KDD Dataset from Kaggle

## How It Works
1. Load network traffic data
2. Encode categorical features
3. Train ML classifier
4. Predict benign vs malicious traffic

## Results
Model achieves high accuracy in detecting attacks. You can see it in confusion matrix in results folder

## How to Run
```bash
pip install -r requirements.txt
cd src
python train\_model.py
