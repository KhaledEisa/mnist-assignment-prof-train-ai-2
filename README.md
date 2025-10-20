# MNIST Digit Classification

Assignment 1 - ML & Deep Learning

## Team
- Khaled Sherif Eissa 221010359
- Samir Mohamed Elshamy 221006600

## What is this?
We built 3 different models to recognize handwritten digits (0-9):
1. Random Forest (traditional ML)
2. Neural Network
3. CNN

Also made a Streamlit app where you can draw digits and test the models.

## How to run

### 1. Install stuff
```bash
pip install -r requirements.txt
```

### 2. Train models
Open `mnist_classifier.ipynb` and run all cells. This will train the models and save them.

### 3. Run the app
```bash
streamlit run app.py
```

or just double click `launch_app.bat`

## Results
- Random Forest: ~97% accuracy
- Neural Network: ~98% accuracy  
- CNN: ~99% accuracy (best one)

CNN works best because it understands the 2D structure of images better than the others.
