# Parkinsons Deep Learning

Classifying Parkinsons in people via the UCI Dataset: https://archive.ics.uci.edu/ml/datasets/Parkinsons

# Setup

Create a virtual environment

```bash
virtualenv .venv
```

Download and install all required libraries

```bash
pip install -r requirements.txt
```

Create ".env" file and populate it with your classification model settings

```bash
EPOCHS = *value here excluding asterisks*
TEST_SIZE = *value here excluding asterisks*
```

Run the python file

```bash
python main.py
```

# Data Wrangling

- Normalisation / Scaling
- Dimensionality Reduction
- Outlier detection

# Fine Tuning Classification Models

- Grid search cross validation
- Brute force dimensionality reduction
- Ensure splits are consistent/same

## Neural Network

- Regularisation
- Dropout
