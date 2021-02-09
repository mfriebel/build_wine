from sklearn.datasets import load_wine
import pandas as pd

def train_model():
    d = load_wine()
    print(d['DESCR'])
    X = pd.DataFrame(d['data'], columns=d['feature_names'])
    y = d['target']  # cultivator

train_model()