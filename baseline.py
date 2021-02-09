from sklear.ensemble import RandomForestClassifier
import pandas as pd

def train_model(X,y):
    m = RandomForestClassifier()
    m.fit(X,y)
    y_pred = m.predict(X)
    return m