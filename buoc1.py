import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression

# 1. Load dữ liệu
df = pd.read_csv("Metro_Interstate_Traffic_Volume.csv")

print("=== 5 dòng đầu ===")
print(df.head())
