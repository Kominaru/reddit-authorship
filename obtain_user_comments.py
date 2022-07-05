import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

data=pd.read_csv("coronavirus_2021q1_all.csv")

print(f"{len(pd.unique(data['Author']))} Unique users")

data.groupby('Author').count().plot()
value_counts=data['Author'].value_counts()