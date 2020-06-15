import numpy as np
import pandas as pd
import seaborn as sns

class_names = ["P_face", "P_left_ear", "P_right_ear"]

df = pd.read_csv("mousie.csv")
distributions = df[class_names].values
df["Prediction"] = np.argmax(distributions, axis=1)
df["Confidence"] = [distr[k] for distr, k in zip(distributions, df["Prediction"].values)]

sns.scatterplot(x="X", y="Y", hue="Prediction", data=df, size="Confidence")