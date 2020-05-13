import numpy as np
import pandas as pd
import seaborn as sns

class_names = ["P_face", "P_left_ear", "P_right_ear"]

df = pd.read_csv("mousie.csv")
distributions = df[class_names].values
df["Pred_class"] = np.argmax(distributions, axis=1)
df["Pred_class_name"] = df["Pred_class"].apply(lambda k: class_names[k])
df["Confidence"] = [distr[k] for distr, k in zip(distributions, df["Pred_class"].values)]

accuracy = (df["Class"] == df["Pred_class"]).mean()
print("Accuracy: {}".format(accuracy))

sns.scatterplot(x="X", y="Y", hue="Pred_class_name", data=df, size="Confidence")