import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('brainsize.txt', sep='\t', na_values=['NA'])

sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.show()

sns.scatterplot(data=df, x='Weight', y='MRI_Count', hue='Gender')
plt.show()