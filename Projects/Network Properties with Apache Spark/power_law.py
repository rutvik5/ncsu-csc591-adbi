import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv['stanford/youtube.graph.large.csv']
new_df = df.drop(['Unnamed: 0'],axis=1)
plot_df = new_df.sort_values(by=['degree'])
plt.plot(plot_df['degree'], plot_df['count'])
plt.show()
