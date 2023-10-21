import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv('/datavisualization/bayes_features.csv')

data.head(10)

fig,ax=plt.subplots(figsize=(8,8))
colors=['red','green']
ax.tick_params(direction='in')

ax.scatter(data.positive,data.negative,s=0.1,marker='*',
              c=[colors[int(k)] for k in data.sentiment])

plt.xlim(-250,0)
plt.ylim(-250,0)

plt.xlabel("Positive")
plt.ylabel("Negative")

plt.show()
