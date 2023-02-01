#%%
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

#%% Plot figure (a)
ResultA = pd.read_csv('ResultA.csv')

#%%
fig, ax = plt.subplots()
sns.lineplot(data=ResultA, 
            x="episode_len", y="abs_error", hue="samp_size",
            ci=68, palette = "colorblind")
ax.set_xscale('log', basex=2)
ax.set_yscale('log', basey=2)
ax.legend(['n=512'], title="Training set size")
plt.xlabel("Horizon T")
plt.ylabel("Mean absolute error")
plt.rcParams.update({'axes.titlesize':12, 'axes.labelsize':12, 'xtick.labelsize':12,'ytick.labelsize':14,'legend.title_fontsize': 10,'legend.fontsize': 12})
plt.savefig("Fig_A.pdf")
plt.show()

#%% Plot figure (b)
ResultB  = pd.read_csv('ResultB.csv')

fig, ax = plt.subplots()
sns.lineplot(data=ResultB, 
            x="episode_num", y="abs_error", hue="episode_len",
            ci=68, palette = "colorblind")
ax.set_xscale('log', basex=2)
ax.set_yscale('log', basey=2)
ax.legend(['T=1', 'T=3','T=5'], title="Horizon")
plt.xlabel("Training set size n")
plt.ylabel("Mean absolute error")
plt.rcParams.update({'axes.titlesize':12, 'axes.labelsize':12, 'xtick.labelsize':12,'ytick.labelsize':14,'legend.title_fontsize': 10,'legend.fontsize': 12})
plt.savefig("Fig_B.pdf")
plt.show()