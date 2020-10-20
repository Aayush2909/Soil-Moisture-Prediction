import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Dataset Visualization/relation plotter
def relationPlotter(dataset, n):
    sns.set_theme(style="whitegrid")

    fig, axs = plt.subplots(ncols=3, figsize=(20,5))
    sns.scatterplot(x='Tsoil', y='Soil Moisture', data=dataset, ax=axs[0]).set_title("Field {} Stats".format(str(n)))
    sns.scatterplot(x='Tair', y='Soil Moisture', data=dataset, ax=axs[1]).set_title("Field {} Stats".format(str(n)))
    sns.scatterplot(x='RHpercent',y='Soil Moisture', data=dataset, ax=axs[2]).set_title("Field {} Stats".format(str(n)))

    fig.savefig("Field{} Stats.png".format(str(n)))


#Model history/data plotter
def historyPlotter(model_history):
    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(1, 1, figsize=(10,6))
    ax.plot(np.sqrt(model_history.history['loss']), 'r', label='train')
    ax.plot(np.sqrt(model_history.history['val_loss']), 'b' ,label='val')
    ax.set_xlabel(r'Epoch', fontsize=20)
    ax.set_ylabel(r'Loss', fontsize=20)
    ax.legend()
    ax.tick_params(labelsize=20)

    fig.savefig("Model Statistics.png")

