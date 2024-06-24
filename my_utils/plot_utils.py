# Module containing some useful plotting functions.

import matplotlib.pyplot as plt
import pandas as pd
import pickle


def plot_hists(fold:int, datapath:str, subsets_saving_path:str):
    df = pd.read_pickle(f"{datapath}/ALL_annotations_df.pkl")
    with open(subsets_saving_path+f"train_fold_{fold}.pkl", 'rb') as file:
        train_indices = pickle.load(file)
    with open(subsets_saving_path+f"test_fold_{fold}.pkl", 'rb') as file:
        test_indices = pickle.load(file)
     
    fig, axs = plt.subplots(1, 2, figsize=(7.5, 5))
    _, bins, bars = axs[0].hist(df.iloc[train_indices]["target"], bins=2)
    bin_width = bins[1] - bins[0]
    desired_width = bin_width * 0.8
    for bar in bars:
        bar.set_width(desired_width)
        bar.set_x(bar.get_x() + (bin_width - desired_width) / 2)
    axs[0].bar_label(bars)
    axs[0].set_title("train set")
    axs[0].set_ylabel("counts")
    
    _, bins, bars = axs[1].hist(df.iloc[test_indices]["target"], bins=2)
    bin_width = bins[1] - bins[0]
    desired_width = bin_width * 0.8
    for bar in bars:
        bar.set_width(desired_width)
        bar.set_x(bar.get_x() + (bin_width - desired_width) / 2)
    axs[1].bar_label(bars)
    axs[1].set_title("test set")
    axs[1].set_ylabel("counts")

    fig.tight_layout()
    fig.subplots_adjust(top=0.85)
    fig.suptitle(f'Distribution of classes in fold {fold}', size=16)
    return None