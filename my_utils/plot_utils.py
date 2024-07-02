# Module containing some useful plotting functions.

import matplotlib.pyplot as plt
from typing import Optional
from my_utils.att_cdam_utils import get_cmap
import pandas as pd
import random
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


def plot_res_class(original_img, maps, model_output, save_name: Optional[str] = None):
    """Using matplotlib, plot the original image and the relevance maps"""
    maps2plot = []
    target_names = []
    maps2plot.append(maps[0])
    target_names.append("Attention Map")
    for key in maps[1].keys():
        maps2plot.append(maps[1][key])
        target_names.append("CDAM"+ "\nMalignant class")

    if len(maps2plot) == 2:
        # Binary classification:
        plt.figure(figsize=(6, 3))
        num_plots = 3
        plt.subplot(1, num_plots, 1)
        plt.imshow(original_img, cmap='gray')
        plt.title("Original")
        plt.axis("off")
        for i, m in enumerate(maps2plot):
            plt.subplot(1, num_plots, i + 2)
            plt.imshow(m, cmap=get_cmap(m))  
            plt.title(target_names[i])
            plt.axis("off")
        plt.suptitle(f"Probability of malignant class: {round(model_output, 2)}")
        plt.subplots_adjust(wspace=0.0, hspace=0)

    if save_name:
        import os
        if not os.path.exists("relevance_maps"):
            os.makedirs("relevance_maps")
        plt.savefig(f"relevance_maps/{save_name}", format="png", transparent=True, bbox_inches='tight')

    return None


def sample_test_example(FOLD:int)->str:
    df = pd.read_pickle("dataset/ALL_annotations_df.pkl")
    with open(f"dataset/splitted_sets/test_fold_{FOLD}.pkl", "rb") as f:
        test_examples = pickle.load(f)
    idx = random.choice(test_examples)
    ID = df.iloc[idx]["path"]
    return ID