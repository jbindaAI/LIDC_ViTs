# Module containing some useful plotting functions.

import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional
from my_utils.att_cdam_utils import get_cmap
import pandas as pd
import random
import pickle
import os


# Loading annotations for all dataset
with open("dataset/ALL_annotations_df.pkl", "rb") as file:
    ann_df = pickle.load(file)


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


def plot_res_class(original_img, maps, model_output, save_name:Optional[str]=None):
    """Using matplotlib, plot the original image and the relevance maps"""
    maps2plot = []
    target_names = []
    maps2plot.append(maps[0])
    target_names.append("Attention Map")
    for key in maps[1].keys():
        maps2plot.append(maps[1][key])
        target_names.append("CDAM"+ "\nMalignant class")

    fig, axs = plt.subplots(1, 3, figsize=(9, 4), layout='constrained')
    ## Original:
    axs[0].imshow(original_img[:,:,0], cmap='gray')
    axs[0].set_title("Original")
    axs[0].tick_params(axis='both', 
                       which='both', 
                       bottom=False, 
                       left=False,
                       labelbottom=False,
                       labelleft=False
                      )
    ## Attention map:
    axs[1].imshow(maps2plot[0], cmap=get_cmap(maps2plot[0]))
    axs[1].set_title(target_names[0])
    axs[1].tick_params(axis='both', 
                       which='both', 
                       bottom=False, 
                       left=False,
                       labelbottom=False,
                       labelleft=False
                      )
    ## CDAM Map:
    cdam = axs[2].imshow(maps2plot[1], cmap=get_cmap(maps2plot[1]))
    axs[2].set_title(target_names[1])
    axs[2].tick_params(axis='both', 
                       which='both', 
                       bottom=False, 
                       left=False,
                       labelbottom=False,
                       labelleft=False
                      )
    fig.colorbar(cdam, ax=axs[2], shrink=0.6)
    plt.suptitle(f"Probability of malignant class: {round(model_output, 2)}")

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


def plot_CDAM_reg(original_img, attention_map, cdam_maps, preds, save_name:Optional[str]=None):
    """Using matplotlib, plot the original image and the relevance maps"""
    maps2plot=[]
    target_names=[]
    for key in cdam_maps.keys():
        maps2plot.append(cdam_maps[key])
        target_names.append(key)

    for map_, title in zip(maps2plot, target_names):
        # Biomarker Regression:
        fig, axs = plt.subplots(1, 4, figsize=(12, 4), layout='constrained')

        ## Original img:
        axs[0].imshow(original_img[:,:,0], cmap='gray')
        axs[0].set_title("Original")
        axs[0].tick_params(axis='both', 
                           which='both', 
                           bottom=False, 
                           left=False,
                           labelbottom=False,
                           labelleft=False
                          )
        
        ## Attention map:
        axs[1].imshow(attention_map, cmap=get_cmap(attention_map))
        axs[1].set_title("Attention Map")
        axs[1].tick_params(axis='both', 
                           which='both', 
                           bottom=False, 
                           left=False,
                           labelbottom=False,
                           labelleft=False
                          )
                      
        ## CDAM Map:
        cdam = axs[2].imshow(map_, cmap=get_cmap(map_))
        axs[2].set_title(title)
        axs[2].tick_params(axis='both', 
                           which='both', 
                           bottom=False, 
                           left=False,
                           labelbottom=False,
                           labelleft=False
                          )
        fig.colorbar(cdam, ax=axs[2], shrink=0.6)

        ## Histogram
        sns.histplot(ann_df,
                     x=title.lower(),
                     kde=True,
                     bins=14,
                     stat="percent",
                     ax=axs[3])
        axs[3].axvline(x=preds[title],
                          color='red',
                          linestyle='--',
                          linewidth=2,
                          label=r'$\hat{y}$'
                         )
        title = (title + "=" + str(preds[title]) + " [mm]") if title == "Diameter" else title + "=" + str(preds[title])
        axs[3].set_title(title)
        axs[3].legend()
        
        plt.suptitle("CDAM maps for biomarkers")
        plt.show()
    
        if save_name:
            if not os.path.exists("relevance_maps"):
                os.makedirs("relevance_maps")
            plt.savefig(f"relevance_maps/{save_name}""_"+title, format="png", transparent=True, bbox_inches='tight')
    return None


def plot_ori_att_reg(original_img, attention_map):
    fig, axs = plt.subplots(1, 2, figsize=(5, 10))
    axs[0].imshow(original_img[:,:,0], cmap='gray')
    axs[0].set_title("Original")
    axs[0].tick_params(axis='both',
                       which='both',
                       bottom=False,
                       left=False,
                       labelbottom=False,
                       labelleft=False
                      )
    axs[1].imshow(attention_map, cmap=get_cmap(attention_map))
    axs[1].set_title("Attention Map")
    axs[1].tick_params(axis='both',
                       which='both',
                       bottom=False,
                       left=False,
                       labelbottom=False,
                       labelleft=False
                      )

    return None