# Imports
import torch
import pickle
import os
from types import MethodType
from typing import Literal, Tuple, Dict
from Biomarker_Model import Biomarker_Model
from End2End_Model import End2End_Model
from my_utils.att_cdam_utils import get_maps
from my_utils.loading_data_utils import load_img
from my_utils.plot_utils import plot_res_class, plot_CDAM_reg, plot_ori_att_reg

# Globals
### Normalizing factors:
current_directory = os.getcwd()
with open(current_directory+"/dataset/splitted_sets/fitted_factors.pkl", 'rb') as f:
    fitted_factors = pickle.load(f)

# MAIN

def cdam_pipeline(NODULE: str, 
                   SLICE: int,
                   NODULE_VIEW: Literal["axial", "coronal", "sagittal"],
                   TASK: Literal["Regression", "Classification"],
                   MODEL_BCKB: Literal["dino_vits8", "dino_vitb8", 
                  "dino_vits16", "dino_vitb16", "vit_b_16", "vit_l_16", "dinov2_vits14_reg", "dinov2_vitb14_reg"],
                   CKPT_VERSION: int,
                   FOLD: int=1)->Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Function running CDAM pipeline.
    """
    # Translating arguments:
    ckpt_versions = {1:"",
                     2:"-v1",
                     3:"-v2"}
    patch_sizes = {"dino_vits8":8,
                   "dino_vitb8":8,
                   "dino_vits16":16,
                   "dino_vitb16":16,
                   "vit_b_16":16,
                   "dinov2_vits14_reg":14,
                   "dinov2_vitb14_reg":14
                  }
    PATCH_SIZE = patch_sizes[MODEL_BCKB]

    # Retrieving model numer:
    E2E_model_numers = {"dino_vits8":39,
                   "dino_vitb8":38,
                   "dino_vits16":32,
                   "dino_vitb16":35,
                   "vit_b_16":9,
                    "dinov2_vits14_reg":4,
                    "dinov2_vitb14_reg":1
                   }
    biom_model_numers = {"dino_vits8":20,
                   "dino_vitb8":22,
                   "dino_vits16":21,
                   "dino_vitb16":23,
                   "vit_b_16":24,
                    "dinov2_vits14_reg":2,
                    "dinov2_vitb14_reg":4
                   }
    if TASK == "Classification":
        MODEL_NR = E2E_model_numers[MODEL_BCKB]
    else:
        MODEL_NR = biom_model_numers[MODEL_BCKB]
    
    # Loading model and registering hooks:
    device = "cuda" if torch.cuda.is_available() else "cpu"    
    if TASK == "Regression":
        biom_model = Biomarker_Model.load_from_checkpoint(current_directory + f"/ckpt/Biomarkers/{MODEL_BCKB}_{MODEL_NR}_{FOLD}{ckpt_versions[CKPT_VERSION]}.ckpt").to(device).eval()
        model = biom_model
    else:
        E2E_model = End2End_Model.load_from_checkpoint(current_directory + f"/ckpt/End2End/{MODEL_BCKB}_{MODEL_NR}_{FOLD}{ckpt_versions[CKPT_VERSION]}.ckpt").to(device).eval()
        model = E2E_model
    
    ## Creating hooks:
    activation = {}
    def get_activation(name):
        """
        Function to extract activations before the last MHSA layer.
        """
        def hook(model, input, output):
            activation[name] = output[0].detach()
        return hook
    
    grad = {}
    def get_gradient(name):
        """
        Function to extract gradients.
        """
        def hook(model, input, output):
            grad[name] = output
        return hook

    if MODEL_BCKB in ["vit_b_16", "vit_l_16"]:
        # Slight modifications to the original architecture:
        # https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py
        # To extract attention weights we need to overwrite the original forward method in the last encoder.

        def forward_new(self, input: torch.Tensor):
            torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
            x = self.ln_1(input)
            # Original: x, _ = self.self_attention(x, x, x, need_weights=False)
            x, attn_output_weights = self.self_attention(x, x, x, need_weights=True, average_attn_weights=False)
            x = self.dropout(x)
            x = x + input
        
            y = self.ln_2(x)
            y = self.mlp(y)
            return x + y
        
        # Overwriting:
        model.backbone.encoder.layers.encoder_layer_11.forward = MethodType(forward_new, model.backbone.encoder.layers.encoder_layer_11)
        
    # DINO backbone by default has method to obtain attention weights.
    # But classic ViT needs registering a special hook as below.
    last_selfattn = {}
    def get_last_selfattn(name):
        """
        Function to extract attention weights from the last MHSA.
        """
        def hook(model, input, output):
            last_selfattn[name] = output[1].detach()
        return hook
    
    ## Registering hooks:
    ### We store the: 
    #### i) normalized activations entering the last MHSA layer.
    #### ii) gradients wrt the normalized inputs to the final attention layer.
    ### Both are required to compute CDAM score.
    ### We don't need to register hook on MHSA to extract attention weights in case of DINO, because DINO backbone has it already implemented.
    if MODEL_BCKB in ["dino_vits8", "dino_vitb8", "dino_vits16", "dino_vitb16", "dinov2_vits14_reg", "dinov2_vitb14_reg"]:
        final_block_norm1 = model.backbone.blocks[-1].norm1
    elif MODEL_BCKB in ["vit_b_16", "vit_l_16"]:
        final_block_norm1 = model.backbone.encoder.layers[-1].ln_1
        
    activation_hook = final_block_norm1.register_forward_hook(
        get_activation("last_att_in"))
    grad_hook = final_block_norm1.register_full_backward_hook(
        get_gradient("last_att_in"))
    
    if MODEL_BCKB in ["vit_b_16", "vit_l_16"]:
        # Additional hook in case of classic ViT
        final_block_selfattn = model.backbone.encoder.layers.encoder_layer_11.self_attention
        last_selfattn_hook = final_block_selfattn.register_forward_hook(
            get_last_selfattn("last_selfattn"))

    # Taking mean and std from fitted factors
    MEAN, STD, SCALER = fitted_factors[f"fold_{FOLD}"]

    # Loading image from repository:
    img, original_img = load_img(crop_path=current_directory + f"/dataset/crops/{NODULE}", 
                                 crop_view=NODULE_VIEW, 
                                 slice_=SLICE,
                                 MEAN=MEAN,
                                 STD=STD, 
                                 device=device)

    # Model inference:
    model = model.to(device)
    attention_map, CDAM_maps, model_output = get_maps(model, MODEL_BCKB, img, grad, activation, last_selfattn, TASK, patch_size=PATCH_SIZE, scaler=SCALER, clip=True)

    return (original_img, attention_map, CDAM_maps, model_output)


def call_CDAM(NODULE, SLICE, NODULE_VIEW, TASK, MODEL_BCKB, CKPT_VERSION, FOLD):
    original_img, attention_map, CDAM_maps, model_output = cdam_pipeline(NODULE=NODULE,
                  SLICE=SLICE,
                  NODULE_VIEW=NODULE_VIEW,
                  TASK=TASK,
                  MODEL_BCKB=MODEL_BCKB, 
                  CKPT_VERSION=CKPT_VERSION,
                  FOLD=FOLD
                 )
    if TASK == "Classification":
        plot_res_class(original_img=original_img, maps=[attention_map, CDAM_maps], model_output=model_output)
    elif TASK == "Regression":
        plot_CDAM_reg(original_img, attention_map, CDAM_maps, model_output)