import shutil
import torch
import wandb
import os



def load_ckp(wandb_base_path, name, epoch):      

    # load check point
    path = "results/"+name + "/" + name + "_" + str(epoch) + ".pth"
    load_path = os.path.join(wandb_base_path, path)
    wandb.restore(path)
    checkpoint = torch.load(load_path)

    return checkpoint

def load_ckp_direct(wandb_base_path, name, epoch):      

    # load check point
    load_path = "/content/drive/MyDrive/driving_uncertainty/models/image-dissimilarity/best_net_baseline_void_prior_spadedecoder_mult_3.pth"
    checkpoint = torch.load(load_path)
    return checkpoint
