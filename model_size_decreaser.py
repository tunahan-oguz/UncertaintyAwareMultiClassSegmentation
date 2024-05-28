import os
import torch
import yaml
import train_app.utils as utils
import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-p", type=str)
args = parser.parse_args()

ckpt = torch.load(args.p)
torch.save(ckpt["state_dict"], args.p.replace(".ckpt", ".pth"))
    