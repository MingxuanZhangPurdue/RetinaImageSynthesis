import glob
import random
import os
import numpy as np

from torch.utils.data import Dataset
from PIL import Image


class ImageDataset(Dataset):
    def __init__(self, root, mode="train", transforms_A=None):
        
        self.transforms_A = transforms_A
        
        if mode == "train":
            self.A_files = sorted(glob.glob(os.path.join(root, "train", "A") + "/*.*"))
         
        elif mode == "val":
            self.A_files = sorted(glob.glob(os.path.join(root, "val", "A") + "/*.*"))
            

    def __getitem__(self, index):

        img_A = Image.open(self.A_files[index % len(self.A_files)])
        
        if self.transforms_A != None:
            img_A = self.transforms_A(img_A)
            
        return img_A

    def __len__(self):
        return len(self.A_files)
