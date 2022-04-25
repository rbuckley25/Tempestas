import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import random
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


from torch.utils.data import DataLoader, Subset 
from torch.utils.data import ConcatDataset
from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision.utils import save_image

town03_data = CustomImageDataset('ClearNoon','Town03',test=False)
town03_test_data = CustomImageDataset('ClearNoon','Town03',test=True)
town04_data = CustomImageDataset('ClearNoon','Town04',test=False)
town04_test_data = CustomImageDataset('ClearNoon','Town04',test=True)
train_data = ConcatDataset([town03_data,town04_data])
test_data = ConcatDataset([town03_test_data,town04_test_data])



def recode_tags(sem_image,recode_dict):
    for value in recode_dict.keys():
        sem_image[sem_image==value] = recode_dict[value]
    
    return sem_image

tag_convert_dict = {0:[70,130,180],
                   1:[70,70,70],
                   2:[100,40,40],
                   3:[55,90,80],
                   4:[220,20,60],
                   5:[153,153,153],
                   6:[157,234,50],
                   7:[128,64,128],
                   8:[244,35,232],
                   9:[107,142,35],
                   10:[0,0,142],
                   11:[102,102,156],
                   12:[220,220,0],
                   13:[70,130,180],
                   14:[81,0,81],
                   15:[150,100,100],
                   16:[230,150,140],
                   17:[180,165,180],
                   18:[250,170,30],
                   19:[110,190,160],
                   20:[170,120,50],
                   21:[45,60,150],
                   22:[145,170,100],
                  }

def generate_semantic_im(RGB_image):
    new_obs = RGB_image
    new_obs = new_obs.permute(2,0,1).reshape(1,3,128,128)
    out,_ = model(new_obs)
    sample = out.cpu().argmax(dim=1)
    pic = replace(sample.numpy())
    return Image.fromarray(pic,'RGB')

def replace(a):
    a = a.reshape(128,128)
    pic = np.zeros((128,128,3),dtype='uint8')
    for x, y in np.ndindex(a.shape):
        value = a[x,y]
        RGB_values = tag_convert_dict[value]
        pic[x,y,0] = RGB_values[0]
        pic[x,y,1] = RGB_values[1]
        pic[x,y,2] = RGB_values[2]
    return pic

def main():
    num = random.randint(0,6000)
    data = train_data.__getitem__(num)
    imgs = []
    org = data[0]
    sem = replace(data[1])

    augmenter = T.TrivialAugmentWide()
    imgs = [augmenter(orig_img) for _ in range(4)]

    imgs.append(augmenter(Image.fromarray(org.numpy().transpose(1,2,0))))
    imgs.append(augmenter(Image.fromarray(sem.reshape(128,128,3))))
    imgs.append(augmenter(generate_semantic_im(data[0])))
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

if __name__ == "__main__":
    main()