import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

#recode image tags from 22 to 13
def recode_tags(sem_image):
    recode_dict = {0:0,1:1,2:2,3:3,4:4,5:5,6:6,7:7,8:8,9:9,
                    10:10,11:11,12:12,13:0,14:3,15:1,16:3,17:2,18:5,19:3,20:4,21:3,22:9
                    }
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

def generate_semantic_im(RGB_image):
    new_obs = RGB_image.reshape(1,3,128,128)
    out,_ = model(new_obs)
    sample = out.cpu().argmax(dim=1)
    print(sample.shape)
    pic = replace(sample.numpy())
    return Image.fromarray(pic,'RGB')

#Dataset classes for loading images
class CustomImageDataset(Dataset):
    def __init__(self, weather, town, test=False , transform=None, target_transform=None):
        dirt = './Datasets/'+weather+'/'+town
        if test:
            dirt = dirt+'/test'
        
        self.sem_dir = dirt+'/Semantic'
        self.rgb_dir = dirt+'/RGB'
        self.transform = transform
        self.target_transform = target_transform
        self.names = os.listdir(self.rgb_dir)

    def __len__(self):
        return len(os.listdir(self.sem_dir))

    def __getitem__(self, idx): 
        img_path = os.path.join(self.rgb_dir, self.names[idx])
        image = read_image(img_path)
        label_name = self.names[idx].split('.')[0]+'.npy'
        label = np.load(os.path.join(self.sem_dir, label_name))
        label = torch.tensor(label).permute(2,0,1)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


#Image Transformers
class CropResizeTransform:
    def __init__(self, top, left, width, height, size):
        self.top = top
        self.left = left
        self.width = width
        self.height = height
        self.size = size

    def __call__(self, x):
        return TF.resized_crop(x,self.top,self.left,self.width,self.height,self.size)

class Hflip:
    def __init__(self):
        pass

    def __call__(self, x):
        return TF.hflip(x)


def AE_initalize_weights(layer):
    if isinstance(layer, torch.nn.Linear) or isinstance(layer,torch.nn.Linear):
        nn.init.kaiming_uniform_(layer.weight.data,nonlinearity='relu')