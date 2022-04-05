import numpy as np
from itertools import count
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from torchvision.io import read_image

import glob
import carla
import gym_carla

#########
# Pixel Recode and Image Generation Function
#########
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

def generate_semantic_im(RGB_image,model):
    new_obs = RGB_image.reshape(1,3,128,128)
    out,_ = model(new_obs)
    sample = out.cpu().argmax(dim=1)
    pic = replace(sample.numpy())
    return Image.fromarray(pic,'RGB')



#########
# Auto Encoder Dataset Loader
#########
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


class GANImageDataset(Dataset):
    def __init__(self, weather, town, model, transform=None, target_transform=None):
        self.dir = './Datasets/'+weather+'/'+town+'/'+model+'/test_latest/images'
        self.transform = transform
        self.target_transform = target_transform
        self.real = glob.glob(self.dir+'/*real.png')


    def __len__(self):
        return len(list(self.real))

    def __getitem__(self, idx): 
        img_path = self.real[idx]
        image = read_image(img_path)
        label_name_split = self.real[idx].split('_')
        label_name = label_name_split[0]+'_'+label_name_split[1]+'_fake.png'
        label = read_image( label_name)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


#########
# Image Transformations
#########
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




#########
# Model Weight Initalization
#########

def AE_initalize_weights(layer):
    if isinstance(layer, torch.nn.Linear) or isinstance(layer,torch.nn.Linear):
        nn.init.kaiming_uniform_(layer.weight.data,nonlinearity='relu')


#########
# Training Function
#########

def run_AE_demo(num_episodes, DQN, AE, Base, env, device):
    env.use_fixed = 'H'
    env.route_idx = 1
    input_size = 73
    results = np.zeros((num_episodes,3))
    for i_episode in range(num_episodes):
        rewards = 0
        # Initialize the environment and state
        obs = env.reset()
        ego_location = env.ego.get_location()
        ego_dir = gym_carla.envs.misc.get_lane_dis(env.waypoints,ego_location.x,ego_location.y)
        #pos gets a distanc d and array w which has to be seperated out in below line
        ego_pos = np.asarray((ego_dir[0],ego_dir[1][0],ego_dir[1][1]),dtype=np.float32)
        state = np.concatenate((ego_pos,np.zeros(6)))
        state = torch.tensor(state).reshape(1,9,1,1)

        new_obs = torch.tensor(obs['camera'])
        new_obs = new_obs.permute(2,0,1).reshape(1,3,128,128)
        
        _,latent_space = AE(new_obs)
        state = torch.cat((state,latent_space.cpu()),1).reshape(1,input_size)
        # Resize, and add a batch dimension (BCHW)
        for t in count():
            # Select and perform an action
            
            with torch.no_grad():
                action = DQN(state.float()).argmax().view(1,1)
                obs, reward, done, info  = env.step(action.item())

                new_obs = torch.tensor(obs['camera'])
                new_obs = new_obs.permute(2,0,1).reshape(1,3,128,128)
                _,latent_space = AE(new_obs)

                sem = Base.decode(latent_space).detach().cpu().argmax(dim=1)
                sem = replace(sem.numpy().reshape(1,128,128).transpose(1,2,0))
                env.show_images(sem)

                rewards += reward
                reward = torch.tensor([reward], device=device)

                #pos gets a distanc d and array w which has to be seperated out in below line
                pos = np.asarray((info['position'][0],info['position'][1][0],info['position'][1][1]))
                ang = np.asarray(info['angular_vel'])
                acc = np.asarray(info['acceleration'])
                steer = np.asarray(info['steer'])
                next_state = np.concatenate((pos, ang, acc, steer), axis=None)
                
                
                info_state = torch.tensor(next_state).reshape(1,9,1,1)
                next_state = torch.cat((info_state,latent_space.cpu()),1).reshape(1,input_size)

            state = next_state

            if done:
                results[i_episode,0] = rewards
                results[i_episode,1] = t
                if t  == env.max_time_episode:
                    results[i_episode,2] = 1
                else:
                    results[i_episode,2] = 0
                print(results)
                print('########')
                break