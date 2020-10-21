import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import torchvision.transforms.functional as TF




class VideoDataset(Dataset):
    '''Video Dataset'''
    
    def __init__(self, path='./', data_type='world', nframes=10, patch_size=32, norm=True, transform=None):
        '''
        Params:
            path (str): path to frame folder
            data_type (str): either 'world' or 'head', or 'retinal'
            transform: type of transform 
        '''
        
        if not data_type in ('ducks','world','head','retinal'):
            print(f'Unrecognized task: {data_type}, should be one of (world, head, or retinal)')
        self.data_type = data_type
        self.nframes = nframes
        self.patch_size = patch_size
        self.transform = transform
        self.norm=norm
        
        if data_type == 'ducks':
            self.path = path
        else:
            self.path = os.path.join(path, data_type)
        
        self.png_paths = [os.path.join(self.path,name) for name in os.listdir(self.path) if '.png' in name]
        self.range = len(self.png_paths) - nframes

            
        def meanstdnorm(tensor):
            mean = torch.mean(tensor)
            std = torch.std(tensor)
            tensor = (tensor - mean) / std
            return(tensor)
        
        def randcrop(tensor):
            _, _, xshape, yshape = tensor.shape
            randx = np.random.randint(0,xshape-self.patch_size)
            randy = np.random.randint(0,yshape-self.patch_size)
            tensor = tensor[:,:,randx:randx+self.patch_size,randy:randy+self.patch_size]
            return(tensor)
        
        self.normfunc = meanstdnorm
        self.cropfunc = randcrop

    def __len__(self):
        return self.range
    
    def __getitem__(self, idx):
        #print(self.png_paths[idx+1])
        #frames = []
        #for i in range(self.nframes):
        #    frame = Image.open(self.png_paths[idx+i])
        #    frames.append(TF.to_tensor(frame))
        frames = torch.stack([TF.to_tensor(Image.open(self.png_paths[idx+i])).squeeze(0) for i in range(self.nframes)]).unsqueeze(0)
        sample = {'frames': frames}
        sample['frames'] = self.cropfunc(sample['frames'])
        if self.norm:
            sample['frames'] = self.normfunc(sample['frames'])
        if self.transform:
            sample['frames'] = self.transform(sample['frames'])
        return(sample)