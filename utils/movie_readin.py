import numpy as np
import imageio
from scipy import stats
import torch
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

def readMovMp4(path, maxframes=4096):
    d = []
    reader = imageio.get_reader(path,'ffmpeg')
    for i, im in enumerate(reader):
        d.append(im)
        if(i>maxframes):
            break
    print(f'Full Movie Shape: {np.shape(np.array(d))}')
    return np.array(d)
     

def get_movie(movie_fpath, pixel_patch_size, maxframes, frame_patch_size=128,
             normalize_patch=False, normalize_movie=True, encoding='mp4', 
             crop=False):

    if(encoding=='mp4'):
        # info about movie
        fps = 120
        degrees = 70 #degrees subtended by camera
        #read in movie
        #maxframes = fps * 1 # 1 seconds
        m = readMovMp4(movie_fpath, maxframes)
        #np.shape(m)
    # crop out the edges of the movie - they may be a bit blurry/distorted due to moment lens abberation    
    if(crop):
        m = m[:,100:-100,300:-300]
        
    nframes, frameh, framew, ncolorchannels = np.shape(m)
    ppd = frameh/degrees

    # remove color channel:
    m = np.mean(m,axis=3)

    #convert to degrees
    framewdeg = framew/ppd 
    framehdeg = frameh/ppd
    
    #sampling rate
    deltawdeg = 1./ppd
    deltahdeg = 1./ppd 
    deltathz = 1./fps

    #normalize_movie
    if(normalize_movie):
        print('normalizing movie...')
        m = m - np.mean(m)
        m = m/(np.std(m)+0.1)
    
    #make patches
    if (pixel_patch_size != None):
        htiles = np.int(np.floor(frameh/pixel_patch_size))
        wtiles = np.int(np.floor(framew/pixel_patch_size))
        ftiles = np.int(np.floor(nframes/frame_patch_size))  
        
        print('making patches...')
        m = np.asarray(np.split(m[:,:,0:pixel_patch_size*wtiles], wtiles,2)) # tile column-wise
        m = np.asarray(np.split(m[:,:,0:pixel_patch_size*htiles], htiles,2)) #tile row-wise
        m = np.asarray(np.split(m[:,:,0:frame_patch_size*ftiles], ftiles,2)) #tile time-wise
        m = np.transpose(np.reshape(np.transpose(m,(4,5,3,0,1,2)),
                                   (pixel_patch_size, pixel_patch_size, frame_patch_size,-1)),(3,0,1,2)) #stack tiles together
    
    #normalize patches
    if(normalize_patch):
        print('normalizing patches...')
        #m = m - np.mean(m,axis=(1,2,3),keepdims=True)
        #m = m/np.std(m,axis=(1,2,3),keepdims=True)
        
        #normalize each full patch - divide by geom norm and log transform 
        #invn = 1/np.prod([m.shape[1],m.shape[2],m.shape[3]])
        #m = np.nan_to_num(np.log(m))
        geom_means = stats.mstats.gmean(m+0.01,axis=(1,2,3))[:,np.newaxis,np.newaxis,np.newaxis]
        #print(geom_means.shape)
          #print(np.min(geom_means))
        m = m - np.nan_to_num(geom_means)
        
        m = m/(np.std(m)+0.1)
        
    #transpose & shuffle
    m = np.transpose(m, (0, 3, 1, 2)) #change axis to [batchsize, frame_patch_size, x_patchsize, y_patchsize]
    np.random.shuffle(m)
        
    return(m)

# make it a Pytorch dataset (inherits from Dataset)
class NaturalMovieDataset(data.Dataset):
    """Dataset of Stationary Naural Movies"""
    
    def __init__(self, movie_filepath, pixel_patchsize, frame_patchsize, maxframes,
                     normalize_patch=True, normalize_movie=False, encoding='mp4'):
        """
        Args:
            movie_filepath (string): Path to the movie file
            pixel_patchsize (int): Number of pixels on the edge of a patch
            frame_patchsize (int): Number of frames in the movie
        """
        self.movies = get_movie(movie_filepath, pixel_patchsize, maxframes,
                                frame_patchsize, normalize_patch=normalize_patch,
                                normalize_movie=normalize_movie, encoding='mp4',
                                crop=True)

    def __len__(self):
        return len(self.movies)

    def __getitem__(self, idx):
        movie = self.movies[idx,:,:,:]
        movie = torch.from_numpy(movie)
        sample = Variable(movie)
        return sample
    

def CreateNatMovieDataset(framerate=240, patchsize=16, seconds=10, folder=None):
    '''
    Create a torch compatable dataset of natural movie patches
    
    Parameters:
        framerate (int): frames per second of movie (120 or 240). Determines the dataset used
        patchsize (int): number of pixels on an edge of a patch
        seconds (float): number of seconds each patch corresponds to. Size of patch will be seconds*framerate.
        
    Returns:
        dataset: An instance of NaturalMovieDataset with movie patches
    '''
    
    # choose a folder to read in from if not specified
    if(folder == None):
        folder = '/home/vasha/research/datasets/stationary_motion/pixel2xlmomentlens/full_framerate'
    
    
    