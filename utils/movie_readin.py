import numpy as np
import imageio
from scipy import stats

def readMovMp4(path, maxframes=4096):
    d = []
    reader = imageio.get_reader(path,'ffmpeg')
    for i, im in enumerate(reader):
        d.append(im)
        if(i>maxframes):
            break
    print(np.shape(np.array(d)))
    return np.array(d)
     

def get_movie(movie_fpath, pixel_patch_size, maxframes, frame_patch_size=120,
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
    # crop out the edges of the movie - they may be a bit blurry/distorted    
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
        print(np.min(geom_means))
        m = m - np.nan_to_num(geom_means)
        
        m = m/(np.std(m)+0.1)
        
    #transpose & shuffle
    m = np.transpose(m, (0, 3, 1, 2)) #change axis to [batchsize, frame_patch_size, x_patchsize, y_patchsize]
    np.random.shuffle(m)
        
    return(m)
