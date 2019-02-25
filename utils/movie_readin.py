import numpy as np
import imageio
from scipy import stats
import torch
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import os
import cv2
import pickle as pickle

def script_to_fix_frame0():
    folders = listFolders('/data/stationary_motion/pixel2xlmomentlens/pngs')
    print(folders)
    for folder in folders:
        path = f'/data/stationary_motion/pixel2xlmomentlens/pngs/{folder}'
        print(path)
        frame0 = cv2.imread(os.path.join(path,'frame_0.png'))
        print(frame0.shape)
        frame1 = cv2.imread(os.path.join(path,'frame_1.png'))
        print(frame1.shape)
        #newframe = frame0[100:-100,300:-300,:]
        #print(newframe.shape)
        #uncommment to write. may need to change permissions for success
        #result = cv2.imwrite(os.path.join(path,'frame_0.png'),newframe)
        #print(result)

def readMovMp4(path, maxframes=4096, offset_frames=0):
    d = []
    reader = imageio.get_reader(path,'ffmpeg')
    # don't append until we reach offset frames; break after we hit maxframes after that
    #print (maxframes+offset_frames)
    for i, im in enumerate(reader):
        if(i > offset_frames):
            d.append(im)
        elif(i>maxframes+offset_frames):
            break
    return np.array(d)
     
def readMovPng(path, maxframes, offset_frames=0):
    pngpath = os.path.join(path, f'frame_0.png')
    framew, frameh, framech = np.shape(cv2.imread(pngpath))
    d = np.zeros((maxframes, framew, frameh, framech))
    #offset_frames = offset_frames+1 #zeroth frame has now been cropped
    fnum = offset_frames
    while(fnum < maxframes+offset_frames):
        pngpath = os.path.join(path, f'frame_{fnum}.png')
        im = cv2.imread(pngpath)
        #print(im.shape)
        #d.append(im)
        d[fnum-offset_frames] = im
        fnum +=1
    return np.array(d)
     

def get_movie(movie_fpath, pixel_patch_size, 
              maxframes, frame_size=128,normalize_patch=False, 
              normalize_movie=False, encoding='png', 
              crop=False, offset_frames=0, verbose=False):
    # info about movie
    fps = 120
    degrees = 70 #degrees subtended by camera
    if(encoding=='mp4'):
        m = np.asarray(readMovMp4(movie_fpath, maxframes, offset_frames))
    elif(encoding=='png'):
        m = np.asarray(readMovPng(movie_fpath, maxframes, offset_frames))
        #png formatted movies are already cropped.
        crop=False
        
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
        if(verbose):
            print('normalizing movie...')
        m = m - np.mean(m)
        m = m/(np.std(m)+0.1)
    
    #make patches
    if (pixel_patch_size != None):
        htiles = np.int(np.floor(frameh/pixel_patch_size))
        wtiles = np.int(np.floor(framew/pixel_patch_size))
        ftiles = np.int(np.floor(nframes/frame_size))  
        
        if(verbose):
            print('making patches...')
        m = np.asarray(np.split(m[:,:,0:pixel_patch_size*wtiles], wtiles,2)) # tile column-wise
        m = np.asarray(np.split(m[:,:,0:pixel_patch_size*htiles], htiles,2)) #tile row-wise
        m = np.asarray(np.split(m[:,:,0:frame_size*ftiles], ftiles,2)) #tile time-wise
        m = np.transpose(np.reshape(np.transpose(m,(4,5,3,0,1,2)),
                                   (pixel_patch_size, pixel_patch_size, frame_size,-1)),(3,0,1,2)) #stack tiles together
    
    #normalize patches
    if(normalize_patch):
        if(verbose):
            print('normalizing patches...')
        #m = m - np.mean(m,axis=(1,2,3),keepdims=True)
        #m = m/np.std(m,axis=(1,2,3),keepdims=True)
        
        #normalize each full patch - divide by geom norm and log transform 
        #invn = 1/np.prod([m.shape[1],m.shape[2],m.shape[3]])
        #m = np.nan_to_num(np.log(m))
        geom_means = stats.mstats.gmean(m+0.01,axis=(1,2,3))[:,np.newaxis,np.newaxis,np.newaxis]
        m = m - np.nan_to_num(geom_means)
        
        m = m/(np.std(m)+0.1)
        
    #transpose & shuffle
    m = np.transpose(m, (0, 3, 1, 2)) #change axis to [batchsize, frame_size, x_patchsize, y_patchsize]
    np.random.shuffle(m)
        
    return(m)

def get_pkl_patch_movie(filepath):
    '''
    Read in a set of movie patches stored as pkl files
    Parameters:
        filepath (str): Path to folder containing patches
    Retrns:
        pathes: 
    '''
    patches_file = os.path.join(filepath, f'patchset_0.pkl')
    with open(patches_file, 'rb') as pickle_file:
        patches_dict = pickle.load(pickle_file)
    patches = patches_dict['patches'].astype(np.float32)
    return(patches)
    
class NaturalMovieDataset(data.Dataset):
    """ 
    Dataset of Stationary Natural Movies Combined from many movies.
    Reads in from one or more pkl files.
    
    TODO: Make this work for multiple pickle files!
    """
    def __init__(self, file_dir):
        """
        Args:
            file_dir (str): Path to the folder containing .pkl files stored in format patchset_x.pkl
        """

        self.file_dir = file_dir
        
        
    def __len__(self):
        """
        How many movie patches are present?
        """
        #npklfiles = len([name for name in os.listdir(self.file_dir) if '.pkl' in name])
        #for now, just put the length of first one
        
        #patches_file = os.path.join(self.file_dir, f'patchset_0.pkl')
        #with open(patches_file, 'rb') as pickle_file:
        #    patches_dict = pickle.load(pickle_file)
        #patches = patches_dict['patches']
        #npatches = np.shape(patches)[0]
        #print(np.shape(patches))
        return(npatches)
    
    def getallitems(self):
        patches_file = os.path.join(self.file_dir, f'patchset_0.pkl')
        with open(patches_file, 'rb') as pickle_file:
            patches_dict = pickle.load(pickle_file)
        patches = patches_dict['patches']
        return(patches)
    
    def __getitem__(self, idx):
        patches = self.getallitems()
        patches = patches[idx]
        return(patches)

    
    
class NaturalMovieDatasetOneMovie(data.Dataset):
    """Dataset of Stationary Naural Movies taken from one movie"""
    
    def __init__(self, filepath, pixel_patchsize, frame_patchsize, maxframes,
                     normalize_patch=True, normalize_movie=False, encoding='patch'):
        """
        Args:
            movie_filepath (string): Path to the movie file
            pixel_patchsize (int): Number of pixels on the edge of a patch
            frame_patchsize (int): Number of frames in the movie
        """
        if(encoding=='mp4'):
            self.movies = get_movie(filepath, pixel_patchsize, maxframes,
                                    frame_patchsize, normalize_patch=normalize_patch,
                                    normalize_movie=normalize_movie, encoding='mp4',
                                    crop=True)
    def __len__(self):
        return len(self.movies)

    def __getitem__(self, idx):
        movie = self.movies[idx,:,:,:]
        movie = torch.from_numpy(movie)
        #sample = Variable(movie)
        return movie
    
    
def listFiles(folder, searchterm=''):
    '''
    Create a list of all the files of a given category.
    Option to search within files for a term (ex framerate)
    
    Parameters:
        folder (str): name of folder in which files are stored.
        searchterm (str): filter on filenames (for example framerate)
    Returns:
        filelist (list of str): list of filenames filtered by searchterm
    '''
    # walk through folder and find moviefiles
    file_list = []
    for root, dirs, files in os.walk(folder, topdown=False):
        for name in files:
            # check if framerate value is in name
            if(str(searchterm) in name):
                fname = os.path.join(root, name)
                file_list.append(fname)
    return(file_list)

def listFolders(folder, searchterm=''):
    '''
    Create a list of all the folders of a given category.
    Option to search within files for a term (ex framerate)
    
    Parameters:
        folder (str): name of folder in which files are stored.
        searchterm (str): filter on filenames (for example framerate)
    Returns:
        filelist (list of str): list of filenames filtered by searchterm
    '''
    # walk through folder and find moviefiles
    folder_list = []
    for root, dirs, files in os.walk(folder, topdown=False):
        for adir in dirs:
            if(searchterm in adir):
                folder_list.append(adir)
            #dirname = os.path.join(root,adir)
            #folder_list.append(dirname)

    return(folder_list)


def createNatMoviePatches(framerate=120, patchsize=16, seconds=2, read_folder=None, write_folder=None, patches_per_file=2000):
    '''
    Create a dataset of natural movie patches
    
    Parameters:
        framerate (int): frames per second of movie (120 or 240). Determines the dataset used
        patchsize (int): number of pixels on an edge of a patch
        seconds (float): number of seconds each patch corresponds to. Size of patch will be seconds*framerate.
        read_folder (str): folder where full movies are stored
        write_folder (str): folder where patched movies are stored
        frames_per_file (int): number of total frames per file write (limits file size).
        
    Returns:
        dataset: An instance of NaturalMovieDataset with movie patches
    '''
    
    print('Creating Movie Patches....')
    # choose a folder to read in from if not specified
    if(read_folder == None):
        read_folder = '/data/stationary_motion/pixel2xlmomentlens/pngs'
    # movies are 240 frames (2 seconds) long *for now*
    max_framenumber = framerate*seconds
    offset_list = np.arange(0, max_framenumber, framerate*seconds)
    
    # choose a folder to write to if not specified
    if(write_folder == None):
        write_folder = f'/data/stationary_motion/pixel2xlmomentlens/patches/patches_{patchsize}px_{seconds}s_{framerate}fps'
    try:
        os.stat(write_folder)
    except:
        os.mkdir(write_folder)
    
    # get list of movies
    #images_list = listFiles(read_folder, searchterm=str(framerate))
    movies_list = listFolders(read_folder, searchterm=str(framerate))
    print(movies_list)
    #print(movies_list)
    # list of patches
    #patches_list = []
    # list of filenumbers
    filenames = []
    filenum = 0
    reads_counter = 0
    # loop through images, pulling off nsecond patches at a time
    for offset in offset_list:
        for movie_dir in movies_list:
            print(movie_dir, end=', ')
            movie_path = os.path.join(read_folder, movie_dir)
            # check frames counter
            if(reads_counter >= patches_per_file):
                # write to disk
                np.random.shuffle(np.asarray(patches))
                mydict = {'patches': patches}
                fname = f'patchset_{filenum}.pkl'
                filetowrite = os.path.join(write_folder, fname)
                with open(filetowrite, 'wb') as handle:
                    pickle.dump(mydict, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    #pickle.dump(mydict, open(filetowrite, 'wb'))#write_patch(patches, filenum, write_folder)
                filenames.append(fname)
                # reset lists
                reads_counter = 0
                filenum += 1
            # if we haven't hit our size limit, read in a set of patches

            # print(framerate*seconds, offset)
            m = get_movie(movie_path, patchsize, 
                          maxframes = framerate*seconds,
                          frame_size = framerate*seconds,
                          normalize_patch=False,
                          normalize_movie=False,
                          encoding='png', crop=True,
                          offset_frames=offset,
                          verbose=False)
            if(reads_counter==0):
                patches = m
            else:
                patches = np.vstack((patches,m))
                
            reads_counter += 1

    # Write last patchlist
    print(f'Writing {np.shape(patches)[0]} patches, each of size {np.shape(patches)[1:]}...', end='')
    np.random.shuffle(np.asarray(patches))
    mydict = {'patches': patches}
    fname = f'patchset_{filenum}.pkl'
    filetowrite = os.path.join(write_folder, fname)
    with open(filetowrite, 'wb') as handle:
        pickle.dump(mydict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    filenames.append(fname)
    
    #write filelist
    flname = os.path.join(write_folder, 'filenames.txt')
    file = open(flname, 'w')
    for f in filenames:
        file.write(f)
        file.write('\n')
    file.close()
    
    print('Done!')
    return()