import matplotlib.pyplot as plt
import numpy as np

def plot_temporal_rf(imw_matrix, number, vectorize=False):
    """plot temporal receptive field
    Params:
        inw_matrix (matrix) (neuons, timepoints, height, width) matrix of weight kernel
        number (int) which inw matrix to sample from"""
    
    ts = np.array(imw_matrix[number].data)
    frames, height, width = np.shape(ts)
    
    if(vectorize):
        ts = np.reshape(ts, (frames, height*width)).T
        fig = plt.figure(figsize=(10,6))
        plt.imshow(ts,cmap='Greys_r')
        plt.axis('off')
        plt.tight_layout()

    else:
        fig = plt.figure(figsize=(10,6))
        for i, frame in enumerate(ts):
            #plt.imshow(frame.data)
            #plt.show()
            plt.subplot(1, frames, i+1)
            #print(frame.shape)
            plt.imshow(frame, cmap='Greys_r')
            plt.axis('off')
        plt.tight_layout()
        
def plot_movies_recons(movie, recon, number, vectorize=True): 
    
    movie = np.array(movie[number].data)
    recon = np.array(recon[number].data)
    
    frames, height, width = np.shape(movie)
    
    movie = np.reshape(movie, (frames, height*width)).T
    recon = np.reshape(recon, (frames, height*width)).T
    
    fig = plt.figure(figsize=(10,6))
    
    plt.subplot(121)
    plt.imshow(movie, cmap='Greys_r')
    plt.axis('off')
    
    plt.subplot(122)
    plt.imshow(recon, cmap='Greys_r')
    plt.axis('off')
    
    plt.tight_layout()



    
    
    