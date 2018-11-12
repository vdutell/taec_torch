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
    
    
    vmin = np.min(movie)
    vmax = np.max(movie)
    fig = plt.figure(figsize=(10,6))
    
    plt.subplot(121)
    plt.imshow(movie, cmap='Greys_r', vmin=vmin, vmax=vmax)
    plt.axis('off')
    plt.colorbar()
    
    plt.subplot(122)
    plt.imshow(recon, cmap='Greys_r', vmin=vmin, vmax=vmax)
    plt.axis('off')
    plt.colorbar()
    
    plt.tight_layout()


def pad_data(data):
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]),
    (1, 1), (1, 1))                       # add some space between filters
    + ((0, 0),) * (data.ndim - 3))        # don't pad the last dimension (if there is one)
    padded_data = np.pad(data, padding, mode="constant", constant_values=1)
    # tile the filters into an image
    padded_data = padded_data.reshape((n, n) + padded_data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, padded_data.ndim + 1)))
    padded_data = padded_data.reshape((n * padded_data.shape[1], n * padded_data.shape[3]) + padded_data.shape[4:])
    return padded_data
    
    
def plot_tiled_rfs(data, rescale=False, normalize=False, colorbar=True):
    #data is of shape ???

    #Rescale data
    if(rescale):
        mean_data = np.mean(data)
        min_data = np.amin(data)
        max_data = np.amax(data)
        data = (((data-min_data)/(max_data-min_data))*2)-1

    if normalize:
        data = normalize_data(data)

    if len(data.shape) >= 3:
        data = pad_data(data)
    #print(data.shape)

    fig = plt.imshow(data, 
                     cmap="Greys_r",
                     interpolation="none")
    fig.set_clim(vmin=-1.0, vmax=1.0)
    plt.xticks([])
    plt.yticks([])
    plt.tick_params(
        axis="both",
        bottom="off",
        top="off",
        left="off",
        right="off") 
    plt.axis('off')

    if(colorbar):
        plt.colorbar()

    return (fig)

    
def plot_temporal_weights(wmatrix):
    # data is of shape (frames,height,width)
    nframes = wmatrix.shape[1]

    fig = plt.figure(figsize=(18,6), dpi= 200);


    for frame in range(nframes):
        plt.subplot(1,nframes,frame+1);
        plot_tiled_rfs(wmatrix[:,frame,:,:],rescale=True,colorbar=False);

    return(fig)

def plot_one_temporal_weight(wmatrix):
    '''
    Plot a the temporal weight matrix for one hidden layer neuron
    
    Parameters
        wmatrix (numpy array) of size [frames, x,y]
    Returns
        fig: plot figure
    '''
