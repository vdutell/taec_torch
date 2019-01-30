import torch
import utils.movie_readin as mru
from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn
import numpy as np
    
class AEC(nn.Module):
    def __init__(self, hidden_nodes, conv_width, pixel_patchsize, lambda_activation):
        super(AEC, self).__init__()

        # model paramters
        self.hidden_nodes = hidden_nodes
        self.conv_width = conv_width
        self.pixel_patchsize = pixel_patchsize
        self.temporal_conv_kernel_size = (conv_width, pixel_patchsize, pixel_patchsize)
        self.lambda_activation = lambda_activation
        
        # model structure
        self.tconv = nn.utils.weight_norm(nn.Conv3d(1,
                                                   hidden_nodes, 
                    kernel_size=self.temporal_conv_kernel_size,
                                                   stride=1),
                                        name='weight')
        
        #self.tconv = nn.Conv3d(1,
        #                       hidden_nodes,
        #                       kernel_size=self.temporal_conv_kernel_size,
        #                       stride=1)
            
        self.tdeconv = nn.ConvTranspose3d(hidden_nodes,
                                          1,
                                          kernel_size = np.transpose(self.temporal_conv_kernel_size),
                                          stride=1)
        
    #def tdeconv_tied(self, acts):
    #    out = F.conv_transpose3d(acts,
    #                            self.tconv.weight) #tied weights
    #    return(out)
    
    
    def format_data_numpy(self, x):
        x = np.detach().cpu().numpy()
        return(x)
    
    def encode(self, x):
        noise = 0
        return F.relu(self.tconv(x + noise))

    def decode(self, z):
        #recon = self.tdeconv_tied(z) # tied weights
        return self.tdeconv(z)

    def forward(self, x):
        activations = self.encode(x)
        #z = self.reparametrize(mu, logvar)
        decoded = self.decode(activations)
        return activations, decoded
    
    #scheduler = MultiStepLR(optimizer, milestones=[30,80], gamma=0.1)

    def loss_func(self, x, xhat, activations):
            #recon_loss = ((x[self.conv_width:-self.conv_width]-xhat[self.conv_width:-self.conv_width])**2).mean()
            recon_loss = nn.MSELoss()(xhat, x)
            #a = nn.CrossEntropyLoss()(output_y, y_labels)
            #recon_loss = ((x-xhat)**2).mean()
            activation_loss = torch.abs(activations).mean() * self.lambda_activation
            loss = recon_loss + activation_loss
            #total_loss = sum(losses)
            return(loss)
        
    def calc_snr(self, x, xhat):
        '''
        Calculate the ssignal to noise ratio for a reconstructed signal
        Params:
            x(Array): The signal before transformation
            x_hat (Array): The signal after transformation. Must be of same type of s.
        Returns:
            snr (float): The signal to noise ratio of the transformed signal s_prime
        '''

        #vectorize
        x = x.flatten()
        xhat = x.flatten()
        #calc signal and noise
        signal = x.mean()
        noise = (xhat - x).std()
        #calc their ratios
        snr = 10*(signal/noise).log10()

        return(snr)