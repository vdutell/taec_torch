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
    
    def format_data_torch(self, x):
        x = torch.unsqueeze(torch.tensor(x),1)
        x = Variable(x.float()).cuda()
        return(x)
    
    def format_data_numpy(self, x):
        x = np.detach().cpu().numpy()
        return(x)
    
    def encode(self, x):
        noise = 0
        x = x + noise
        activations = F.relu(self.tconv(x))
        return activations

    def decode(self, z):
        recon = self.tdeconv(z) # non tied weights
        #recon = self.tdeconv_tied(z) # tied weights
        return recon

    def forward(self, x):
        activations = self.encode(x)
        #z = self.reparametrize(mu, logvar)
        decoded = self.decode(activations)
        return activations, decoded
    
    #scheduler = MultiStepLR(optimizer, milestones=[30,80], gamma=0.1)

    def loss_func(self, x, xhat, activations):
            #recon_loss = ((x[self.conv_width:-self.conv_width]-xhat[self.conv_width:-self.conv_width])**2).mean()
            recon_loss = ((x-xhat)**2).mean()
            activation_loss = torch.abs(activations).mean() * self.lambda_activation
            loss = recon_loss + activation_loss
            return(loss)