import torch
import numpy as np

def get_movies_recons(model, dloader, num_movies=10):
    was_training = model.training
    model.eval()
    movies_so_far = 0
    movies = []
    recons = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        for inputs in next(iter(dloader)):
            inputs_mod = model.format_data_torch(inputs)
            outputs_mod = model.decode(model.encode(inputs_mod))
            #print(np.shape(outputs_mod.data))
            outputs = outputs_mod.cpu().numpy()
            #outputs = model.format_data_numpy(outputs_mod)
            movies.append(inputs.cpu().numpy())
            recons.append(outputs)
    
    movies = np.asarray(movies).squeeze()
    recons = np.asarray(recons).squeeze()
    return(movies, recons)