'''latent vector manipulation: 
    Take a vector that produces an image with frame
    using the pre trained binary classifier, move the point perpendicularly to the plane
    For now, experiment how the features in the image changes when going along the vector
    Is the quality of the image preserved?
    
    Tools: binary classifier
            latent vector representing an image with a frame
            '''

from joblib import load
import numpy as np
import PIL 
import PIL.Image
from sklearn.svm import SVC
import pickle
import torch

#trained model path
model_path = "/ISIC256/not_our_models/network-snapshot-020000_all.pkl"
outdir = "/data/stylegan2-ada-pytorch/latent_direction_imgs"

with open(model_path, 'rb') as f:
    G = pickle.load(f)['G_ema'].cuda()



# path to a vector that generates an image with frame: /data/generated_imgs_smaller/w_matrix_txt_files/seed0003.class.1.txt
svm = load("/data/stylegan2-ada-pytorch/svm_10k_imgs.joblib")
w_vector = np.transpose( np.loadtxt('/data/generated_imgs_smaller/w_matrix_txt_files/seed0003.class.1.txt')[:, np.newaxis] ) 

# the normal of he hyperplane is in the direction of 1 label
b = svm.intercept_
w = svm.coef_
w_matrix = np.zeros((1000, 1, 14, 512))

# distance from point w (the image) to the plane
d0 = np.linalg.norm(np.dot(w_vector, w.T) + b)/np.linalg.norm(w)
print("d0",d0)
# d = np.linalg.norm(np.dot(w_matrix[0,0,0,:], w.T) + b)/np.linalg.norm(w)
# print("d", d)
# by parametrising, we get: 

for i in  range(1000):
    done = False
    for j in range(14):
        w_matrix[i,0,j,:] = w_vector - i*0.001*w
        d = np.linalg.norm(np.dot(w_matrix[i,0,j,:], w.T) + b)/np.linalg.norm(w)
    #     # print("d",d)
        if d > 3*d0:
            # print(w_matrix[i,0,j,:])
            # print(d)
            stop_idx = i
            print(stop_idx)
            done = True
            break
    if done:
        break

# print(w_matrix.shape)
# print(w_matrix[i,:][np.newaxis].shape)
for i in range(stop_idx):
    img = G.synthesis(torch.tensor(w_matrix[i,:,:,:], device='cuda'), noise_mode='const', force_fp32=True)
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/para_{i}.jpg')