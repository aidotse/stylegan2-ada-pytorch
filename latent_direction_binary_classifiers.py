''' Training a classifier to find the boundary hyper plane that separates images with certain features '''

''' need csv file with labels on some feature we want to find a decision boudary for, 
    and txt files of w-vectors'''

import click
import numpy as np
from sklearn.feature_extraction import img_to_graph
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
import pandas as pd
import os
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score
import pickle
from joblib import dump, load

import torch

'''The process is as follows:
    * Input latent space representation of images, use the style vector
    * Using a pretrained classifier, pass in a csv file containing labels that separates a certain feature
    * Train a binary classifier to find the hyper plane that separates the two labels
    * '''


# @click.command()
# @click.pass_context
# @click.option('--csv_path', help='csv file with psuedo-labels', required=True)
# @click.option('--w_matrix', type=np.ndarray, help='Style vectors of input images. Should be of shape [num_samples, num_features]')
# @click.option('--z_matrix', type=np.ndarray, help='Random vectors of input images. Should be of shape [num_samples, num_features]')

file_path = '/data/psuedoannotations/psudo_10k_imgs_tmp.csv'
# assuming the csv and w/z_mat are in the same order
# z_matrix_path = '/data/generated_imgs_smaller/z_mat.npy'
# w_matrix_path = '/data/generated_imgs_smaller/w_mat.npy'

# having one txt for w-vector file per image is better
w_vectors_path = '/data/generated_imgs_smaller/w_matrix_txt_files/'

csv_file = pd.read_csv(file_path, header=None)
# print(csv_file[0])
# z_matrix = np.load(z_matrix_path)
# w_matrix = np.empty((csv_file.shape[0], 512))
w_matrix = np.zeros((csv_file.shape[0], 512))
labels = np.empty((csv_file.shape[0], 1), dtype=int)

img_file_name = os.listdir(w_vectors_path)[0].split('.')[0]

# load vector from txt file
matrix = np.loadtxt(w_vectors_path+os.listdir(w_vectors_path)[0])

classifier = SVC(kernel='linear')

# given the image name, do:
# append w vector to w_matrix
# append the corresponding label from the 4th column (black frame)

for idx in range(csv_file.shape[0]):
# for idx in range(10):
    # image name without . 
    image_name = csv_file[0][idx].split('.')[0].split('_')[0]
    # load vector (stupid to do it this way)
    w_vector = np.transpose( np.loadtxt(w_vectors_path + image_name + '.class.1.txt')[:, np.newaxis] ) 
    # w_vector = torch.tensor(w_vector)
    label =int(csv_file[csv_file[0] == (image_name + "_1.jpg")][4])
    # 
    w_matrix[idx,:] = w_vector
    labels[idx] = label


# print(w_matrix.shape)
# print(labels.ravel().shape)


kf = StratifiedKFold(n_splits = 10)
# SVM classifier
print(labels[:, 0])
labels = labels.ravel()
svm = SVC(kernel='linear')

svm_acc_score = np.zeros((kf.n_splits))

for i, (train_index, test_index) in zip(range(10), kf.split(w_matrix,labels.ravel())):
    svm.fit(w_matrix[train_index], labels[train_index])
    y_pred = svm.predict(w_matrix[test_index])
    svm_acc_score[i] = accuracy_score(labels[test_index], y_pred)


svm_mean_acc_score = np.mean(svm_acc_score)
print("SVM accuracy score", svm_mean_acc_score)

print("Saving model...")
# filename = 'svm_10k_imgs.sav'
outfile = '/data/stylegan2-ada-pytorch/svm_10k_imgs.joblib'
dump(svm, outfile)
# with open(outfile, 'wb') as pickle_file:
#     dump(svm, pickle_file)
# PATH_TO_DIR = '/data/stylegan2-ada-pytorch/'
# open(filename, 'wb')
# pickle.dump(svm, filename)
