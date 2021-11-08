#	...................................................................................
'''
author: Palash Nandi.
'''
#	...................................................................................

import math
import random
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from numpy import linalg as eigen_decomp

import read_dataset as rd
#	.....................................
def check_ratios(x):
    sum_ = np.sum(x)
    _ = [ i/sum_ for i in x]
    _.sort(reverse=True)

    print(f'The normalized variances are: {_}')

# 	.....................................
def PCA_from_scratch_linalg(df):
	print(f'PCA from scratch using linalg of Numpy')

	mean_arr = df.mean(axis = 0).to_numpy()
	X = np.array(df) - mean_arr
	X_T = X
	X = X.T
	# print(f'X: {X.shape}')
	# print(f'X_T: {X_T.shape}')
	
	covarience_matrix = np.matmul(X, X_T) / X.shape[1]

	print(f'X: {X.shape}, X_T: {X_T.shape}, cov_mat: {covarience_matrix.shape}\n')
	# print(covarience_matrix)
	
	lambdas, e_vecs = eigen_decomp.eig(covarience_matrix)
	# print(lambs)
	# Numpy returns the eigen vectors column-wise
	print(f'\nEigen vectors are:\n{np.transpose(e_vecs)}\n')
	check_ratios(lambdas)

# 	.....................................
def PCA_from_scratch_SVD(df):
	print(f'\n\nPCA from scratch using SVD of Numpy')
	mean_arr = df.mean(axis = 0).to_numpy()
	X = np.array(df) - mean_arr
	X_T = X
	X = X.T
	# print(f'X: {X.shape}')
	# print(f'X_T: {X_T.shape}')
	covarience_matrix = np.matmul(X, X_T) / X.shape[1]
	print(f'X: {X.shape}, X_T: {X_T.shape}, cov_mat: {covarience_matrix.shape}\n')
	# print(covarience_matrix)

	u, s, vh = np.linalg.svd(X, full_matrices=True)
	# print(f'u: {u.shape}, s: {s.shape}, vh: {vh.shape}')
	e_vec = []
	for eigenvector in np.transpose(u):
	    e_vec.append(np.dot(eigenvector.T, np.dot(covarience_matrix, eigenvector)))
	
	# Numpy returns the eigen vectors row-wise
	print(f'\nEigen vectors are:\n{np.transpose(u)}\n')
	check_ratios(e_vec)

# 	.....................................
def PCA_from_scratch_sklearn(df):
	#  PCA needs df in (row_number, features) format
	print(f'\n\nPCA from sklearn')
	
	eigenvalue_list = []
	mean_arr = df.mean(axis = 0).to_numpy()
	X = np.array(df) - mean_arr
	X_T = X
	X = X.T
	pca = PCA(n_components=X.shape[0])
	pca.fit(X_T)

	# print(pca.explained_variance_ratio_)
	# print(pca.singular_values_)
	# print(pca.components_.shape)

	for eigenvalue, eigenvector in zip(pca.explained_variance_ratio_, pca.components_):    
	    eigenvalue_list.append(eigenvalue)

	print(f'\nEigen vectors are:\n{pca.components_}\n')
	print(f'Eigen values: {eigenvalue_list}')
		


#	.....................................
# dataset_path = 'iris_dataset.csv'
# dataset_path = 'cancer_data.csv'
# dataset_path = 'mall_customer_dataset.csv'
dataset_path = 'wine_quality_dataset.csv'

print('\n\n\n')

df = rd.read_dataset(dataset_path)
PCA_from_scratch_linalg(df)
PCA_from_scratch_SVD(df)
PCA_from_scratch_sklearn(df)

print('\n\n\n')