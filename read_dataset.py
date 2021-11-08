#	...................................................................................
'''
author: Palash Nandi.
'''
#	...................................................................................
import pandas as pd
import os
import numpy as np

'''
read_dataset():
input: a path to the dataset
output: returns a MinMax normalized df 
'''
def read_dataset(name):
	path = '/home/palash/ML_GitHub/PCA/dataset/' + name

	df = pd.read_csv(path)
	total_columns = df.shape[1]
	default_col_names = np.arange(total_columns)
	df.columns = default_col_names
	
	print(f'Accessing dataset: {name}')	
	print(f'df.shape: {df.shape}')

	return df


