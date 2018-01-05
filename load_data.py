import networkx as nx
import numpy as np
import scipy.io as sio
from mat2graph_measure import *
import os
from sklearn_pandas import DataFrameMapper, cross_val_score
import pdb

# This specific analysis is for participants undergoing four levels of propofol Sedation
# The number of conditions can easily be modfified to fit other experimental designs

def load_data(data_path):
	count = 0
	# loading in data
	dataset = np.zeros((336,8))
	for dirpath, dirnames, files in os.walk(data_path):
		for scalind, scale in enumerate(dirnames):
			for filind, filename in enumerate(os.listdir(os.path.join(data_path, scale))):
				# loading in each mat file
				mat = sio.loadmat(os.path.join(data_path, scale, filename))
				if 'awakemat' in mat:
					matrix = mat['awakemat']
					data = calc_graph(matrix)
					dataset[count,0] = count
					dataset[count,1] = scalind
					dataset[count,2] = 1
					dataset[count,3:] = data
					count = count + 1
					print(count)
					print(filename)
					print(scale)
					print(data)
				elif 'mildmat' in mat:
					matrix = mat['mildmat']
					data = calc_graph(matrix)
					dataset[count,0] = count
					dataset[count,1] = scalind
					dataset[count,2] = 2
					dataset[count,3:] = data
					count = count + 1
					print(count)
					print(filename)
					print(scale)
					print(data)
				elif 'modmat' in mat:
					matrix = mat['modmat']
					data = calc_graph(matrix)
					dataset[count,0] = count
					dataset[count,1] = scalind
					dataset[count,2] = 3
					dataset[count,3:] = data
					count = count + 1
					print(count)
					print(filename)
					print(scale)
					print(data)
				else:
					matrix = mat['recovmat']
					data = calc_graph(matrix)
					dataset[count,0] = count
					dataset[count,1] = scalind
					dataset[count,2] = 4
					dataset[count,3:] = data
					count = count + 1
					print(count)
					print(filename)
					print(scale)
					print(data)
					np.savetxt("dataset_corr_fisher.csv", dataset, delimiter=",", fmt='%f')
		return(dataset)
