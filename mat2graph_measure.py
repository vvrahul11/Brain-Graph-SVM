import networkx as nx
import numpy as np
import scipy.io as sio
from scipy import stats
import os
import community
from glob import glob
from copy import copy
import pdb

# Converts np array to thresholded graph
def mat2graph_threshold(matrix, x):
	np.fill_diagonal(matrix,0)
	zscore = stats.zscore(matrix)
	mathresh = copy(zscore)
	thresh = np.percentile(zscore, x)
	mathresh[mathresh<thresh] = 0
	G = nx.from_numpy_matrix(mathresh, parallel_edges=False, create_using = nx.MultiGraph())
	return(G)

def calc_graph(matrix):
	thresholds = [90,85,80,75]
	glob = np.zeros((1,4))
	loc = np.zeros((1,4))
	Q = np.zeros((1,4))
	Ch = np.zeros((1,4))
	Ph = np.zeros((1,4))

	data = np.zeros((1,5))

	# Run graph measure analysis
	for index, threshold in enumerate(thresholds):
		graph = mat2graph_threshold(matrix, threshold)

		# Calculating global and average local efficiency
		glob[0,index] = nx.global_efficiency(graph)
		loc[0,index] = nx.local_efficiency(graph)

		# Community detection and modularity (1.25 )
		part = community.best_partition(graph, weight='1.25')
		Q[0,index] = community.modularity(part, graph)

		# Calculating connector and provincial hubs
		Z = module_degree_zscore(matrix, part)
		P = participation_coefficient(matrix, part)
		# connector hubs
		ch = np.zeros(matrix.shape[0])
		for i in range(len(ch)):
			if P[i] > 0.8 and Z[i] < 1.5:
				ch[i] = 1.0

			Ch[0,index] = np.sum(ch)

		# provincial hubs
		ph = np.zeros(matrix.shape[0])
		for i in range(len(ph)):
			if P[i] <= 0.3 and Z[i] >= 1.5:
				ph[i] = 1
			Ph[0,index] = np.sum(ph)

	# Averaging over each graph threshold
	meanglob = np.mean(glob)
	meanloc = np.mean(loc)
	meanQ = np.mean(Q)
	meanCh = np.mean(Ch)
	meanPh = np.mean(Ph)
	data[0,0] = meanglob
	data[0,1] = meanloc
	data[0,2] = meanQ
	data[0,3] = meanCh
	data[0,4] = meanPh
	return(data)

def module_degree_zscore(matrix,partition):
	# Reference: Guimera R, Amaral L. Nature (2005) 433:895-900.
	# This version is for undirected graphs
	# Adapted from Mika Rubinov (UNSW/U Cambridge) & Alex Fornito (U Melbourne)

	N = matrix.shape[0]
	Z = np.zeros(N)
	par = np.array(list(partition.values())).astype(float)
	max = np.arange(par.max())
	for i in range(max.shape[-1]):
		X = matrix[par == i]
		Koi = sum(X[:,par == i],2)
		Z[par==i] = (Koi - np.mean(Koi)) / np.std(Koi)
	return(Z)

def participation_coefficient(matrix,partition):
	# Reference: Guimera R, Amaral L. Nature (2005) 433:895-900.
	# This version is for undirected graphs
	# Adapted from Mika Rubinov (UNSW/U Cambridge) & Alex Fornito (U Melbourne)

	N = matrix.shape[0] # number of nodes
	par = np.array(list(partition.values())).astype(float) + 1
	Ko = sum(matrix,2) # sum rows
	Gc = np.matmul(((matrix != 0)*1), np.diag(par)) # neighbor community affiliation
	Kc2 = np.zeros(N) # commmunity-specific neighbors
	max = np.arange(par.max())
	for i in range(max.shape[-1]):
		Kc2 = Kc2 + np.power(sum(np.multiply(matrix, Gc==i)),2)

	P = np.ones(N) - Kc2 / np.power(Ko,2)

	return(P)
