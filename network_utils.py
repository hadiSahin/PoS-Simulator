from xml.dom.minicompat import NodeList
import networkx as nx
import pandas as pd 
import os 
import numpy as np
import config

allNodes = []
data_path = ['data']

def init_w():
	# arr = np.random.normal(config.MAX_ALGORAND / 2, 10, config.MAX_NODES)
	# if (arr <= 0).any():
	# 	arr[arr <= 0] = 1
	arr = np.loadtxt("C:/Users/Administrator/OneDrive - Florida International University/Documents/rBlockchain/PoSSimulator/data/initialDist/normal1k.csv", delimiter=",")
	sum = np.sum(arr)
	return arr, sum




