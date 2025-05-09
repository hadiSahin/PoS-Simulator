from xml.dom.minicompat import NodeList
import networkx as nx
import pandas as pd 
import os 
import numpy as np
import config
from arg_parser import parser

allNodes = []
data_path = ['data']
args = parser.parse_args()
groups = []

def init_w():


	if args.distribution == 'uniswap':
		rep_scale = np.loadtxt("data/initialDist/rep_scale_5000.csv", delimiter=",")
		data = pd.read_csv("data/initialDist/groups3172.csv", delimiter=",")
		rep = np.loadtxt("data/initialDist/rep_normal_3172.csv", delimiter=",")
	else:
		rep_scale = np.loadtxt("data/initialDist/rep_scale_5000.csv", delimiter=",")
		data = pd.read_csv("data/initialDist/groups2500.csv", delimiter=",")
		rep = np.loadtxt("data/initialDist/rep_normal_2500.csv", delimiter=",")

	
	# data = pd.read_csv("data/initialDist/groups3172.csv", delimiter=",")
	# arr = np.random.normal(config.MAX_ALGORAND / 2, 10, config.MAX_NODES)
	# if (arr <= 0).any():
	# 	arr[arr <= 0] = 1
	#reputation values
	
	rep_scale = np.sort(rep_scale)
	# rep = np.loadtxt("data/initialDist/rep_normal_3172.csv", delimiter=",")
	
	rep = np.sort(rep)
	if args.distribution == 'normal':
		print("This is normal distribution")
		arr = np.loadtxt("data/initialDist/normal2500.csv", delimiter=",")
		arr = np.sort(arr)
	elif args.distribution == 'pareto':
		print("This is pareto distribution")
		arr = np.loadtxt("data/initialDist/pareto2500.csv", delimiter=",")
		arr = np.sort(arr)
	elif args.distribution == 'uniform':
		print("This is uniform distribution")
		# arr = np.loadtxt("data/initialDist/pareto3172.csv", delimiter=",")
		arr = np.loadtxt("data/initialDist/uniform2500.csv", delimiter=",")
		arr = np.sort(arr)
	elif args.distribution == 'uniswap':
		print("This is uniswap's distribution")
		arr = np.loadtxt("data/initialDist/pareto3172.csv", delimiter=",", skiprows=1, usecols=1)
		# arr = np.loadtxt("data/initialDist/uniform2500.csv", delimiter=",")
		arr = np.sort(arr)
		
	# arr =np.sort(arr, axis=0)clear
	sum = np.sum(arr)
	# print(arr.shape)
	# print(data.shape)

	print("Maximum value:", arr[-1])



    
	data['w'] = arr
	data['W'] = sum
	data["gW"] = data.groupby("group")["w"].transform("sum")
	data["rep"] = rep
	# data["scale"] = rep_scale

	# row = data[data['w'] == 5000181.739]
	# print(row)

     
	# print(data.head())
	# print(data.tail())
	return data, rep_scale









