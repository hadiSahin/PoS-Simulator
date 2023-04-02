import config
from network_utils import *
from node import *
from itertools import tee
from math import sqrt
import itertools
import numpy as np
import pandas as pd
import time
import random

def selectProposer():
    stakes_list = []
    node_list = []
    for node in allNodes: 
        node_list.append(node.nodeId)
        stakes_list.append(node.w)
        # print(stakes_list)
    proposer = random.choices(node_list, weights=stakes_list, k=1)
    return proposer 

def Splitting():
    transfers = config.REWARD*(1-config.RATE)/(len(allNodes))
    n = int(len(allNodes))-1
    sybil = np.random.choice(allNodes, int(config.MAX_NODES*config.SYBIL/100))
    # print(f"list of sybils {len(sybil)}")
    for node in sybil:
        if node.w > 2*config.SPLIT_STAKES and transfers > node.splits*config.COST_SPLIT and audit(transfers, node.w, node.splits) is True:
            # print(f"node id {node.nodeId}: node w {node.w} is compared to {2*config.SPLIT_STAKES}")
            node.w += -1
            node.splits +=1 
            # print(f"Transfers {transfers}")
            # print(f"Nodes which created sybil {node.nodeId}")
            n +=1
            # print(f"Number of nodes {n}")
            # if node.nodeId != node.parent:
            allNodes.append(Node(n, 1, node.W, parent=node.parent, splits = 0))

def audit(transfers, w, splits):
    if config.PROBA == 0:
        return True 
    else:
        return True if ((1-config.PROBA)/config.PROBA)*transfers-w*config.SLASH*(splits+1) > 0 else False 


   



