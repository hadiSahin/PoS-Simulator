import config
from network_utils import *
from node import *
from functions import *
# import networkx as nx
# from math import sqrt
# from itertools import tee
import matplotlib.pyplot as plt


def compute_reward(proposer):
    B_i = config.REWARD
    rate = config.RATE
    transfers = B_i*(1-rate)/(len(allNodes))
    for node in allNodes: 
        node.W += B_i
        # node.w += transfers
        if node.nodeId == proposer:
            # reward_basic = B_i*rate
            node.w += B_i
    return transfers


def single_run(round):
    results = pd.DataFrame(columns=['round', 'nodeId', 'w', 'W', 'parent', 'splits', 'proposer'])
    proposer = selectProposer()[0]
    # Splitting()
    compute_reward(proposer)
    
    
    for node in allNodes:
        new_row = pd.DataFrame({'round': round, 'nodeId':node.nodeId , 'w':node.w, 'W':node.W, 'parent':node.parent, 'splits':node.splits, 'proposer': proposer}, index=[0])
        results = pd.concat([results, new_row], ignore_index=True)
        print(f"Round {round} Node Id:{node.nodeId}, stakes:{node.w}, all stakes:{node.W}, Parent:{node.parent}, Splits:{node.splits}  , Proposer:{proposer}")

    return results

def main():
    
    w,W = init_w()
    for i in range(config.MAX_NODES):
        allNodes.append(Node(i, w[i], W, parent=i, splits=0))

    # for node in allNodes:
    #     print(f"Round {1} Node Id:{node.nodeId}, stakes:{node.w}, all stakes:{node.W}, Parent:{node.parent}, Splits:{node.splits}")
    
    results = pd.DataFrame(columns=['round', 'nodeId', 'w', 'W', 'parent', 'splits', 'proposer'])
    for i in range(config.ROUNDS):
        print(f"----------------------------Round-{i}-----------------------------------------------------")
        data = single_run(i)
        results = pd.concat([results, data], ignore_index=True)
    results.to_csv('data/chap4/purePoS_n1k_r10k.csv', index=False)
    # print(results.head())
    # data = results.groupby(["round","parent"]).agg({'round': 'first', 'nodeId': 'first','w': 'sum', 'W': 'first', 'proposer': 'first'})
    # data.to_csv('data/PurePoS.csv', index=False)


if __name__ == '__main__':
    main()