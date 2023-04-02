import config
from network_utils import *
from node import *
from functions import *
import random
import matplotlib.pyplot as plt

def create_groups(num_groups):
    groups = []
    nodes = list(range(config.MAX_NODES))

    for i in range(num_groups):
        group_size = config.MAX_NODES // num_groups
        group = random.sample(nodes, group_size)
        groups.append(group)
        nodes = list(set(nodes) - set(group))
    return groups

def find_node_group(node_id, groups):
    for i, group in enumerate(groups):
        if node_id in group:
            return i
    return -1


def compute_reward(proposer, groups):
    group_index = find_node_group(proposer, groups)
    # print(f"groups {groups}")
    # print(f"group index {group_index}")
    B_i = config.REWARD
    # rate = config.RATE
    for node in allNodes: 
        node.W += B_i
        # node.w += transfers
        if node.nodeId in groups[group_index]:
            # print(f"group selected {groups[group_index]}")
            transfers = (B_i*node.w)/(len(groups[group_index]))
            node.w += transfers
    return transfers


def single_run(round,groups):
    results = pd.DataFrame(columns=['round', 'nodeId', 'w', 'W', 'parent', 'splits', 'proposer'])
    proposer = selectProposer()[0]
    compute_reward(proposer, groups)
    for node in allNodes:
        new_row = pd.DataFrame({'round': round, 'nodeId':node.nodeId , 'w':node.w, 'W':node.W, 'parent':node.parent, 'splits':node.splits, 'proposer': proposer}, index=[0])
        results = pd.concat([results, new_row], ignore_index=True)
        print(f"Round {round} Node Id:{node.nodeId}, stakes:{node.w}, all stakes:{node.W}, Parent:{node.parent}, Splits:{node.splits}  , Proposer:{proposer}")

    return results

def main():
    
    w,W = init_w()
    for i in range(config.MAX_NODES):
        allNodes.append(Node(i, w[i], W, parent=i, splits=0))
    num_groups=10
    # for node in allNodes:
    #     print(f"Round {1} Node Id:{node.nodeId}, stakes:{node.w}, all stakes:{node.W}, Parent:{node.parent}, Splits:{node.splits}")
    groups = create_groups(num_groups) #
    results = pd.DataFrame(columns=['round', 'nodeId', 'w', 'W', 'parent', 'splits', 'proposer'])
    for i in range(config.ROUNDS):
        print(f"----------------------------Round-{i}-----------------------------------------------------")
        data = single_run(i, groups)
        results = pd.concat([results, data], ignore_index=True)
    results.to_csv('data/chap4/pooledPoS_n1k_r10k.csv', index=False)
    # print(results.head())
    data = results.groupby(["round","parent"]).agg({'round': 'first', 'nodeId': 'first','w': 'sum', 'W': 'first', 'proposer': 'first'})
    data.to_csv('data/chap4/PooledPoS.csv', index=False)


if __name__ == '__main__':
    main()