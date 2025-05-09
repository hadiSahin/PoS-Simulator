import config
from network_utils import *
from node import *
from functions import *
# import networkx as nx
# from math import sqrt
# from itertools import tee
import matplotlib.pyplot as plt
from arg_parser import parser
import quantecon as qe
import warnings

warnings.simplefilter('ignore')

def main():

    args = parser.parse_args()
    
    data_init, data_rep = init_w()
    # print(data_init)

    for i in range(args.max_nodes):
        allNodes.append(Node(i, data_init['w'][i], data_init['W'][i], parent = i, 
                             sybil = 0, winit = data_init['w'][i], Winit = data_init['W'][i], group = data_init['group'][i], 
                             gSize = data_init['gSize'][i], gW = data_init['gW'][i], rep = data_init['rep'][i]))
        

    # for node in allNodes:
    #     print(f"Round {1} Node Id:{node.nodeId}, stakes:{node.w}, all stakes:{node.W}, Parent:{node.parent}, Splits:{node.splits} , group: {node.group}, group weight : {node.gW}")
    
    # print(f"all nodes: {allNodes}")

    # print(f"print maximum number of rounds: {args.rounds}")

    print(f"total number of nodes: {len(allNodes)}")

    results_main = pd.DataFrame(columns=['round', 'nodeId', 'w', 'W', 'parent', 'splits', 'proposer'])
    results_gini = pd.DataFrame(columns=['round', 'gini'])
    results_del= pd.DataFrame(columns=['round', 'nodeId', 'stakes'])
    
    expRep = calculate_expected_reputation(data_rep, config.DECAY_RATE)

    # print(f"Expected reputation: {expRep}")
    
    print(f"Decay rate: {config.DECAY_RATE}")

    for i in range(args.rounds):

        # for node in allNodes:
        #     print(f"Round {i} Node Id:{node.nodeId}, stakes:{node.w}, all stakes:{node.W}, Parent:{node.parent}, Splits:{node.sybil} , group: {node.group}, group weight : {node.gW}")
        



        print(f"----------------------------Round-{i}-----------------------------------------------------")  

        # for node in allNodes:
        # # print(f"Round {i} Node Id:{node.nodeId}, stakes:{node.w}, all stakes:{node.W}, Parent:{node.parent}, Splits:{node.sybil} , group: {node.group}, group weight : {node.gW}")
        #     if node.nodeId == 3171:
        #         print(f"Round {i} Node Id:{node.nodeId}, stakes:{node.w}, all stakes:{node.W}, Parent:{node.parent}, Splits:{node.sybil} , group: {node.group}, group weight : {node.gW}")

    

        if args.node_count != "fixed":
            deleteNode(i)
      
        if args.splitting == "True":
            splitting_decision(expRep)

        proposer = selectProposer(data_init)

        print(f"This is the proposer node {proposer}")
    
        data_main = single_run(i, proposer, data_rep)


        if args.splitting == "True":
            data_splitting = data_main.groupby(["round","parent"]).agg({'round': 'first', 'nodeId': 'first','w': 'sum', 'W': 'first', 'proposer': 'first'})
            wealth = data_splitting['w'].to_numpy()
            # gini = qe.gini_coefficient(wealth) # Gini coefficient
            gini = calculate_gini(wealth)
        else: 
            wealth = data_main['w'].to_numpy()
            
            # gini = qe.gini_coefficient(wealth) # Gini coefficient
            gini = calculate_gini(wealth)


        print(f"this is the gini coefficient: {gini}")

        # results from gini
        new_gini = pd.DataFrame({'round': i, 'gini' : gini}, index=[0])
        results_gini = pd.concat([results_gini, new_gini], ignore_index=True)
        
        # main results
        results_main = pd.concat([results_main, data_main], ignore_index=True)


        
    if args.splitting == "noSybil":
        fileDestination1 = "data/results/main_"+str(args.incentives)+"_"+str(args.max_nodes)+"_"+str(args.rounds)+"_"+str(args.reward)+"_"+str(args.node_count)+"_"+str(args.distribution)+"_"+str(args.splitting)
        fileDestination3 = "data/results/gini_"+str(args.incentives)+"_"+str(args.max_nodes)+"_"+str(args.rounds)+"_"+str(args.reward)+"_"+str(args.node_count)+"_"+str(args.distribution)+"_"+str(args.splitting)
    else: 
        fileDestination1 = "data/results/main_"+str(args.incentives)+"_"+str(args.max_nodes)+"_"+str(args.rounds)+"_"+str(args.reward)+"_"+str(args.node_count)+"_"+str(args.distribution)+"_"+str(args.splitting)
        fileDestination3 = "data/results/gini_"+str(args.incentives)+"_"+str(args.max_nodes)+"_"+str(args.rounds)+"_"+str(args.reward)+"_"+str(args.node_count)+"_"+str(args.distribution)+"_"+str(args.splitting)
                
    results_main.to_csv(fileDestination1, index=False)
    results_gini.to_csv(fileDestination3, index=False)
    


    
    # print(results.head())
    # data = results.groupby(["round","parent"]).agg({'round': 'first', 'nodeId': 'first','w': 'sum', 'W': 'first', 'proposer': 'first'})
    # data.to_csv('data/flat_%25_grouped.csv', index=False)


if __name__ == '__main__':
    main()
