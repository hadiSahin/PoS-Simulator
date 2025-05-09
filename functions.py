import config
from network_utils import *
from node import *
from itertools import tee
import math 
from scipy.stats import binom
import itertools
import numpy as np
import pandas as pd
import time
import random
from arg_parser import parser


max_nodes = args.max_nodes
# print(f"this is the period: {period}")


def single_run(round, proposer, rep_data):

    results = pd.DataFrame(columns=['round', 'nodeId', 'w', 'W', 'parent', 'sybil', 'proposer'])

    
    # data_cost = deleteNode(reward,round)
    calculate_and_assign_reward(proposer, round)

    print(f"Number of remaining nodes is  {len(allNodes)} at round {round}")
    
    for node in allNodes:
        new_row = pd.DataFrame({'round': round, 'nodeId':node.nodeId , 'w':node.w, 'W':node.W, 'parent':node.parent, 'sybil':node.sybil, 'proposer': proposer}, index=[0])
        results = pd.concat([results, new_row], ignore_index=True)

        # update reputation
        reputation_update(rep_data, node)

        # print(f"Round {round} Node Id:{node.nodeId}, stakes:{node.w}, totalStakes:{node.W}, Parent:{node.parent}, Splits:{node.splits}  , Proposer:{proposer}, initialStake = {node.winit}, initialTotalStakes = {node.Winit}")
    # print(results)
    return results



def selection_prob_first_round(w_ratio, percentage):
  
    # Calculate the number of draws based on percentage and total population
    n = round(percentage * len(allNodes))

    # print(f"The number of first selected group is: {n}")
    
    # Probability of not being selected in a single draw
    prob_not_selected_single = 1 - w_ratio
    
    # Probability of not being selected in n draws
    prob_not_selected_n = prob_not_selected_single ** n
    
    # Probability of being selected at least once
    prob_selected = 1 - prob_not_selected_n
    
    return prob_selected



def reputation_update(data, node):

    data = data.reshape(1, -1)

    # print(data.shape)

    scale = data[0]

    # print(f"previous rep: {node.rep}")

    # Check if there is any value in scale greater than node.rep
    if np.any(scale > node.rep):
        # Find the first value in scale greater than node.rep
        index = np.argmax(scale > node.rep)
        node.rep = scale[index]  # Update reputation
    else:
        # Keep the existing reputation if no larger value exists
        node.rep = node.rep

    # print(f"updated rep: {node.rep}")
    


def required_selection(winit, Winit, w, W, i, node):

    reward  = config.REWARD

    if args.incentives == 'pool':
        reward = config.REWARD*node.w/node.gW
        # print(f"Reward in pool incentive mechanism: {reward}")

    if args.incentives == 'algorand':
        reward = config.REWARD*node.w/node.W
        # print(f"Reward in algorand incentive mechanism: {reward}")

    if args.incentives == 'geometric':
        period = int(args.rounds/args.period)
        new_round = 0
        if i <= period:
            new_round = i 
            reward = config.REWARD*period
        elif  i <= 2*period and i > period :
            if i % period == 0:
                new_round = period
            else :
                new_round = i % period
            reward = config.REWARD*period/2
        elif i > 2*period:
            if (i-period) % (args.rounds-2*period) == 0:
                new_round = (args.rounds-2*period)
            else :
                new_round = (i-period) % (args.rounds-2*period)
            period = args.rounds-2*period
            reward = config.REWARD*(args.rounds-2*period)/4
        reward = (1+reward)**(new_round/period)-(1+reward)**((new_round-1)/period)
        # print(f"Reward in algorand incentive mechanism: {reward}")

    
    number = ((winit / Winit) * (W + (config.ROUNDS - i) * reward) - w) / reward

    rounded_nearest= round(number)

    return rounded_nearest

def probability_at_least_k(n, k, p):
    return 1 - binom.cdf(k - 1, n, p)  # Complement of P(X <= k-1)




def calculate_expected_reputation(data, decay_rate):
   
    # Period indices (1 to n)
    periods = np.arange(1, len(data) + 1)  # Start from 1

    # Apply decay rate progressively
    decayed_reputation = np.array(data) * (1 - decay_rate) ** periods

    # Calculate the average decayed reputation
    average_decayed_reputation = np.mean(decayed_reputation)

    # print(f"Average Decayed Reputation: {average_decayed_reputation}")

    return average_decayed_reputation






def calculate_and_assign_reward(proposer,i):
    args = parser.parse_args()
    if args.incentives == "pure":
        for node in allNodes: 
            node.W += config.REWARD
            if node.nodeId == proposer:
                # print(f"Previous stake of the proposer: {node.w}")
                node.w += config.REWARD
                # print(f"{node.nodeId} receieves a participation reward of {config.REWARD} ")
                # print(f"Current stake of the proposer: {node.w}")
    elif args.incentives == "pool":
        for node in allNodes: 
            node.W += config.REWARD
            # print(f"Total weight is updated")    
            if node.group == proposer:
                # print(f"Previous stake of the proposer: {node.w}")
                node.w += config.REWARD*(node.w/node.gW) 
                # print(f"{node.nodeId} receieves a participation reward of {config.REWARD} ")
                # print(f"Current stake of the proposer: {node.w}")  
            node.gW += config.REWARD
            # print(f"Group weight is updated")    
    elif args.incentives == "algorand":
        for node in allNodes: 
            # print(f"Previous stake of the proposer: {node.w}")
            node.w += config.REWARD*(node.w/node.W)
            # print(f"Current stake of the proposer: {node.w}")  
            node.W += config.REWARD
    elif args.incentives == "geometric":
        period = int(args.rounds/args.period)
        round = i+1
        new_round = 0
        if round <= period:
            new_round = round
            reward = config.REWARD*period
        elif  round <= 2*period and round > period :
            if round % period == 0:
                new_round = period
            else :
                new_round = round % period
            reward = config.REWARD*period/2
        elif round > 2*period:
            if (round-period) % (args.rounds-2*period) == 0:
                new_round = (args.rounds-2*period)
            else :
                new_round = (round-period) % (args.rounds-2*period)
            period = args.rounds-2*period
            reward = config.REWARD*(args.rounds-2*period)/4
        rewardpR = (1+reward)**(new_round/period)-(1+reward)**((new_round-1)/period)
        # print(f"this is the period : {period} ")
        # print(f"this is the old round : {round} ")
        # print(f"this is the new round : {new_round} ")
        # print(f"this is total reward : {reward} ")
        # print(f"this is reward per round: {rewardpR} ")
        for node in allNodes: 
            node.W += rewardpR
            # print(f"total stake is increased by {rewardpR}")
            if node.nodeId == proposer:
                # print(f"Previous stake of the proposer: {node.w}")
                node.w += rewardpR
                # print(f"{node.nodeId} receieves a participation reward of {rewardpR} ")
                # print(f"Current stake of the proposer: {node.w}")  
            
    elif args.incentives == "universal":
        for node in allNodes: 
            node.W += config.REWARD
            rate = args.transfer_rate
            transfers = config.REWARD*(1-rate)/(len(allNodes))
            node.w += transfers
            if node.nodeId == proposer:
                node.w += config.REWARD*rate
    elif args.incentives == 'boosted':
        for node in allNodes: 
            node.W += config.REWARD
            if node.nodeId == proposer:
                # print(f"Previous stake of the proposer: {node.w}")
                node.w += config.REWARD
                # print(f"Current stake of the proposer: {node.w}")  


def selectProposer(data_init):
    stakes_list = []
    rep_list = []
    node_list = []

    if args.incentives == 'pool':
        data = data_init.groupby("group", as_index=False).agg({'gW': 'first'})
        # In this case the proposer is the group
        proposer = random.choices(data['group'], weights=data['gW'], k=1)[0] 
        

    elif args.incentives == 'boosted':

        df = pd.DataFrame(columns=['nodeId', 'w'])

        for node in allNodes: 
            node_list.append(node.nodeId)
            stakes_list.append(node.w)
            rep_list.append(node.rep)
            # print(stakes_list)

        sample_size = math.floor(len(rep_list) * config.FIRST_SAMPLE_RATE)
        first_group = random.choices(node_list, weights=rep_list, k=sample_size)
        
        # print(f"sample size is {sample_size}")
        # print(first_group)
        # print(f"the length of stake list: {len(stakes_list)}")
        # print(f"the number of nodes: {len(node_list)}")
        # proposer = random.choices(node_list, weights=wealth_new, k=1) 
        df = pd.DataFrame({"nodeId": node_list, "w": stakes_list})
        mask = df['nodeId'].isin(first_group)
        df = df[mask]
        nodes = df['nodeId'].to_numpy()
        wealth = df['w'].to_numpy()
        thresh = np.percentile (wealth, args.threshold)
        # print(f"the threshold is : {thresh}")
        tax = np.maximum(wealth - thresh, 0) * args.taxRate # 0.5 is the default tax rate 
        # print(f"the tax is : {tax}")
        wealth_new = wealth - tax + tax.sum() / len(nodes) # new wealth after redistribution
        # print(f"this is the new wealth: {len(wealth_new)}")
        # print(wealth_new)
        proposer = random.choices(nodes, weights=wealth_new, k=1)[0]
        # proposer = [1]
        print(f"{proposer} is selected to create the new block")

    else:
        for node in allNodes: 
            node_list.append(node.nodeId)
            stakes_list.append(node.w)
            # print(stakes_list)
        proposer = random.choices(node_list, weights=stakes_list, k=1)[0] # this is basically pure PoS selection
        print(f"This is the proposer :{proposer}")
    return proposer



def splitting_decision(expected_rep):

    n = int(len(allNodes))-1

    # We assume that nodes that have equal and higher than two coins and no prior sybil node can create a sybil node.

    # pool_sybil_creators = [node for node in allNodes if node.sybil == 0 and node.w >= 2*config.SYBIL_STAKE]
    
    pool_sybil_creators = [node for node in allNodes if node.w >= 2]

    if any(node.w < 2 for node in pool_sybil_creators):
        raise ValueError("Node with w < 2 found in pool_sybil_creators.")

    sybil = np.random.choice(pool_sybil_creators, int(config.MAX_NODES*config.SYBIL), replace=False)

    if any(node.w < 2 for node in sybil):
        raise ValueError("Node with w < 2 found in pool_sybil_creators.")

    


    if args.incentives == 'boosted':


        # Create the DataFrame
        data = []
        for node in allNodes:
            data.append({"nodeId": node.nodeId, "w": node.w, "rep": node.rep})
        
        df = pd.DataFrame(data)

        # Calculate thresholds and taxes
        thresh1 = np.percentile(df['w'], args.threshold)
        df['tax1'] = np.maximum(df['w'] - thresh1, 0) * config.TAX_RATE

        # Updated stake distribution
        df['updated_w'] = df['w'] - df['tax1'] + df['tax1'].sum() / (n + 1)


        for node in sybil:
            # Find the node in the DataFrame
            node_index = df.index[df['nodeId'] == node.nodeId].item()

            # Subtract 1 from the node's stake
            df.at[node_index, 'w'] -= 1

            # Add a new stake of 1 for sybil nodes
            new_row = pd.DataFrame({
                "nodeId": [f"sybil_{len(df) + 1}"],
                "w": [1],
                "rep": [0],
            })
            df = pd.concat([df, new_row], ignore_index=True)

            # Recalculate thresholds and taxes for the new list
            thresh2 = np.percentile(df['w'], args.threshold)
            df['tax2'] = np.maximum(df['w'] - thresh2, 0) * config.TAX_RATE
            df['updated_w2'] = df['w'] - df['tax2'] + df['tax2'].sum() / (n + 1)

            # Parent without sybil
            wParent1 = df.at[node_index, 'updated_w']
            ratio_rep_parent = df.at[node_index, 'rep']/ df['rep'].sum()
            ratio_stake_parent = wParent1 / node.W
            # print(f"Node's stake without splitting: {wParent1}")
            

            # Parent with sybil
            wParent2 = df.at[node_index, 'updated_w2']
            ratio_stake_parent_splitting = wParent2 / node.W
            ratio_rep_parent_w_sybil = df.at[node_index, 'rep'] / (df['rep'].sum() + expected_rep)
            # print(f"Node's stake with splitting: {wParent2}")
            # print(f"reputation weight {df['updated_w2'].iloc[-1]}")
            ratio_stake_sybil = df['updated_w2'].iloc[-1]

            # Sybil ratios
            ratio_rep_sybil = expected_rep / (df['rep'].sum() + expected_rep)
            ratio_stake_sybil = ratio_stake_sybil / node.W


            # Probability calculations
           
            p1 = selection_prob_first_round(ratio_rep_parent, config.FIRST_SAMPLE_RATE)* ratio_stake_parent
            p2 = selection_prob_first_round(ratio_rep_parent_w_sybil, config.FIRST_SAMPLE_RATE)* ratio_stake_parent_splitting
            q1 = selection_prob_first_round(ratio_rep_sybil, config.FIRST_SAMPLE_RATE)*ratio_stake_sybil


            # print(f"The p1,p2 1nd q1 probabilities are: {p1},{p2}, and {q1} ")
            if p1 < 0 or p2 < 0 or q1 < 0:
                raise ValueError(f"Negative probability detected: p1={p1}, p2={p2}, q1={q1}")
            

            # Benefits calculations
            benefits_no_splitting = (
                p1 * ((node.w + config.REWARD) / (node.W + config.REWARD)) +
                (1 - p1) * (node.w / (node.W + config.REWARD))
            )

            benefits_splitting = (
                p2 * ((node.w-1 + config.REWARD) / (node.W + config.REWARD)) +
                (1 - p2) * ((node.w-1) / (node.W + config.REWARD)) +
                q1 * ((1 + config.REWARD) / (node.W + config.REWARD)) +
                (1 - q1) * (1 / (node.W + config.REWARD))
            )

            if benefits_splitting > benefits_no_splitting:
                node.w -= 1
                node.sybil += 1
                node.gSize += 1
                n += 1
                allNodes.append(
                    Node(
                        n, 1, node.W, parent=node.parent, sybil=0, winit=node.winit,
                        Winit=node.Winit, group=node.group, gSize=node.gSize, gW=node.gW, rep=0
                    )
                )

                print(f"Benefit: {benefits_splitting-benefits_no_splitting}")
        print(f"Total number of nodes: {n}")





def deleteNode(round):

    # args = parser.parse_args()
    # rate =args.transfer_rate
    # data = pd.DataFrame(columns=['round', 'nodeId','stakes','cost'])


    # stakes_list = []
    # node_list = []

    data_boosted = pd.DataFrame(columns=["nodeId", "w", "rep"])
    

   

    for node in allNodes:

       
        # pure 
        if args.incentives == 'pure':

            p1 = node.w/node.W

            k = required_selection(node.winit, node.Winit, node.w, node.W, round, node)

            # print(f"Required number of selection: {k}")

            prob_k = probability_at_least_k(config.ROUNDS, k, p1)

            # print(f"Probability of at least k: {prob_k}")

            # p2 = 1-(1-p1)**(config.ROUNDS-round+1)


            # print(f"PURE: Current probability: {p1}")
            # print(f"PURE: Long term probability: {p2}")
            # # Benefits calculations
            # benefits_continuing = (
            #     p2 * ((node.w + config.REWARD) / (node.W + config.REWARD)) +
            #     (1 - p2) * (node.w / (node.W + config.REWARD))
            # )

    
            # starting_stake_ratio = node.winit/node.Winit

        # pool 
        elif args.incentives == 'pool':

            p1 = node.gW/node.W

            # print(f"Probability of group selection: {p1}")

            k = required_selection(node.winit, node.Winit, node.w, node.W, round, node)

            # print(f"Required number of selection: {k}")

            prob_k = probability_at_least_k(config.ROUNDS, k, p1)

            # print(f"Probability of at least k: {prob_k}")
  


        #algorand 
        elif args.incentives == 'algorand':

            p1 = 1

            k = required_selection(node.winit, node.Winit, node.w, node.W, round, node)

            # print(f"Required number of selection: {k}")

            prob_k = probability_at_least_k(config.ROUNDS, k, p1)

            # print(f"Probability of at least k: {prob_k}")

        # geometric
        elif args.incentives == 'geometric':

            p1 = node.w/node.W

            k = required_selection(node.winit, node.Winit, node.w, node.W, round, node)

            # print(f"Required number of selection: {k}")

            prob_k = probability_at_least_k(config.ROUNDS, k, p1)

            # print(f"Probability of at least k: {prob_k}")


        # boosted
        elif args.incentives == 'boosted':

            new_row = pd.DataFrame([{"nodeId": node.nodeId, "w": node.w, "rep": node.rep}])
            df = pd.concat([data_boosted, new_row], ignore_index=True)

            # Calculate thresholds and taxes
            thresh1 = np.percentile(df['w'], args.threshold)
            df['tax1'] = np.maximum(df['w'] - thresh1, 0) * config.TAX_RATE

            # Updated stake distribution
            df['updated_w'] = df['w'] - df['tax1'] + df['tax1'].sum() / len(allNodes)

            # Find the node in the DataFrame
            node_index = df.index[df['nodeId'] == node.nodeId].item()

      
            wParent1 = df.at[node_index, 'updated_w']

            # print(f"Updated weight: {wParent1}")
            # print(f"Regular weight: {node.w}")

            # Parent without sybil
            wParent1 = df.at[node_index, 'updated_w']
            ratio_rep_parent = df.at[node_index, 'rep']/ df['rep'].sum()
            ratio_stake_parent = wParent1 / node.W
            # print(f"Node's stake without splitting: {wParent1}")
        
            # p1 = selection_prob_first_round(ratio_rep_parent, config.FIRST_SAMPLE_RATE)* ratio_stake_parent

            p1 = math.ceil(config.FIRST_SAMPLE_RATE*config.MAX_NODES)*ratio_rep_parent* ratio_stake_parent


            k = required_selection(node.winit, node.Winit, node.w, node.W, round, node)

            # print(f"Required number of selection: {k}")

            prob_k = probability_at_least_k(config.ROUNDS, k, p1)

            # print(f"Probability of at least k: {prob_k}")

            # print(ratio_rep_parent)
            # print(ratio_stake_parent)
            # print(f"Selection probability: {p1}")
            # print(f"the round number is: {round}")

            # p2 = 1-(1-p1)**(config.ROUNDS-round+1)



            # print(f"Long term probability: {p2}")
            # # Benefits calculations
            # benefits_continuing = (
            #     p2 * ((node.w + config.REWARD) / (node.W + config.REWARD)) +
            #     (1 - p2) * (node.w / (node.W + config.REWARD))
            # )

    
            # starting_stake_ratio = node.winit/node.Winit

        # data = pd.DataFrame(columns=['round', 'nodeId','stakes'])
        # new_row = pd.DataFrame({'round': [round], 'nodeId': [node.nodeId], 'stakes': [node.w]})
        # data = pd.concat([data, new_row], ignore_index=True) 

        if prob_k <= 0.5:
            allNodes.remove(node)
            # print(f"Node {node.nodeId} leaves the network at round {round}")
        # print(f"Total number of nodes is {len(allNodes)} in round {round}")

    # return data



def calculate_gini(x):
    sum = 0
    n = len(x)
    for i in range(n):
        # print(f"#x: {i}")
        for j in range(n):
            # print(f"#i: {i}")
            # print(f"#j: {j}")
            a=abs(x[i]-x[j])
            # print(f"#a: {a}")
            sum += a
    # print(sum)
    return sum/(2*(n**2)*np.average(x))
