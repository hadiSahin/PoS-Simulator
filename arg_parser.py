import argparse
import config 

parser = argparse.ArgumentParser()
parser.add_argument('-n','-nodes', action='store', dest='max_nodes',
                    default=config.MAX_NODES,
                    help='set MAX_NODES', type=int)
parser.add_argument('-rounds', action='store', dest='rounds',
                    default=config.ROUNDS,
                    help='set ROUNDS', type=int)
parser.add_argument('-reward', action='store', dest='reward',
                    default=config.REWARD,
                    help='set REWARD rate', type=int)
# parser.add_argument('-transfer', action='store', dest='transfer_rate',
#                     default=config.RATE,
#                     help='set TRANSFER RATE', type=int)
parser.add_argument('-i','-incentives', action='store', default = 'pure', dest='incentives',
                    choices=['pure', 'pool', 'geometric', 'algorand','universal','boosted'], type=str)
parser.add_argument('-d','-dist', action='store', default = 'pareto', dest='distribution',
                    choices=['normal', 'pareto', 'uniform','uniswap'], type=str)
parser.add_argument('-nc','-node_count', action='store', dest='node_count', default = 'fixed', 
                    choices=['fixed', 'decreasing'], type=str)
parser.add_argument('-thresh', action='store', dest='threshold',
                    default=config.WEALTH_THRESHOLD,
                    help='set wealth threshold', type=int)
parser.add_argument('-tax', action='store', dest='taxRate',
                    default=config.TAX_RATE,
                    help='set wealth threshold', type=int)
# parser.add_argument('-cost', action='store', dest='cost',
#                     default=config.COST_STAY,
#                     help='set cost of staying', type=int)
parser.add_argument('-period', action='store', dest='period', default = config.PERIOD, 
                    help='set PERIOD', type=int)
parser.add_argument('-sp','-split', action='store', dest='splitting', default = 'noSybil', 
                    choices=['Sybil', 'noSybil'], type=str)
parser.add_argument('-p','-punish', action='store', dest='punishment', default = 'False', 
                    choices=['False', 'True'], type=str)
# parser.add_argument('-slash', action='store', dest='slash',
#                     default=config.SLASH,
#                     help='set SLASH RATE', type=float)
# parser.add_argument('-prob_caught', action='store', dest='prob_caught',
#                     default=config.PROB_CAUGHT,
#                     help='set PROBABILITY OF BEING CAUGHT', type=float)
