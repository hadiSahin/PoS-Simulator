# !/bin/python3

# import config

class Node(object):
	def __init__(self, Id, w, W, parent, splits):
		self.nodeId = Id
		self.w = w
		self.W = W
		self.parent = parent
		self.splits = splits


