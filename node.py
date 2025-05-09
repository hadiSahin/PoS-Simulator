# !/bin/python3

# import config

class Node(object):
	def __init__(self, Id, w, W, parent, sybil, winit, Winit, group, gW, gSize, rep):
		self.nodeId = Id
		self.w = w
		self.W = W
		self.parent = parent
		self.sybil = sybil
		self.winit = winit
		self.Winit = Winit
		self.group = group
		self.gSize = gSize
		self.gW = gW
		self.rep = rep


