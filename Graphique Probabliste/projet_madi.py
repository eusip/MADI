from math import log as log
from math import exp as exp
from copy import copy
from functools import reduce
import random
import pyAgrum as gum
import pyAgrum.lib.notebook as gnb
import numpy as np 
import pydotplus as dot

class FactorGraph(gum.UndiGraph):
	def __init__(self):
		super().__init__()
		# key: nid; value: potential object
		self.node_cpt = {}  
		# key: nid; value: 'variable', 'factor' 
		self.node_type = {}
		# key: (nid, 'parents'/'children'); value: list of neighbors
		self.neighbors = {}
		# key: (recipient nid, sender nid); value: message
		self.messages = {}  
		self.bn = None

	def addVariable(self, node):
		"""node: node in the BN"""
		# returns node value in the undiGraph
		nid = self.addNode()  
		self.node_cpt[nid] = []
		self.node_type[nid] = 'variable'

	def addFactor(self, node):
		"""node: node in the BN"""
		# returns node value in the undiGraph
		nid = self.addNode()  
		self.node_cpt[nid] = self.bn.cpt(node)
		self.node_type[nid] = 'factor'

		return nid

	def build(self, bn):
		"""bn: gum.BayesNet"""
		self.bn = bn
		# add variables to undirected graph
		for node in self.bn.nodes():
			self.addVariable(node)
			self.neighbors[(node, 'children')] = []
			self.neighbors[(node, 'parents')] = []
			self.node_cpt[node] = []
		# add edges between variables in undirected graph
		for u, v in self.bn.arcs():
			self.addEdge(u, v)
		# insert factors into undirected graph
		for node in self.bn.nodes():
			nid = self.addFactor(node)
			# initialize nodes in dict `neighbors`
			self.neighbors[(nid, 'children')] = []
			self.neighbors[(nid, 'parents')] = []
			parents = self.bn.parents(node)
			for parent in parents:
				# erase original edges and add new edges between variables and factors
				self.eraseEdge(parent, node)
				self.addEdge(parent, nid)
				# initialize nodes in dict `neighbors`
				self.neighbors[(parent, 'children')].append(nid)
				self.neighbors[(nid, 'parents')].append(parent)
			self.addEdge(nid, node)
			self.neighbors[(nid, 'children')].append(node)
			self.neighbors[(node, 'parents')].append(nid)

	def getVariables(self):
		var_list = []
		for key, value in self.node_type.items():
			if value == 'variable':
				var_list.append(key)

		return var_list

	def getFactors(self):
		fac_list = []
		for key, value in self.node_type.items():
			if value == 'factor':
				fac_list.append(key)

		return fac_list

	def show(self):
		var_names = []

		dot_data = """graph FG {\n\tlayout = neato;
			\n// variables
			\nnode [shape = rectangle, margin = 0.04,
			width = 0, height = 0, style = filled, color = "coral"];\n"""

		for node in self.nodes():
			if self.node_type[node] == 'variable':
				dot_data += str(self.bn.variable(node).name()) + ";"
				# this works because variable nodes in UndiGraph and BN
				# are numbered the same

		dot_data += """\n\n// factors
			\nnode [shape=point,width=0.1,height=0.1, style=filled,color="burlywood"];\n"""

		for node in self.nodes():
			if self.node_type[node] == 'factor':
				dot_data += str(node) + ";\n"

		dot_data += """\n//variable - factor edges\nedge [len=0.7];\n"""

		dot_data += "\n\n"

		# for key, value in self.node_type.items():
		# 	if value == 'variable':
		# 		var_names.append(key)

		for u, v in self.edges():
			if u in self.getVariables():
				u = self.bn.variable(u).name()
			if v in self.getVariables():
				v = self.bn.variable(v).name()
			dot_data += str(u) + "--" + str(v) + ";\n"

		dot_data += "}"

		# print(str(dot_data))

		return dot_data


class TreeSumProductInference():
	def __init__(self, fg):
		self.fg = fg
		self.root = None

	def getRootNodes(self):
		"""This function determines the root node for the message passing algorithm"""
		root_list = []
		for node in self.fg.nodes():
			if self.fg.neighbors[(node, 'children')] == [] and self.fg.node_type[node] == 'variable':
				root_list.append(node)

		return root_list

	def getStartNodes(self):
		"""This function determines the node from which the message passing algorithm starts"""
		start_list = []
		for node in self.fg.nodes():
			if self.fg.neighbors[(node, 'parents')] == [] and self.fg.node_type[node] == 'factor':
				start_list.append(node)

		return start_list

	def makeInference(self):
		"""This algorithm implements the sum-product message passing algorithm"""

		for i in range(len(self.fg.getVariables())):
			self.fg.node_cpt[i] = []

		self.fg.messages = {}

		# print('forward pass')
		
		descendants = []

		for node in self.getStartNodes():
			children = self.fg.neighbors[(node, 'children')]
			# print('start node: {}, children: {}'.format(node, children))
			for child in children:
				self.fg.messages[(child, node)] = self.fg.node_cpt[node].margSumIn(
					self.fg.bn.variable(child).name())
			
			descendants.extend(child for child in children if child not in descendants)

		# print('descendants are: ', str(descendants))

		while descendants:	
			nid = descendants.pop(0)
			children = self.fg.neighbors[(nid, 'children')]
			parents = self.fg.neighbors[(nid, 'parents')]
			# print('nid {}, parents {}, children {}'.format(nid, parents, children))
			if self.fg.node_type[nid] == 'variable':
				msg_cnt = 0
				for parent in parents:  # store a copy of cpt table for computing posterior
					# print('the message from {} to {} is: {}'.format(
					# 	parent, nid, self.fg.messages[nid, parent]))
					self.fg.node_cpt[nid].append(copy(self.fg.messages[nid, parent]))
					if (nid, parent) in self.fg.messages.keys():
						msg_cnt += 1
				for child in children:  
					# compute product of all cpt tables
					if msg_cnt == 1:
						# print('child {}, node {}'.format(child, nid))
						self.fg.messages[(child, nid)] = self.fg.messages[(nid, parent)]
					else:
						self.fg.messages[(child, nid)] = reduce(
							lambda x, y: x*y, [self.fg.messages[(nid, parent)] for parent in parents])
			else:  # 'factor'
				msg_cnt = 0
				for parent in parents:
					if (nid, parent) in self.fg.messages.keys():
						msg_cnt += 1
				if msg_cnt == 1:
					for parent in parents:
						if (nid, parent) in self.fg.messages.keys():
							send_message = self.fg.messages[(nid, parent)] * self.fg.node_cpt[nid]
				else:
					send_message = reduce(lambda x, y: x*y, \
						[self.fg.messages[(nid, parent)] for parent in parents]) * self.fg.node_cpt[nid]
				for child in children:
					# print('the message from {} to {} is: {}'.format(nid, child, \
					# 	send_message.margSumIn(self.fg.bn.variable(child).name())))
					self.fg.messages[(child, nid)] = send_message.margSumIn(
						self.fg.bn.variable(child).name()).normalize()
			
			descendants.extend(child for child in children if child not in descendants)
			# print('the descendants of this generation of nodes is: {} '.format(str(descendants)))

		# print('backwards pass')

		descendants = []

		for node in self.getRootNodes():
			parents = self.fg.neighbors[(node, 'parents')]
			# print('node {}, parents {}'.format(node, parents))
			for parent in parents:
				self.fg.messages[(parent, node)] = 1
			
			descendants.extend(parent for parent in parents if parent not in descendants)

		# print('the descendants of this generation of nodes is: {} '.format(str(descendants)))

		while descendants:	
			nid = descendants.pop(0)
			children = self.fg.neighbors[(nid, 'children')]
			parents = self.fg.neighbors[(nid, 'parents')]
			# print('nid {}, parents {}, children {}'.format(nid, parents, children))
			if self.fg.node_type[nid] == 'variable':
				msg_cnt = 0
				for child in children:
					# store a copy of cpt table for computing posterior
					# print('the message from {} to {} is: {}'.format(nid, child, \
					# 	self.fg.messages[(nid, child)]))
					self.fg.node_cpt[nid].append(copy(self.fg.messages[(nid, child)]))
					if (nid, child) in self.fg.messages.keys():
						msg_cnt += 1
				for parent in parents:
					if msg_cnt == 1:
						# print('parent {}, node {}'.format(parent, nid))
						self.fg.messages[(parent, nid)] = self.fg.messages[(nid, child)]
					else:
						self.fg.messages[(parent, nid)] = reduce(
							lambda x, y: x*y, [self.fg.messages[(nid, child)] for child in children])
			else:  # 'factor'
				msg_cnt = 0
				for child in children:
					if (nid, child) in self.fg.messages.keys():
						msg_cnt += 1
				if msg_cnt == 1:
					for child in children:
						if (nid, child) in self.fg.messages.keys():
							if self.fg.messages[(nid, child)] == 1:
								send_message = self.fg.node_cpt[nid]
							else:
								send_message = self.fg.messages[(nid, child)] * self.fg.node_cpt[nid]
							# print('the message from {} to {} is: {}'.format(nid, parent, \
							# 	send_message.margSumIn(self.fg.bn.variable(parent).name()).normalizeAsCPT())
				else:
					for child in children:
						if (nid, child) in self.fg.messages.keys():
							if self.fg.messages[(nid, child)] == 1:
								send_message *= self.fg.node_cpt[child]
							else:
								send_message *= self.fg.messages[(nid, child)]
								print(send_message)
						else:
							send_message *= self.fg.node_cpt[nid]
						send_message = reduce(lambda x, y: x*y, \
							[self.fg.messages[(nid, child)] for child in children]) * self.fg.node_cpt[nid]
				for parent in parents:
					self.fg.messages[(parent, nid)] = send_message.margSumIn(
							self.fg.bn.variable(parent).name()).normalize()
		
			descendants.extend(parent for parent in parents if parent not in descendants)
			# print('the descendants of this generation of nodes is: {} '.format(str(descendants)))
			
	def posterior(self, name):
		"""name: name of the node in BN"""
		nid = self.fg.bn.idFromName(name)
		cpt = reduce(lambda x, y: x*y, self.fg.node_cpt[nid]).margSumIn(
			self.fg.bn.variable(nid).name()).normalizeAsCPT()
		
		return cpt


class TreeMaxProductInference():
	def __init__(self, fg):
		self.fg = fg
		self.root = None

	def getRootNodes(self):
		"""This function determines the root node for the message passing algorithm"""
		root_list = []
		for node in self.fg.nodes():
			if self.fg.neighbors[(node, 'children')] == [] and self.fg.node_type[node] == 'variable':
				root_list.append(node)

		return root_list

	def getStartNodes(self):
		"""This function determines the node from which the message passing algorithm starts"""
		start_list = []
		for node in self.fg.nodes():
			if self.fg.neighbors[(node, 'parents')] == [] and self.fg.node_type[node] == 'factor':
				start_list.append(node)

		return start_list

	def makeInference(self):
		"""This algorithm implements the max-product message passing algorithm"""

		for i in range(len(self.fg.getVariables())):
			self.fg.node_cpt[i] = []

		self.fg.messages = {}

		# print('forward pass')

		descendants = []

		for node in self.getStartNodes():
			children = self.fg.neighbors[(node, 'children')]
			# print('start node: {}, parents: {}'.format(node, parents))
			for child in children:
				self.fg.messages[(child, node)] = self.fg.node_cpt[node].margMaxIn(
						self.fg.bn.variable(child).name())
			
			descendants.extend(child for child in children if child not in descendants)

		# print('descendants of the start node are: ', str(descendants))

		while descendants:	
			nid = descendants.pop(0)
			children = self.fg.neighbors[(nid, 'children')]
			parents = self.fg.neighbors[(nid, 'parents')]
			# print('nid {}, parents {}, children {}'.format(nid, parents, children))
			if self.fg.node_type[nid] == 'variable':
				msg_cnt = 0
				for parent in parents:  # store a copy of cpt table for computing posterior
					# print('the message from {} to {} is: {}'.format(
					# 	parent, nid, self.fg.messages[nid, parent]))
					self.fg.node_cpt[nid].append(copy(self.fg.messages[nid, parent]))
					if (nid, parent) in self.fg.messages.keys():
						msg_cnt += 1
				for child in children:
					# compute product of all cpt tables
					if msg_cnt == 1:
						# print('child {}, node {}'.format(child, nid))
						self.fg.messages[(child, nid)] = self.fg.messages[(nid, parent)]
					else:
						self.fg.messages[(child, nid)] = reduce(
							lambda x, y: x*y, [self.fg.messages[(nid, parent)] for parent in parents])
			else:  # 'factor'
				msg_cnt = 0
				for parent in parents:
					if (nid, parent) in self.fg.messages.keys():
						msg_cnt += 1
				if msg_cnt == 1:
					for parent in parents:
						if (nid, parent) in self.fg.messages.keys():
							send_message = self.fg.messages[(nid, parent)] * self.fg.node_cpt[nid]
				else:
					send_message = reduce(lambda x, y: x*y, \
						[self.fg.messages[(nid, parent)] for parent in parents]) * self.fg.node_cpt[nid]
				for child in children:
					# print('the message from {} to {} is: {}'.format(nid, child, \
					# 	send_message.margMaxIn(self.fg.bn.variable(child).name())))
					self.fg.messages[(child, nid)] = send_message.margMaxIn(
						self.fg.bn.variable(child).name()).normalize()
			
			descendants.extend(child for child in children if child not in descendants)
			# print('the descendants of this generation of nodes are: {} '.format(str(descendants)))

	def argmax(self):
		descendants = []		
		max_config = {}

		for node in self.fg.getVariables():
			max_config.update(reduce(lambda x, y: x*y, self.fg.node_cpt[node]).argmax()[0])

		return max_config


class TreeMaxSumInference():
	def __init__(self, fg):
		self.fg = fg
		self.root = None

	def getRootNodes(self):
		"""This function determines the root node for the message passing algorithm"""
		root_list = []
		for node in self.fg.nodes():
			if self.fg.neighbors[(node, 'children')] == [] and self.fg.node_type[node] == 'variable':
				root_list.append(node)

		return root_list

	def getStartNodes(self):
		"""This function determines the set of nodes from which the message passing algorithm starts"""
		start_list = []
		for node in self.fg.nodes():
			if self.fg.neighbors[(node, 'parents')] == []:
				start_list.append(node)

		return start_list

	def getLnCPT(self, cpt):
		"""This function transforms the probabilities in a cpt to log-probabilities."""
		log_prob = [np.log(i) for i in np.array(cpt.tolist()).flatten()]
		
		return cpt.fillWith(log_prob)

	def getExpCPT(self, cpt):
		"""This function transforms log-probabilites in a cpt to probabilities"""
		prob = [np.exp(i) for i in np.array(cpt.tolist()).flatten()]

		return cpt.fillWith(prob)

	def makeInference(self):
		"""This algorithm implements the max-sum message passing algorithm"""
		
		for i in range(len(self.fg.getVariables())):
			self.fg.node_cpt[i] = []

		self.fg.messages = {}

		# print('forward pass')

		descendants = []

		for node in self.getStartNodes():
			children = self.fg.neighbors[(node, 'children')]
			# print('start node: {}, parents: {}'.format(node, parents))
			for child in children:
				self.fg.messages[(child, node)] = self.getExpCPT(self.fg.node_cpt[node]).margMaxIn(
						self.fg.bn.variable(child).name())
			
			descendants.extend(child for child in children if child not in descendants)
		
		# print('descendants are: ', str(descendants))

		# print(self.fg.messages)

		while descendants:	
			nid = descendants.pop(0)
			# print('\n\n node', nid,'\n\n')
			children = self.fg.neighbors[(nid, 'children')]
			parents = self.fg.neighbors[(nid, 'parents')]
			# print('nid {}, parents {}, children {}'.format(nid, parents, children))
			if self.fg.node_type[nid] == 'variable':
				msg_cnt = 0
				for parent in parents:  # store a copy of cpt table for computing posterior
					# print('the message from {} to {} is: {}'.format(
					# 	parent, nid, self.fg.messages[nid, parent]))
					self.fg.node_cpt[nid].append(copy(self.fg.messages[nid, parent]))
					if (nid, parent) in self.fg.messages.keys():
						msg_cnt += 1
				for child in children:
					# compute product of all cpt tables
					if msg_cnt == 1:
						# print('single message: {}'.format(self.fg.messages[(nid, parent)]))
						self.fg.messages[(child, nid)] = self.fg.messages[(nid, parent)]
					else:
						# print('multiple messages: {}'.format([self.fg.messages[(nid, parent)] for parent in parents]))
						self.fg.messages[(child, nid)] = self.getExpCPT(reduce(
							lambda x, y: self.getLnCPT(x)+self.getLnCPT(y)) * self.fg.node_cpt[nid])
			else:  # 'factor'
				msg_cnt = 0
				for parent in parents:
					if (nid, parent) in self.fg.messages.keys():
						msg_cnt += 1
				if msg_cnt == 1:
					for parent in parents:
						if (nid, parent) in self.fg.messages.keys():
							# print('single message: {}'.format(self.fg.messages[(nid, parent)]))
							send_message = self.fg.messages[(nid, parent)] * self.fg.node_cpt[nid]
				else:
					# print('multiple messages: {}'.format([self.fg.messages[(nid, parent)] for parent in parents]))
					send_message = self.getExpCPT(reduce(lambda x, y: self.getLnCPT(x)+self.getLnCPT(y), \
						[self.fg.messages[(nid, parent)] for parent in parents])) * self.fg.node_cpt[nid]
				for child in children:
					# print('the message from {} to {} is: {}'.format(nid, child, \
					# 	send_message.margMaxIn(self.fg.bn.variable(child).name()).normalize()))
					self.fg.messages[(child, nid)] = send_message.margMaxIn(
						self.fg.bn.variable(child).name()).normalize()
			
			descendants.extend(child for child in children if child not in descendants)
			# print('the descendants of this generation of nodes is: {} '.format(str(descendants))
			
		# print(self.fg.messages)

	def argmax(self):
		max_config = {}

		for node in self.fg.getVariables():
			# print(self.fg.node_cpt[node][0])
			max_config.update(self.fg.node_cpt[node][0].argmax()[0])

		return max_config


class LBPSumProductInference():
	def __init__(self, fg):
		self.fg = fg
		# key: (nid); value: node which sent last message
		self.sender = {}
		# key: (nid); value: posterior of the variable
		self.history = {}
		# key: (nid); value: True/False
		self.convergence = [1 for _ in range(len(self.fg.getVariables()))]
		# difference between current posterior value and last posterior value for index 1
		self.epsilon = 1e-2

	def addFactor(self, node, key, value):
		"""node: variable node in the factor graph
			key: node name
			value: evidence value"""
		nid = self.fg.addNode()
		if value == 1:
			self.fg.node_cpt[nid] = (copy(self.fg.bn.cpt(node)).margSumIn(
										self.fg.bn.variable(key).name())).fillWith([0, 1])
		else:
			self.fg.node_cpt[nid] = (copy(self.fg.bn.cpt(node)).margSumIn(
										self.fg.bn.variable(key).name())).fillWith([1, 0])
		
		self.fg.node_type[nid] = 'factor'

		# print('CPT of the added factor is {}'.format(self.fg.node_cpt[nid]))

		return nid

	def addEvidence(self, evidence):
		"""Adds evidence to the factor graph"""
		for key, value in evidence.items():
			node = self.fg.bn.idFromName(key)
			neighbors = self.fg.neighbors[(node, 'children')] + self.fg.neighbors[(node, 'parents')]
			if len(neighbors) >= 2 and len(self.fg.neighbors[(node, 'parents')]) == 1:
				if value  == 1:
					self.fg.node_cpt[self.fg.neighbors[(node, 'parents')][0]].fillWith([0, 1])
				else:
					self.fg.node_cpt[self.fg.neighbors[(node, 'parents')][0]].fillWith([1, 0])
				# print(self.fg.node_cpt[self.fg.neighbors[(node, 'parents')][0]])
			else:
				factor = self.addFactor(node, key, value)
				self.fg.addEdge(factor, node)				
				self.fg.messages[(node, factor)] = self.fg.node_cpt[factor]
				self.fg.messages[(factor, node)] = 1
				self.fg.neighbors[(factor, 'children')] = []
				self.fg.neighbors[(factor, 'parents')] = [node]
				self.fg.neighbors[(node, 'children')] = [factor]
				# print('node: {}, factor {}, key {}:'.format(node, factor, key))
				# print(self.fg.node_cpt[factor])

	def makeInference(self):
		"""This algorithm implements the loopy belief sum-product message passing algorithm"""
		self.fg.messages = {}

		for node in range(len(self.fg.getVariables())):
			self.history[node] = []

		for nid in self.fg.getFactors():
			neighbors = self.fg.neighbors[(nid, 'children')] + self.fg.neighbors[(nid, 'parents')]
			for neighbor in neighbors:
				self.fg.messages[(neighbor, nid)] = self.fg.node_cpt[nid]
		
		for nid in self.fg.getVariables():
			neighbors = self.fg.neighbors[(nid, 'children')] + self.fg.neighbors[(nid, 'parents')]
			for neighbor in neighbors:
				self.fg.messages[(neighbor, nid)] = 1

		while True:
			for nid in self.fg.getVariables():
				neighbors = self.fg.neighbors[(nid, 'children')] + self.fg.neighbors[(nid, 'parents')]
				incoming = [v for k, v in self.fg.messages.items() if nid == k[0]]
				
				for neighbor in neighbors:
						if len(incoming) > 1:
							# print('child {}, node {}'.format(neighbor, nid))
							self.fg.messages[(neighbor, nid)] = reduce(lambda x, y: x*y, incoming)
						else:
							self.fg.messages[(neighbor, nid)] = incoming[0]			
			
			for nid in self.fg.getFactors():
				neighbors = self.fg.neighbors[(nid, 'children')] + self.fg.neighbors[(nid, 'parents')]
				incoming = [message for message in [v for k, v in self.fg.messages.items() if nid == k[0]] \
							if message != 1]
				
				for neighbor in neighbors:
	
							if len(incoming) > 1:
								# print('child {}, node {}'.format(neighbor, nid))
								self.fg.messages[(neighbor, nid)] = reduce(lambda x, y: x*y, incoming) * \
									self.fg.node_cpt[nid].margSumIn(self.fg.bn.variable(neighbor).name())
							else:
								if incoming[0] == 1:
									self.fg.messages[(neighbor, nid)] = self.fg.node_cpt[nid].margSumIn(
										self.fg.bn.variable(neighbor).name())
								else:
									self.fg.messages[(neighbor, nid)] = (incoming[0] * \
										self.fg.node_cpt[nid]).margSumIn(self.fg.bn.variable(neighbor).name())
				
			# print(self.fg.messages)

			for nid in self.fg.getVariables():
				self.history[nid].append(reduce(lambda x, y: x*y, 
					[v for k, v in self.fg.messages.items() if nid == k[0]]))
				# print('The history of node {} is:\n\n {}\n\n'.format(nid, self.history[nid]))
				if len(self.history[nid]) > 1:
					# print('\n\n', self.history[nid], '\n\n')
					next_last = len(self.history[nid]) - 2 
					last =  len(self.history[nid]) - 1
					# print('\n\n','next last {}, last {}'.format(next_last, last),'\n\n')
					self.convergence[nid] = abs(self.history[nid][last].margSumIn(
												self.fg.bn.variable(nid).name()).normalize().max()- \
											self.history[nid][next_last].margSumIn(
												self.fg.bn.variable(nid).name()).normalize().max())
					# print('{}: {} - {} = {}'.format(nid, self.history[nid][last].margSumIn(
					# 	self.fg.bn.variable(nid).name()).normalize().max(), 
					# 	self.history[nid][next_last].margSumIn(
					# 	self.fg.bn.variable(nid).name()).normalize().max(), self.convergence[nid]))
					# print(self.convergence)
					# print(self.epsilon)
			if any(i < self.epsilon for i in self.convergence):
				break
			
	def posterior(self, name):
		"""name: name of the node in BN"""
		nid = self.fg.bn.idFromName(name)
		cpt = self.history[nid][len(self.history[nid]) - 1].margSumIn(
				self.fg.bn.variable(nid).name()).normalizeAsCPT()
		
		return cpt


class LBPMaxSumInference():
	def __init__(self, fg):
		self.fg = fg
		# key: (nid); value: node which sent last message
		self.sender = {}
		# key: (nid); value: posterior of the variable
		self.history = {}
		# key: (nid); value: True/False
		self.convergence = [1 for _ in range(len(self.fg.getVariables()))]
		# difference between current posterior value and last posterior value for index 1
		self.epsilon = 1e-2

	def addFactor(self, node, key, value):
		"""node: variable node in the factor graph
			key: node name
			value: evidence value"""
		nid = self.fg.addNode()
		if value == 1:
			self.fg.node_cpt[nid] = (copy(self.fg.bn.cpt(node)).margSumIn(
										self.fg.bn.variable(key).name())).fillWith([0, 1])
		else:
			self.fg.node_cpt[nid] = (copy(self.fg.bn.cpt(node)).margSumIn(
										self.fg.bn.variable(key).name())).fillWith([1, 0])
		
		self.fg.node_type[nid] = 'factor'

		# print('CPT of the added factor is {}'.format(self.fg.node_cpt[nid]))

		return nid

	def addEvidence(self, evidence):
		"""Adds evidence to the factor graph"""
		for key, value in evidence.items():
			node = self.fg.bn.idFromName(key)
			neighbors = self.fg.neighbors[(node, 'children')] + self.fg.neighbors[(node, 'parents')]
			if len(neighbors) >= 2 and len(self.fg.neighbors[(node, 'parents')]) == 1:
				if value  == 1:
					self.fg.node_cpt[self.fg.neighbors[(node, 'parents')][0]].fillWith([0, 1])
				else:
					self.fg.node_cpt[self.fg.neighbors[(node, 'parents')][0]].fillWith([1, 0])
				# print(self.fg.node_cpt[self.fg.neighbors[(node, 'parents')][0]])
			else:
				factor = self.addFactor(node, key, value)
				self.fg.addEdge(factor, node)				
				self.fg.messages[(node, factor)] = self.fg.node_cpt[factor]
				self.fg.messages[(factor, node)] = 1
				self.fg.neighbors[(factor, 'children')] = []
				self.fg.neighbors[(factor, 'parents')] = [node]
				self.fg.neighbors[(node, 'children')] = [factor]
				# print('node: {}, factor {}, key {}:'.format(node, factor, key))
				# print(self.fg.node_cpt[factor])

	def getLnCPT(self, cpt):
		"""This function transforms the probabilities in a cpt to log-probabilities."""
		log_prob = [np.log(i) for i in np.array(cpt.tolist()).flatten()]
		
		return cpt.fillWith(log_prob)

	def getExpCPT(self, cpt):
		"""This function transforms log-probabilites in a cpt to probabilities"""
		prob = [np.exp(i) for i in np.array(cpt.tolist()).flatten()]

		return cpt.fillWith(prob)

	def makeInference(self):
		"""This algorithm implements the loopy belief sum-product message passing algorithm"""
		self.fg.messages = {}

		for node in range(len(self.fg.getVariables())):
			self.history[node] = []

		for nid in self.fg.getFactors():
			neighbors = self.fg.neighbors[(nid, 'children')] + self.fg.neighbors[(nid, 'parents')]
			for neighbor in neighbors:
				self.fg.messages[(neighbor, nid)] = self.fg.node_cpt[nid].normalize()

		for nid in self.fg.getVariables():
			neighbors = self.fg.neighbors[(nid, 'children')] + self.fg.neighbors[(nid, 'parents')]
			for neighbor in neighbors:
				self.fg.messages[(neighbor, nid)] = 0

		# print('\n\n before variables: \n\n', self.fg.messages,'\n\n')


		while True:
			for nid in self.fg.getVariables():
				neighbors = self.fg.neighbors[(nid, 'children')] + self.fg.neighbors[(nid, 'parents')]
				incoming = [v for k, v in self.fg.messages.items() if nid == k[0]]
				print(incoming)
				for neighbor in neighbors:
						if len(incoming) > 1:
							# print('child {}, node {}'.format(neighbor, nid))
							self.fg.messages[(neighbor, nid)] = self.getExpCPT(reduce(lambda x, y: self.getLnCPT(x)+self.getLnCPT(y), incoming))
						else:
							self.fg.messages[(neighbor, nid)] = incoming[0]			
			
			# print('\n\n after variables: \n\n', self.fg.messages,'\n\n')

			for nid in self.fg.getFactors():
				neighbors = self.fg.neighbors[(nid, 'children')] + self.fg.neighbors[(nid, 'parents')]
				incoming = [message for message in [v for k, v in self.fg.messages.items() if nid == k[0]] \
							if message != 0]
				
				for neighbor in neighbors:
	
							if len(incoming) > 1:
								# print('child {}, node {}'.format(neighbor, nid))
								self.fg.messages[(neighbor, nid)] = (self.getExpCPT(reduce(lambda x, y: self.getLnCPT(x)+self.getLnCPT(y), incoming)) * \
									self.fg.node_cpt[nid]).margMaxIn(self.fg.bn.variable(neighbor).name())
							else:
								if incoming[0] == 0:
									self.fg.messages[(neighbor, nid)] = self.fg.node_cpt[nid].margMaxIn(
										self.fg.bn.variable(neighbor).name())
								else:
									self.fg.messages[(neighbor, nid)] = (incoming[0] * \
										self.fg.node_cpt[nid]).margMaxIn(self.fg.bn.variable(neighbor).name())
				
			# print('\n\n after factor: \n\n', self.fg.messages,'\n\n')

			for nid in self.fg.getVariables():
				self.history[nid].append(reduce(lambda x, y: x*y, 
					[v for k, v in self.fg.messages.items() if nid == k[0]]))
				# print('The history of node {} is:\n\n {}\n\n'.format(nid, self.history[nid]))
				if len(self.history[nid]) > 1:
					# print('\n\n', self.history[nid], '\n\n')
					next_last = len(self.history[nid]) - 2 
					last =  len(self.history[nid]) - 1
					# print('\n\n','next last {}, last {}'.format(next_last, last),'\n\n')
					self.convergence[nid] = abs(self.history[nid][last].margMaxIn(
												self.fg.bn.variable(nid).name()).normalize().max()- \
											self.history[nid][next_last].margMaxIn(
												self.fg.bn.variable(nid).name()).normalize().max())
					# print('{}: {} - {} = {}'.format(nid, self.history[nid][last].margSumIn(
						# self.fg.bn.variable(nid).name()).normalize().max(), 
						# self.history[nid][next_last].margSumIn(
						# self.fg.bn.variable(nid).name()).normalize().max(), self.convergence[nid]))
					# print(self.convergence)
					# print(self.delta)
			if all(_ < self.epsilon for _ in self.convergence):
				break
			
	def posterior(self, name):
		"""name: name of the node in BN"""
		nid = self.fg.bn.idFromName(name)
		cpt = self.history[nid][len(self.history[nid]) - 1].margMaxIn(
				self.fg.bn.variable(nid).name()).normalizeAsCPT()
		
		return cpt