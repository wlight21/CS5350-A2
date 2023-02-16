import numpy as np
import copy
from perceptron import Perceptron

class AveragePerceptron(Perceptron):
	def __init__(self): Perceptron.__init__(self)
	def perceptron(self, df, epochs, eta, mew=0):
		self.w = np.array([np.random.uniform(-0.01, 0.01) for i in range(len(df.columns) - 1)])
		self.b = np.random.uniform(-0.01, 0.01)
		self.avg_w = self.sum_w = np.zeros(len(self.w))
		self.avg_b = self.sum_b = 0
		per_epoch = []
		for t in range(epochs):
			examples = 0
			for i in range(len((permute := df.iloc[np.random.permutation(len(df))]))):
				examples += 1
				y = permute.iloc[i][0]
				x = np.array(permute.iloc[i][1:])
				if y * (np.dot(self.w, x) + self.b) < mew:
					self.w += eta*y*x
					self.b += eta*y
					self.updates += 1
				self.sum_w += self.w
				self.sum_b += self.b
			self.avg_w = self.sum_w / examples
			self.avg_b = self.sum_b / examples
			per_epoch.append(copy.copy(self))
		return per_epoch
	def predict(self, x): return self.sign(np.dot(self.avg_w, np.array(x)) + self.avg_b)