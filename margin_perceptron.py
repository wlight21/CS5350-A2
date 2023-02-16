import numpy as np
import copy
from perceptron import Perceptron

class MarginPerceptron(Perceptron):
	def __init__(self): Perceptron.__init__(self)
	def perceptron(self, df, epochs, eta, mew):
		self.w = np.array([np.random.uniform(-0.01, 0.01) for i in range(len(df.columns) - 1)])
		self.b = np.random.uniform(-0.01, 0.01)
		per_epoch = []
		for t in range(epochs):
			for i in range(len((permute := df.iloc[np.random.permutation(len(df))]))):
				y = permute.iloc[i][0]
				x = np.array(permute.iloc[i][1:])
				if y * (np.dot(self.w, x) + self.b) <= mew:
					self.w += eta*y*x
					self.b += eta*y
					self.updates += 1
			per_epoch.append(copy.copy(self))
		return per_epoch