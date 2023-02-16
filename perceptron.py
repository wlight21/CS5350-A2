import numpy as np
from string_builder import StringBuilder

class Perceptron:
	def __init__(self): 
		np.random.seed(131324)
		self.w = None
		self.b = None
		self.updates = 0

	def sign(self, x): return 1 if x > 0 else -1
	def predict(self, x): return self.sign(np.dot(self.w, np.array(x)) + self.b)

	def accuracy(self, df):
		acc = 0
		for i in range(len(df)):
			if df.iloc[i][0] * self.predict(df.iloc[i][1:]) >= 0: acc += 1
		return acc / len(df)

	def perceptron(self, df, epochs, eta, mew=0): pass

