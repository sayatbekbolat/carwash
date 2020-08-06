from scipy.spatial import distance
from collections import Counter
import numpy as np
from datetime import datetime
class Car():
	"""docstring for Car"""
	def __init__(self, ID,  inTime, img):
		self.num = []
		self.id = ID,
		self.inTime = inTime
		self.outTime = 'time'
		self.delay = 0
		self.image = img
		self.vector = np.zeros(2048)

	def get_id(self):
		return self.id
	
	def add_num(self, numb):
		if len(self.num)<100:
			self.num.append(numb)
		else:
			self.num = self.num[1:]
			self.num.append(numb)

	def res_num(self):
		if len(self.num)>1:
			c = Counter(self.num)
			return (list(c.keys())[0])
		else:
			return self.num[0]

	def get_last_num(self):
		if len(self.num)>0:
			return self.num[-1]
		else:
			return 'empty'
	
	def get_image(self):
		return self.image
	
	def out(self, outTime):
		self.outTime = outTime
		return self.inTime ,self.res_num(), self.outTime

	def set_vector(self, vector):
		self.vector = vector

	def get_last_vector(self):
		return self.vector


