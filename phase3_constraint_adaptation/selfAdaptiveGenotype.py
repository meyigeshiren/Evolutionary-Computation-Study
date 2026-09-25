from binaryGenotype import binaryGenotype
import random

class selfAdaptiveGenotype(binaryGenotype):
	def __init__(self):
		super().__init__()
		self.parameter = None

	def randomInitialization(self, **kwargs):
		min_parameter = kwargs.pop('min_parameter', 0)
		max_parameter = kwargs.pop('max_parameter', 1)
		super().randomInitialization(**kwargs)
		if min_parameter > max_parameter:
			min_parameter, max_parameter = max_parameter, min_parameter
		self.parameter = random.uniform(min_parameter, max_parameter)

	def recombine(self, mate, **kwargs):
		child = selfAdaptiveGenotype()
		child.gene = super().recombine(mate, **kwargs).gene
		child.parameter = random.choice([self.parameter, mate.parameter])
		return child

	def mutate(self, **kwargs):
		copy = selfAdaptiveGenotype()
		copy.gene = super().mutate(**kwargs).gene
		copy.parameter = self.parameter
		min_parameter = kwargs.get('min_parameter', 0)
		max_parameter = kwargs.get('max_parameter', 1)
		if min_parameter > max_parameter:
			min_parameter, max_parameter = max_parameter, min_parameter
		copy.parameter = random.uniform(min_parameter, max_parameter)
		return copy
