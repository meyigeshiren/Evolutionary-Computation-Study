import random
import math

class binaryGenotype():
	def __init__(self):
		self.fitness = None
		self.gene = None

	def randomInitialization(self, length):
		# TODO: Add random initialization of fixed-length binary gene
		self.gene = [random.randint(0, 1) for _ in range(length)]
		#pass

	def recombine(self, mate, **kwargs):
		child = binaryGenotype()
		
		# TODO: Recombine genes of self and mate and assign to child's gene member variable
		crossover_point = random.randint(1, len(self.gene) - 1)
		child.gene = self.gene[:crossover_point] + mate.gene[crossover_point:]
		#pass

		return child

	def mutate(self, **kwargs):
		copy = binaryGenotype()
		copy.gene = self.gene.copy()
		
		# TODO: mutate gene of copy
		mutation_rate = kwargs.get('mutation_rate', 1.0 / len(self.gene))
		for i in range(len(copy.gene)):
			if random.random() < mutation_rate:
				copy.gene[i] = 1 - copy.gene[i]
		pass

		return copy

	@classmethod
	def initialization(cls, mu, *args, **kwargs):
		population = [cls() for _ in range(mu)]
		for i in range(len(population)):
			population[i].randomInitialization(*args, **kwargs)
		return population
