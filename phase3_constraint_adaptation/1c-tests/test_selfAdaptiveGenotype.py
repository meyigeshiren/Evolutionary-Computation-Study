from test_utils import *
import random, pytest, copy
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from snakeeyes import readConfig
from selfAdaptiveGenotype import selfAdaptiveGenotype

config = readConfig('./configs/green1c1_config.txt', globalVars=globals(), localVars=locals())
iterations = 250
boardsize = 100

class TestUniformRecombination:
	def test_length(self):
		kwargs = copy.deepcopy(config['recombination_kwargs'])
		kwargs['method'] = 'uniform'
		del kwargs['height']
		del kwargs['width']
		for _ in range(iterations):
			size = random.randint(2, boardsize * 5)
			parents = random_pop(2, size, selfAdaptiveGenotype)
			child = parents[0].recombine(parents[1], **kwargs)
			assert len(child.gene) == size
	
	#each locus is selected uniform randomly
	#i.e. every gene has an independent 50/50 chance
	def test_is_uniform(self):
		kwargs = copy.deepcopy(config['recombination_kwargs'])
		kwargs['method'] = 'uniform'
		del kwargs['height']
		del kwargs['width']
		expected_lower_bound = 0.2
		expected_upper_bound = 0.8
		out_of_bounds = 0
		m_iterations = iterations * 100
		ones = all_ones(boardsize, selfAdaptiveGenotype)
		zeroes = all_zeroes(boardsize, selfAdaptiveGenotype)
		hits = [0 for _ in range(boardsize)]
		for _ in range(m_iterations):
			if random.randint(0, 1) == 0:
				child = ones.recombine(zeroes, **kwargs)
			else:
				child = zeroes.recombine(ones, **kwargs)
			num_ones = 0
			for i in range(len(child.gene)):
				if child.gene[i] == 1:
					num_ones += 1
					hits[i] += 1
			ratio = num_ones / boardsize
			if ratio < expected_lower_bound or ratio > expected_upper_bound:
				out_of_bounds += 1
		assert out_of_bounds < 5
		min_bound = 0.4
		max_bound = 0.6
		for hit in hits:
			ratio = hit / m_iterations
			assert ratio < max_bound
			assert ratio > min_bound
	
	#self-adaptive parameter is taken from either parent with 50/50 probability
	def test_self_adaptive_recombination(self):
		kwargs = copy.deepcopy(config['recombination_kwargs'])
		kwargs['method'] = 'uniform'
		del kwargs['height']
		del kwargs['width']
		first = 0
		second = 0
		for _ in range(iterations):
			ones = all_ones(boardsize, selfAdaptiveGenotype)
			zeroes = all_zeroes(boardsize, selfAdaptiveGenotype)
			ones.parameter = random.random()
			zeroes.parameter = random.random()
			child = ones.recombine(zeroes, **kwargs)
			if child.parameter == ones.parameter:
				first += 1
			elif child.parameter == zeroes.parameter:
				second += 1
			else:
				assert False
		assert first >= second * 0.8
		assert first <= second * 1.2
		
		first = 0
		second = 0
		for _ in range(iterations):
			ones = all_ones(boardsize, selfAdaptiveGenotype)
			zeroes = all_zeroes(boardsize, selfAdaptiveGenotype)
			ones.parameter = random.random()
			zeroes.parameter = random.random()
			child = zeroes.recombine(ones, **kwargs)
			if child.parameter == ones.parameter:
				second += 1
			elif child.parameter == zeroes.parameter:
				first += 1
			else:
				assert False
		assert first >= second * 0.8
		assert first <= second * 1.2

	#parents are not modified at all by recombination
	def test_parents_unmodified(self):
		kwargs = copy.deepcopy(config['recombination_kwargs'])
		kwargs['method'] = 'uniform'
		del kwargs['height']
		del kwargs['width']
		for _ in range(iterations):
			size = random.randint(2, boardsize * 5)
			parents = random_pop(2, size, selfAdaptiveGenotype)
			random_fitness(parents)
			copies = [copy.deepcopy(x) for x in parents]
			child = parents[0].recombine(parents[1], **kwargs)
			assert same_object(copies[0], parents[0])
			assert same_object(copies[1], parents[1])

#class Test1PointCrossoverRecombination:
