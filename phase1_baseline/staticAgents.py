import gpac
from collections import deque
import random

class shortestPathPillAgent():
	def __init__(self):
		self.pre_planned_actions = deque()

	def select_action(self, game):
		if len(self.pre_planned_actions) == 0:
			self.pre_planned_actions = path_to_pill('m', game)
		return self.pre_planned_actions.popleft()

class shortestPathFruitAgent(shortestPathPillAgent):
	def select_action(self, game):
		if game.fruit_location is not None and len(self.pre_planned_actions) == 0:
			self.pre_planned_actions = path_to_fruit('m', game)
		return super().select_action(game)

def path_to_pill(player, game):
	'''Search for the shortest path to the nearest pill using BFGS'''
	if len(game.pills) == 0:
		return ['hold']
	visited = set()
	frontier = deque()
	frontier.append((game.players[player], deque()))
	possible_actions = list(gpac.GHOST_ACTIONS.items())
	while frontier:
		base_loc, base_actions = frontier.popleft()
		for action, shift in random.sample(possible_actions, len(possible_actions)):
			actions = base_actions.copy()
			actions.append(action)
			x, y = base_loc[0]+shift[0], base_loc[1]+shift[1]
			if 0 <= x < len(game.map) and 0 <= y < len(game.map[x]) and game.map[x][y]==0 and (x,y) not in visited:
				if (x,y) in game.pills:
					return actions
				else:
					frontier.append(((x,y), actions))
					visited.add((x,y))
	return ['hold'] # failsafe case

def path_to_fruit(player, game):
	'''Search for the shortest path to the fruit using BFGS'''	
	if game.fruit_location == None:
		return ['hold']
	visited = set()
	frontier = deque()
	frontier.append((game.players[player], deque()))
	possible_actions = list(gpac.GHOST_ACTIONS.items())
	while frontier:
		base_loc, base_actions = frontier.popleft()
		for action, shift in random.sample(possible_actions, len(possible_actions)):
			actions = base_actions.copy()
			actions.append(action)
			x, y = base_loc[0]+shift[0], base_loc[1]+shift[1]
			if 0 <= x < len(game.map) and 0 <= y < len(game.map[x]) and game.map[x][y]==0 and (x,y) not in visited:
				if (x,y) == game.fruit_location:
					return actions
				else:
					frontier.append(((x,y), actions))
					visited.add((x,y))
	return ['hold'] # failsafe case