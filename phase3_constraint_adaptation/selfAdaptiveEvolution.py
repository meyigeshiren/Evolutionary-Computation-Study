from baseEvolution import baseEvolutionPopulation
import random

class selfAdaptiveEvolutionPopulation(baseEvolutionPopulation):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 因為每個個體都有自己的突變率，所以把族群統一的突變率關掉 (設為 None)
        self.mutation_rate = None

    def generate_children(self):
        children = list()
        
        # 1. 挑選父母 (一次挑出需要的兩倍數量)
        parents = self.parent_selection(self.population, self.num_children * 2, **self.parent_selection_kwargs)
        
        for i in range(self.num_children):
            # 2. 兩兩一組抓出父母
            parent1 = parents[i * 2]
            parent2 = parents[i * 2 + 1]
            
            # 3. 執行交配
            child = parent1.recombine(parent2, **self.recombination_kwargs)

            # 4. 根據小孩【自己專屬】的 mutation rate 來進行突變
            child_mutation_kwargs = self.mutation_kwargs.copy()
            child_mutation_kwargs['mutation_rate'] = child.parameter
            child = child.mutate(**child_mutation_kwargs)

            children.append(child)

        return children