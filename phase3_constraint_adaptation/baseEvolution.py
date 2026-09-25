class baseEvolutionPopulation():
    def __init__(self, individual_class, mu, num_children, mutation_rate,
                 parent_selection, survival_selection,
                 initialization_kwargs=dict(), parent_selection_kwargs=dict(),
                 recombination_kwargs = dict(), mutation_kwargs = dict(),
                 survival_selection_kwargs=dict(), **kwargs):
        self.mu = mu
        self.num_children = num_children
        self.mutation_rate = mutation_rate
        self.parent_selection = parent_selection
        self.survival_selection = survival_selection
        self.parent_selection_kwargs = parent_selection_kwargs
        self.recombination_kwargs = recombination_kwargs
        self.mutation_kwargs = mutation_kwargs
        self.survival_selection_kwargs = survival_selection_kwargs

        self.population = individual_class.initialization(self.mu, **initialization_kwargs)

    def generate_children(self):
        children = list()

        # TODO: Select parents
        # 準備兩倍於子代數量的父母 (因為 1 個小孩需要 2 個父母)
        parents = self.parent_selection(self.population, self.num_children * 2, **self.parent_selection_kwargs)
        
        # TODO: Recombine parents to generate children
        for i in range(self.num_children):
            # 依序從父母名單中兩兩抓取
            parent1 = parents[i * 2]
            parent2 = parents[i * 2 + 1]
            
            # 執行交配
            child = parent1.recombine(parent2, **self.recombination_kwargs)
            
            # TODO: Mutate children if appropriate
            # 根據突變機率 (mutation_rate) 決定這個小孩是否要突變
            import random
            if random.random() < self.mutation_rate:
                child = child.mutate(**self.mutation_kwargs)
                
            children.append(child)

        return children
        
        


    def survival(self):
        self.population = self.survival_selection(self.population, self.mu, **self.survival_selection_kwargs)
