from baseEvolution import baseEvolutionPopulation
import random
import statistics

class adaptiveEvolutionPopulation(baseEvolutionPopulation):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mutation_rate = kwargs['mutation_rate']

    def generate_children(self):
        children = list()
        # The modified fitness of the population
        modified_fitness_list = []
        min_fitness = min([individual.fitness for individual in self.population])
        for individual in self.population:
            # Temporarily modified fitness: fitness - min_fitness*1.5
            modified_fitness = individual.fitness - min_fitness*1.5
            modified_fitness_list.append(modified_fitness)
        # Normalize the modified fitness of the population
        modified_fitness_max = max(modified_fitness_list)
        if modified_fitness_max == 0:
            normalized_fitness = [0.0 for _ in modified_fitness_list]
        else:
            normalized_fitness = [modified_fitness / modified_fitness_max for modified_fitness in modified_fitness_list]
        # Sort the Normalized modified fitness of the population
        normalized_fitness.sort()
        # Set the mutation rate to the median of the Normalized modified fitness
        self.mutation_rate = statistics.median(normalized_fitness)
        self.mutation_kwargs['mutation_rate'] = self.mutation_rate
        
          # 🌟 建議把「挑選父母」移到 for 迴圈的外面，一次挑好比較有效率！
        parents = self.parent_selection(self.population, self.num_children * 2, **self.parent_selection_kwargs)

        for i in range(self.num_children):
            # 1. 依序抓取父母
            parent1 = parents[i * 2]
            parent2 = parents[i * 2 + 1]

            # 2. 執行交配
            child = parent1.recombine(parent2, **self.recombination_kwargs)

            # 3. 執行突變 (⚠️ 關鍵：使用你剛剛算好的全域 self.mutation_rate)
            # 為了避免 kwargs 重複傳遞參數報錯，我們先確保 mutation_rate 不在 kwargs 裡
            safe_mutation_kwargs = self.mutation_kwargs.copy()
            if 'mutation_rate' in safe_mutation_kwargs:
                safe_mutation_kwargs.pop('mutation_rate')
                
            child = child.mutate(mutation_rate=self.mutation_rate, **safe_mutation_kwargs)

            # 將新生兒加入名單
            children.append(child)

        return children
