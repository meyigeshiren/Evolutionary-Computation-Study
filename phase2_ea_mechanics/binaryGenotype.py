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

    def recombine(self, mate, method, **kwargs):
        child = binaryGenotype()
        child.gene = self.gene.copy()
        # TODO: Recombine genes of self with mate and assign to child's gene member variable
        assert method.casefold() in {'uniform', '1-point crossover', 'multi-dimensional'}
        
        if method.casefold() == 'uniform':
            # perform uniform recombination
            # 均勻交配：針對每一個基因位元，有 50% 的機率保留 self 的基因，50% 替換成 mate 的基因
            for i in range(len(child.gene)):
                if random.random() < 0.5:
                    child.gene[i] = mate.gene[i]
            
        elif method.casefold() == '1-point crossover':
            # perform 1-point crossover
            # 單點交配：隨機切一刀（切點介於 1 到 基因長度-1 之間）
            # 切點前面的基因保留 self 的，切點後面的基因換成 mate 的
            crossover_point = random.randint(1, len(child.gene) - 1)
            child.gene[crossover_point:] = mate.gene[crossover_point:]
            
        elif method.casefold() == 'multi-dimensional':
            # this is a red deliverable (i.e., bonus for anyone)
            height, width = kwargs['height'], kwargs['width']
            # transform the linear gene of both parents to a 2-dimensional representation.
            # Recombine 2D parent genes into 2D child gene using the method of your choice.
            assert len(child.gene) == height*width, f'ERROR: EXPECTED GENOTYPE OF LENGTH {height*width} BUT GOT {len(child.gene)}'
            assert len(mate.gene) == height*width, f'ERROR: EXPECTED GENOTYPE OF LENGTH {height*width} BUT GOT {len(mate.gene)}'
            
            # 多維度交配 (Bonus)：這裡我們實作「2D 區塊交配 (Block Crossover)」
            # 隨機在 2D 網格中圈出一個矩形區塊，把這個區塊內的基因換成 mate 的
            x1, x2 = sorted([random.randint(0, width), random.randint(0, width)])
            y1, y2 = sorted([random.randint(0, height), random.randint(0, height)])
            
            for y in range(y1, y2):
                for x in range(x1, x2):
                    # 將 2D 座標 (x, y) 轉換回 1D 陣列的索引 (index)
                    idx = y * width + x
                    child.gene[idx] = mate.gene[idx]
            
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