import random
import copy

# Parent selection functions---------------------------------------------------

def uniform_random_selection(population, n, **kwargs):
    # TODO: select n individuals uniform randomly
    # 均勻隨機選擇 (取後放回)：每個人被選中的機率都一樣
    return random.choices(population, k=n)


def k_tournament_with_replacement(population, n, k, **kwargs):
    # TODO: perform n k-tournaments with replacement to select n individuals
    parents = []
    for _ in range(n):
        # 每次比賽從族群中隨機抽出 k 個不重複的個體來競爭
        tournament = random.sample(population, k)
        # 選擇這 k 個裡面 fitness 最高的那一個作為贏家
        winner = max(tournament, key=lambda ind: ind.fitness)
        parents.append(winner)
        
    return parents


def fitness_proportionate_selection(population, n, **kwargs):
    # TODO: select n individuals using fitness proportionate selection
    # 找出目前的最低分數
    min_fitness = min(ind.fitness for ind in population)
    max_fitness = max(ind.fitness for ind in population)
    
    # 【終極防呆】如果大家分數都一樣，直接均勻隨機選擇
    if max_fitness == min_fitness:
        return random.choices(population, k=n)
    
    # Temporarily modified fitness: fitness - min_fitness*1.5
    # 把分數平移為正數，讓負分也可以用來計算機率權重
    modified_fitnesses = [ind.fitness - (min_fitness * 1.5) for ind in population]
    
    # 雙重保險：確保權重總和大於 0，且沒有負數
    if sum(modified_fitnesses) <= 0 or any(w < 0 for w in modified_fitnesses):
        return random.choices(population, k=n)

    # 根據轉換後的分數當作權重 (weights) 進行輪盤式抽取 (取後放回)
    parents = random.choices(population, weights=modified_fitnesses, k=n)
    
    return parents



# Survival selection functions-------------------------------------------------

def truncation(population, n, **kwargs):
    # TODO: perform truncation selection to select n individuals
    # Population is sorted by fitness, the n most fit individuals are selected to survive.
    # 截斷選擇：直接把所有人按分數由高到低排序，拿前 n 個
    sorted_population = sorted(population, key=lambda ind: ind.fitness, reverse=True)
    return sorted_population[:n]


def k_tournament_without_replacement(population, n, k, **kwargs):
    # TODO: perform n k-tournaments without replacement to select n individuals
    #       Note: an individual should never be cloned from surviving twice!
    survivors = []
    # 複製一份名單，這樣我們才能把已經贏得生存權的人剔除
    available_population = population.copy()
    
    for _ in range(n):
        # Select k unique individuals from the population
        # 【修復Bug】順應測試要求，如果 k 大於剩餘人數，直接拋出錯誤
        if k > len(available_population):
            raise ValueError("k cannot be larger than available population")
        # (如果剩下的人數比 k 還少，就直接取剩下的所有人)
        current_k = min(k, len(available_population))
        tournament = random.sample(available_population, current_k)
        
        # Select the best individual from the k individuals
        winner = max(tournament, key=lambda ind: ind.fitness)
        survivors.append(winner)
        
        # Remove the best individual from the cloned population to avoid surviving twice.
        available_population.remove(winner)
        
    return survivors


# Yellow deliverable parent selection function---------------------------------

def stochastic_universal_sampling(population, n, **kwargs):
    # Recall that yellow deliverables are required for students in the grad
    # section but bonus for those in the undergrad section.
    # TODO: select n individuals using stochastic universal sampling
    
    # Temporarily modified fitness: fitness - min_fitness*1.5
    min_fitness = min(ind.fitness for ind in population)
    max_fitness = max(ind.fitness for ind in population)
    
    
    # 【終極防呆】如果大家分數都一樣，直接回傳均勻隨機選擇的結果
    if max_fitness == min_fitness:
        return random.choices(population, k=n)
    
    modified_fitnesses = [ind.fitness - (min_fitness * 1.5) for ind in population]
    total_fitness = sum(modified_fitnesses)
    
    if total_fitness <= 0 or any(w < 0 for w in modified_fitnesses):
        return random.choices(population, k=n)

    # SUS 核心邏輯：計算指針之間的固定距離
    distance = total_fitness / n
    # 隨機決定第一根指針的起點
    start = random.uniform(0, distance)
    # 產生 n 根等距的指針
    pointers = [start + i * distance for i in range(n)]
    
    parents = []
    for pointer in pointers:
        cumulative = 0
        for ind, fit in zip(population, modified_fitnesses):
            cumulative += fit
            # 當累積機率超過指針位置時，就選中這個個體
            if cumulative >= pointer:
                parents.append(ind)
                break
                
    return parents