"""
Phase 2: Evolutionary Algorithm Architecture, Selection Dynamics & Spatial Crossover Exploration
Author: Cheng-Bo Li
Description:
    Conducts comparative experiments across:
    1. Standard (mu + lambda) EA (Green 1b)
    2. Stochastic Universal Sampling Selection (Yellow 1c in Phase 2)
    3. 2D Block Recombination Operator (Red 1b)
"""

import os
import concurrent.futures
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from snakeeyes import readConfig
from baseEvolution import baseEvolutionPopulation
from fitness import repair_and_test_map

def run_single_ea(run_id, config_path, evaluations_budget=2000):
    """執行單組獨立的 EA 演化歷程"""
    config_dict = readConfig(config_path, globalVars=globals(), localVars=locals())
    ea = baseEvolutionPopulation(**config_dict['EA_configs'], **config_dict)

    # 評估初期族群 (Generation 0)
    for ind in ea.population:
        ind.fitness, ind.log = repair_and_test_map(ind.gene, **config_dict['fitness_kwargs'])
    ea.evaluations = len(ea.population)

    mean_history = [np.mean([ind.fitness for ind in ea.population])]
    best_history = [np.max([ind.fitness for ind in ea.population])]

    # 世代演化迴圈
    while ea.evaluations < evaluations_budget:
        children = ea.generate_children()
        for child in children:
            child.fitness, child.log = repair_and_test_map(child.gene, **config_dict['fitness_kwargs'])
        
        ea.evaluations += len(children)
        ea.population += children
        ea.survival()

        mean_history.append(np.mean([ind.fitness for ind in ea.population]))
        best_history.append(np.max([ind.fitness for ind in ea.population]))

    best_ind = max(ea.population, key=lambda ind: ind.fitness)
    return run_id, mean_history, best_history, best_ind.fitness, best_ind.log

def execute_experiment(experiment_name, config_path, number_runs=30):
    """執行單項演化實驗之 30 次平行運算"""
    print(f"[*] 啟動實驗: {experiment_name} (配置檔: {config_path})...")
    ea_mean_history = []
    ea_best_history = []
    best_per_run = [0.0] * number_runs

    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(run_single_ea, i, config_path) for i in range(number_runs)]
        for future in concurrent.futures.as_completed(futures):
            r_id, mean_hist, best_hist, best_fit, _ = future.result()
            ea_mean_history.append(mean_hist)
            ea_best_history.append(best_hist)
            best_per_run[r_id] = best_fit
            print(f"  [+] {experiment_name} Run {r_id + 1:02d}/{number_runs} 完成 - 最佳解: {best_fit:.2f}")

    # 輸出原始分數
    with open(f'data/{experiment_name}_bestfitness.txt', 'w') as f:
        for score in best_per_run:
            f.write(f"{score}\n")

    avg_mean = np.mean(ea_mean_history, axis=0)
    avg_best = np.mean(ea_best_history, axis=0)

    # 繪製學習曲線
    evals_axis = [200 + i * 100 for i in range(len(avg_mean))]
    plt.figure(figsize=(10, 6))
    plt.plot(evals_axis, avg_mean, label='Average Mean Fitness', linestyle='--', color='blue')
    plt.plot(evals_axis, avg_best, label='Average Best Fitness', color='red', linewidth=2)
    plt.title(f'EA Performance over 30 Runs ({experiment_name})')
    plt.xlabel('Number of Fitness Evaluations')
    plt.ylabel('Fitness Score')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(f'data/EA_Learning_Curve_{experiment_name}.jpg', dpi=300, bbox_inches='tight')
    plt.close()

    return best_per_run

def compare_results(name_A, data_A, name_B, data_B, output_path):
    """執行 Welch's t-test 並輸出報表"""
    t_stat, p_val = stats.ttest_ind(data_A, data_B, equal_var=False)
    report = (
        f"=== 假設檢定分析: {name_A} vs. {name_B} ===\n"
        f"{name_A} - Mean: {np.mean(data_A):.4f}, Std: {np.std(data_A):.4f}\n"
        f"{name_B} - Mean: {np.mean(data_B):.4f}, Std: {np.std(data_B):.4f}\n"
        f"Welch's t-Statistic: {t_stat:.4f}, p-Value: {p_val:.4e}\n"
        f"顯著性判定 (alpha=0.05): {'具統計顯著差異' if p_val < 0.05 else '未達統計顯著差異'}\n"
    )
    print(report)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)

def main():
    os.makedirs('data', exist_ok=True)

    # 1. 執行基準 EA 實驗 (Green 1b)
    green_best = execute_experiment('Green1b', './configs/green1b_config.txt')

    # 載入隨機搜尋數據進行對照
    if os.path.exists('data/randomSearchResults.txt'):
        with open('data/randomSearchResults.txt', 'r') as f:
            rs_data = [float(line.strip()) for line in f.readlines() if line.strip()]
        compare_results('Green 1b (EA)', green_best, 'Random Search Baseline', rs_data, 'data/Statistical_Analysis_Green1b.txt')

    # 2. 執行選擇機制消融實驗 (Yellow 1c - SUS)
    yellow_best = execute_experiment('Yellow1c', './configs/yellow1c_config.txt')
    compare_results('Yellow 1c (SUS)', yellow_best, 'Green 1b (Tournament)', green_best, 'data/Statistical_Analysis_Yellow1c.txt')

    # 3. 執行 2D 空間交叉算子實驗 (Red 1b - 2D Crossover)
    red_best = execute_experiment('Red1b', './configs/red1b_config.txt')
    compare_results('Red 1b (2D Crossover)', red_best, 'Green 1b (1-point Crossover)', green_best, 'data/Statistical_Analysis_Red1b.txt')

    print("[✓] 階段二全部實驗與統計檢定順利完成！")

if __name__ == '__main__':
    main()