"""
Phase 1: Baseline Formulation & Inferential Statistical Validation
Author: Cheng-Bo Li
Description:
    Conducts 30 independent runs of Random Search baseline (2,000 evaluations each)
    using multi-processing, plots convergence progression, and executes F-test / t-test.
"""

import os
import random
import concurrent.futures
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from binaryGenotype import binaryGenotype
from fitness import repair_and_test_map

def run_single_experiment(run_id, solutions_per_run, length, height, width):
    """執行單次蒙地卡羅隨機搜尋實驗"""
    # 確保多程序平行運算時隨機種子相互獨立
    random.seed(os.urandom(4) + str(run_id).encode('utf-8'))
    
    run_best_fitness = float('-inf')
    run_best_log = None
    stairstep_trajectory = []

    for _ in range(solutions_per_run):
        sol = binaryGenotype()
        sol.randomInitialization(length=length)
        sol.fitness, sol.log = repair_and_test_map(sol.gene, height, width)

        if sol.fitness > run_best_fitness:
            run_best_fitness = sol.fitness
            run_best_log = sol.log

        stairstep_trajectory.append(run_best_fitness)

    return run_id, run_best_fitness, run_best_log, stairstep_trajectory

def run_statistical_analysis(my_data, benchmark_path):
    """執行變異數同質性 F 檢定與平均數 t 檢定"""
    if not os.path.exists(benchmark_path):
        print(f"[Warning] Benchmark data '{benchmark_path}' not found. Skipping inferential statistics.")
        return

    with open(benchmark_path, 'r') as f:
        benchmark_data = [float(line.strip()) for line in f.readlines() if line.strip()]

    # 1. 雙樣本 F 檢定（變異數同質性）
    var_my = np.var(my_data, ddof=1)
    var_bm = np.var(benchmark_data, ddof=1)
    f_stat = var_my / var_bm
    df1, df2 = len(my_data) - 1, len(benchmark_data) - 1
    p_val_f = 2 * min(stats.f.cdf(f_stat, df1, df2), 1 - stats.f.cdf(f_stat, df1, df2))

    equal_var = p_val_f > 0.05

    # 2. 雙樣本 t 檢定（平均數差異）
    t_stat, p_val_t = stats.ttest_ind(my_data, benchmark_data, equal_var=equal_var)

    report = (
        f"=== 推論統計檢定分析成果 ===\n"
        f"隨機搜尋基準 (Random Search)   - Mean: {np.mean(my_data):.4f}, Variance: {var_my:.4f}\n"
        f"對照啟發式原型 (Benchmark EA)   - Mean: {np.mean(benchmark_data):.4f}, Variance: {var_bm:.4f}\n"
        f"--------------------------------------------------\n"
        f"F-Test (變異數同質性): F = {f_stat:.4f}, p = {p_val_f:.4f} (等變異數假設: {equal_var})\n"
        f"T-Test (平均數差異性): t = {t_stat:.4f}, p = {p_val_t:.4e}\n"
    )
    print(report)
    with open('data/statistical_results_phase1.txt', 'w', encoding='utf-8') as f:
        f.write(report)

def main():
    height, width = 20, 35
    length = height * width
    number_of_runs = 30
    evaluations_per_run = 2000
    os.makedirs('data', exist_ok=True)
    os.makedirs('worldFiles', exist_ok=True)

    print(f"[*] 啟動階段一實驗: 執行 {number_of_runs} 組獨立運行，每組預算 {evaluations_per_run} 次評估...")

    highest_fitness_per_run = [0.0] * number_of_runs
    best_stairstep_data = []
    global_best_fitness = float('-inf')
    global_best_log = None

    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(run_single_experiment, i, evaluations_per_run, length, height, width)
            for i in range(number_of_runs)
        ]

        for future in concurrent.futures.as_completed(futures):
            run_id, best_fit, best_log, trajectory = future.result()
            highest_fitness_per_run[run_id] = best_fit
            print(f"  [+] Run {run_id + 1:02d}/{number_of_runs} 完成 - 該組最佳適應度: {best_fit:.2f}")

            if best_fit > global_best_fitness:
                global_best_fitness = best_fit
                global_best_log = best_log
                best_stairstep_data = list(trajectory)

    # 儲存最佳運行的世界紀錄檔
    if global_best_log:
        with open('worldFiles/bestRandom.txt', 'w') as f:
            for line in global_best_log:
                f.write(f"{line}\n")

    # 儲存每組 30 次運行的數據
    with open('data/randomSearchResults.txt', 'w') as f:
        for val in highest_fitness_per_run:
            f.write(f"{val}\n")

    # 繪製最佳運行的階梯收斂圖 (Stairstep Plot)
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(best_stairstep_data)), best_stairstep_data, drawstyle='steps-post', color='navy', linewidth=2)
    plt.title('Fitness Progression of the Best Random Search Run')
    plt.xlabel('Number of Evaluations')
    plt.ylabel('Best Fitness Found So Far')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig('data/bestRandomPlot.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 執行推論統計檢驗
    run_statistical_analysis(highest_fitness_per_run, 'data/mysteryAlgorithmResults.txt')
    print(f"[✓] 階段一實驗完成！全域最佳適應度: {global_best_fitness:.2f}，圖表已存至 data/bestRandomPlot.png")

if __name__ == '__main__':
    main()