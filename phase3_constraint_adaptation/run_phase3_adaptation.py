"""
Phase 3: Constraint Satisfaction, Parameter Self-Adaptation & Specification Gaming
Author: Cheng-Bo Li
Description:
    Implements 4 EA variants:
    1. Fixed penalty constraint satisfaction (Green 1c0)
    2. Self-adaptive mutation rate (Green 1c1)
    3. Self-adaptive penalty coefficient (Yellow 1c - Specification Gaming)
    4. Global heuristic adaptive mutation rate (Red 1c)
    Generates convergence plots, parameter boxplots, and cross-variant F/T tests.
"""

import os
import ast
import concurrent.futures
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from snakeeyes import readConfig
from baseEvolution import baseEvolutionPopulation
from selfAdaptiveEvolution import selfAdaptiveEvolutionPopulation
from adaptiveEA import adaptiveEvolutionPopulation
from fitness import repair_and_test_map

def run_single_phase3(run_id, mode, config_path, evals_budget=2000):
    """執行單組第三階段演化運行"""
    cfg = readConfig(config_path, globalVars=globals(), localVars=locals())
    fit_kwargs = cfg['fitness_kwargs'].copy()
    fixed_penalty = fit_kwargs.pop('penalty_coefficient', 0.2)

    # 根據模型模式選取演化引擎架構
    if mode == 'green1c0':
        ea = baseEvolutionPopulation(**cfg['EA_configs'], **cfg)
    elif mode in ['green1c1', 'yellow1c']:
        ea = selfAdaptiveEvolutionPopulation(**cfg['EA_configs'], **cfg)
    elif mode == 'red1c':
        ea = adaptiveEvolutionPopulation(**cfg['EA_configs'], **cfg)

    # 評估初期族群
    for ind in ea.population:
        raw_score, log, repairs = repair_and_test_map(ind.gene, **fit_kwargs)
        ind.rawFitness = raw_score
        ind.log = log

        if mode == 'green1c0':
            ind.fitness = raw_score - (fixed_penalty * repairs)
        elif mode == 'green1c1':
            ind.fitness = raw_score
        elif mode == 'yellow1c':
            ind.fitness = raw_score - (ind.parameter * repairs)
        elif mode == 'red1c':
            ind.fitness = raw_score

    ea.evaluations = len(ea.population)
    mean_history = [np.mean([ind.fitness for ind in ea.population])]
    best_history = [np.max([ind.fitness for ind in ea.population])]
    param_history = []

    if mode in ['green1c1', 'yellow1c']:
        param_history.append([ind.parameter for ind in ea.population])
    elif mode == 'red1c':
        param_history.append([ea.mutation_rate] * len(ea.population))

    best_raw_fitness = max([ind.rawFitness for ind in ea.population])

    # 進入演化代數疊代
    while ea.evaluations < evals_budget:
        children = ea.generate_children()
        for child in children:
            raw_score, log, repairs = repair_and_test_map(child.gene, **fit_kwargs)
            child.rawFitness = raw_score
            child.log = log

            if mode == 'green1c0':
                child.fitness = raw_score - (fixed_penalty * repairs)
            elif mode == 'green1c1':
                child.fitness = raw_score
            elif mode == 'yellow1c':
                child.fitness = raw_score - (child.parameter * repairs)
            elif mode == 'red1c':
                child.fitness = raw_score

        ea.evaluations += len(children)
        ea.population += children
        ea.survival()

        mean_history.append(np.mean([ind.fitness for ind in ea.population]))
        best_history.append(np.max([ind.fitness for ind in ea.population]))

        if mode in ['green1c1', 'yellow1c']:
            param_history.append([ind.parameter for ind in ea.population])
        elif mode == 'red1c':
            param_history.append([ea.mutation_rate] * len(ea.population))

        cur_best_raw = max([ind.rawFitness for ind in ea.population])
        if cur_best_raw > best_raw_fitness:
            best_raw_fitness = cur_best_raw

    return run_id, mean_history, best_history, param_history, best_raw_fitness

def run_experiment_suite(mode, exp_name, config_path, number_runs=30):
    """驅動單一變體之 30 組平行實驗並儲存圖表統計"""
    print(f"[*] 執行階段三模組: {exp_name}...")
    mean_runs, best_runs, param_runs = [], [], []
    raw_best_per_run = [0.0] * number_runs

    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(run_single_phase3, i, mode, config_path) for i in range(number_runs)]
        for future in concurrent.futures.as_completed(futures):
            r_id, m_hist, b_hist, p_hist, best_raw = future.result()
            mean_runs.append(m_hist)
            best_runs.append(b_hist)
            if p_hist:
                param_runs.append(p_hist)
            raw_best_per_run[r_id] = best_raw

    # 儲存供統計檢定使用之 Raw Best Fitness
    with open(f'data/{exp_name}_bestfitness.txt', 'w') as f:
        for val in raw_best_per_run:
            f.write(f"{val}\n")

    # 1. 繪製適應度綜合收斂圖 (Boxplot + 平均最佳軌跡)
    n_gens = len(mean_runs[0])
    plt.figure(figsize=(10, 6))
    bp = plt.boxplot(mean_runs, positions=range(n_gens), patch_artist=True,
                     boxprops=dict(facecolor='skyblue', alpha=0.6))
    best_mean = np.mean(best_runs, axis=0)
    best_std = np.std(best_runs, axis=0)
    plt.plot(range(n_gens), best_mean, color='green', marker='o', linewidth=2, label='Mean of Best Fitness')
    plt.fill_between(range(n_gens), best_mean - best_std, best_mean + best_std, color='green', alpha=0.15)
    plt.title(f'EA Search: {exp_name} (Fitness Analysis)')
    plt.xlabel('Generations')
    plt.ylabel('Fitness')
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.legend([bp["boxes"][0], plt.gca().lines[0]], ['Average Fitness (Boxplot)', 'Mean of Best Fitness'], loc='lower right')
    plt.savefig(f'data/{exp_name}_fitness_combined.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. 繪製超參數動態變化 Boxplot (若存在)
    if param_runs:
        # 轉換陣列形狀為 (世代數, 30組Runs之個體參數集合)
        gens_param = []
        for g in range(n_gens):
            gen_vals = []
            for r in range(number_runs):
                gen_vals.extend(param_runs[r][g])
            gens_param.append(gen_vals)

        plt.figure(figsize=(10, 5))
        plt.boxplot(gens_param, positions=range(n_gens), patch_artist=True,
                    boxprops=dict(facecolor='#8dd3c7' if mode != 'yellow1c' else '#ffffb3', alpha=0.7))
        y_label = 'Penalty Coefficient' if mode == 'yellow1c' else 'Mutation Rate'
        plt.title(f'EA Search: {exp_name} - Dynamic {y_label}')
        plt.xlabel('Generations')
        plt.ylabel(y_label)
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        img_name = 'penalty_boxplot' if mode == 'yellow1c' else 'mutation_boxplot'
        plt.savefig(f'data/{exp_name}_{img_name}.png', dpi=300, bbox_inches='tight')
        plt.close()

def generate_formal_comparison(name_A, file_A, name_B, file_B, output_path):
    """執行 F 檢定與 t 檢定並產出嚴謹之對照分析表"""
    with open(file_A, 'r') as f:
        data_A = [float(x.strip()) for x in f.readlines() if x.strip()]
    with open(file_B, 'r') as f:
        data_B = [float(x.strip()) for x in f.readlines() if x.strip()]

    mean_A, mean_B = np.mean(data_A), np.mean(data_B)
    var_A, var_B = np.var(data_A, ddof=1), np.var(data_B, ddof=1)
    n_A, n_B = len(data_A), len(data_B)

    # 雙樣本 F 檢定
    f_stat = var_A / var_B if var_A > var_B else var_B / var_A
    df_A, df_B = n_A - 1, n_B - 1
    p_one_tail_f = 1 - stats.f.cdf(f_stat, df_A, df_B)
    p_two_tail_f = 2 * p_one_tail_f
    is_equal_var = p_two_tail_f >= 0.05

    # 雙樣本 t 檢定
    t_res = stats.ttest_ind(data_A, data_B, equal_var=is_equal_var)
    df_t = n_A + n_B - 2 if is_equal_var else (
        ((var_A/n_A + var_B/n_B)**2) / ((var_A/n_A)**2 / (n_A-1) + (var_B/n_B)**2 / (n_B-1))
    )

    report = (
        f"=== 推論統計檢驗報表: {name_A} vs. {name_B} ===\n"
        f"{name_A:<25} Mean: {mean_A:<10.4f} Variance: {var_A:.4f}\n"
        f"{name_B:<25} Mean: {mean_B:<10.4f} Variance: {var_B:.4f}\n"
        f"F-Test: F = {f_stat:.4f}, p = {p_two_tail_f:.4f} ({'變異數同質' if is_equal_var else '變異數異質'})\n"
        f"T-Test: t = {t_res.statistic:.4f}, df = {df_t:.1f}, p = {t_res.pvalue:.4e}\n"
        f"結論: {'具高度顯著差異' if t_res.pvalue < 0.05 else '無統計顯著差異'}\n"
        f"--------------------------------------------------\n"
    )
    print(report)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)

def main():
    os.makedirs('data', exist_ok=True)

    # 1. 依序執行四組變體實驗
    run_experiment_suite('green1c0', 'green1c0', './configs/green1c0_config.txt')
    run_experiment_suite('green1c1', 'green1c1', './configs/green1c1_config.txt')
    run_experiment_suite('yellow1c', 'yellow1c', './configs/yellow1c_config.txt')
    run_experiment_suite('red1c', 'red1c', './configs/red1c_config.txt')

    # 2. 執行研究報告中的五項核心假設檢定
    print("\n[*] 執行五組推論統計檢驗...")
    generate_formal_comparison('Green 1c0 (Fixed Penalty)', 'data/green1c0_bestfitness.txt',
                               'Green 1b (Baseline)', 'data/Green1b_bestfitness.txt',
                               'data/report_green1c0_vs_green1b.txt')

    generate_formal_comparison('Green 1c0 (Fixed Penalty)', 'data/green1c0_bestfitness.txt',
                               'Green 1c1 (Self-Adaptive)', 'data/green1c1_bestfitness.txt',
                               'data/report_green1c0_vs_green1c1.txt')

    generate_formal_comparison('Green 1c1 (Self-Adaptive)', 'data/green1c1_bestfitness.txt',
                               'Green 1b (Expert Knowledge)', 'data/Green1b_bestfitness.txt',
                               'data/report_green1c1_vs_green1b.txt')

    generate_formal_comparison('Red 1c (Global Heuristic)', 'data/red1c_bestfitness.txt',
                               'Green 1c1 (Individual Genetic)', 'data/green1c1_bestfitness.txt',
                               'data/report_red1c_vs_green1c1.txt')

    generate_formal_comparison('Yellow 1c (Adaptive Penalty)', 'data/yellow1c_bestfitness.txt',
                               'Green 1c0 (Fixed Penalty)', 'data/green1c0_bestfitness.txt',
                               'data/report_yellow1c_vs_green1c0.txt')

    print("[✓] 階段三實驗、圖表與統計分析全數產出完成！")

if __name__ == '__main__':
    main()