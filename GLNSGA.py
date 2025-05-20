import numpy as np
from copy import deepcopy
from sklearn.mixture  import GaussianMixture
import plotly.graph_objs  as go
from sklearn.preprocessing  import StandardScaler
from zjs_parameterconfiguration import M,I,Cap,C,P,R,H,D,POP_SIZE,GENS,PC,PM,GMGEN_RATIO,BETA,DRATIO
#
np.random.seed(42)

PSUM = np.sum(P)
print("总生产需求 (PSUM):", PSUM)


# 初始化种群
def init_population():
    population = []
    for _ in range(POP_SIZE):
        X = np.zeros((I,  M))
        for i in range(I):
            remaining_capacity = Cap * (1 - R[i] / H[i])
            for j in range(M):
                X[i, j] = np.random.uniform(0,  remaining_capacity[j])
        X = repair_solution(X)
        population.append(X)
    return population

# 计算目标函数值
def calculate_objectives(individual):
    f1 = np.sum(C * individual)
    U = np.zeros(I)
    carry_over = 0.0
    for i in range(I):
        # 调整后的需求 = 本月需求 - 前月累积量（若前月超额，则减少本月需求）
        adjusted_demand = P[i] - carry_over
        sum_Xi = np.sum(individual[i, :])  # 当月总产量
        # 未完成量 = 调整后需求 - 产量（若产量不足则为正）
        U_i = max(adjusted_demand - sum_Xi, 0)
        # 更新累积量（若超额完成，carry_over为正；若不足则为负，传递给下月）
        carry_over = sum_Xi - adjusted_demand
        U[i] = U_i  # 记录当月未完成量
    f2 = np.sum(D * U)  # 惩罚项计算
    load = np.sum(individual, axis=0)
    f3 = np.sum((load - np.mean(load)) ** 2)
    return np.array([f1, f2, f3])

# 非支配排序
def non_dominated_sort(population, objectives):
    fronts = [[]]
    domination_count = [0] * len(population)
    dominated_solutions = [[] for _ in range(len(population))]
    rank = [0] * len(population)

    for i in range(len(population)):
        for j in range(len(population)):
            if dominates(objectives[i], objectives[j]):
                dominated_solutions[i].append(j)
            elif dominates(objectives[j], objectives[i]):
                domination_count[i] += 1
        if domination_count[i] == 0:
            rank[i] = 0
            fronts[0].append(i)

    i = 0
    while len(fronts[i]) > 0:
        next_front = []
        for idx in fronts[i]:
            for dominated_idx in dominated_solutions[idx]:
                domination_count[dominated_idx] -= 1
                if domination_count[dominated_idx] == 0:
                    rank[dominated_idx] = i + 1
                    next_front.append(dominated_idx)
        i += 1
        fronts.append(next_front)
    if not fronts[-1]:
        fronts.pop()
    return fronts, rank

def dominates(obj1, obj2):
    return np.all(obj1 <= obj2) and np.any(obj1  < obj2)

# 拥挤距离计算
def crowding_distance(front, objectives):
    distance = np.zeros(len(front))
    num_objectives = objectives.shape[1]
    for m in range(num_objectives):
        obj_m = objectives[front, m]
        sorted_indices = np.argsort(obj_m)
        distance[sorted_indices[0]] = distance[sorted_indices[-1]] = float('inf')
        max_obj = obj_m[sorted_indices[-1]]
        min_obj = obj_m[sorted_indices[0]]
        if max_obj == min_obj:
            continue
        for k in range(1, len(front) - 1):
            distance[sorted_indices[k]] += (obj_m[sorted_indices[k + 1]] - obj_m[sorted_indices[k - 1]]) / (max_obj - min_obj)
    return distance

# 修复解
def repair_solution(z_matrix):
    """
    将一个任意的 I×M 矩阵 z_matrix 修复为满足
      (1) 0 <= x_ij <= capacity_ij
      (2) sum(x_ij) == PSUM
    的最优投影解 x_matrix（最小化 ||x - z||^2）。
    """
    # 1) 计算每个位置的容量上界
    cap_matrix = np.zeros_like(z_matrix)
    for i in range(I):
        # 每台设备 j 在月份 i 的最大产能
        cap_matrix[i, :] = Cap * (1 - R[i] / H[i])

    # 2) 展平矩阵，准备做带上界的单纯形投影
    z = z_matrix.flatten()
    u = cap_matrix.flatten()
    s = float(PSUM)

    # 3) 特殊情况快速返回
    if s <= 0:
        return np.zeros_like(z_matrix)
    if np.sum(u) <= s:
        # 若所有上界之和都 <= 总需求，则直接取上界
        return u.reshape(I, M)

    # 4) 二分法求投影参数 θ，使得
    #       x_k = min(max(z_k - θ, 0), u_k)
    #     且 sum(x_k) == s
    def project_bounded_simplex(z, u, s, tol=1e-8, max_iter=1000):
        # 初始 θ 范围：当 θ = min(z-u) 时 sum = sum(u)；当 θ = max(z) 时 sum = 0
        theta_lo = np.min(z - u)
        theta_hi = np.max(z)
        for _ in range(max_iter):
            theta = 0.5 * (theta_lo + theta_hi)
            x = np.minimum(np.maximum(z - theta, 0.0), u)
            total = x.sum()
            if abs(total - s) <= tol:
                break
            if total > s:
                # 太“宽松”，需要增大 θ
                theta_lo = theta
            else:
                # 太“紧凑”，需要减小 θ
                theta_hi = theta
        # 最后一次计算
        return np.minimum(np.maximum(z - theta, 0.0), u)

    x = project_bounded_simplex(z, u, s)
    # 5) 恢复原始形状并返回
    return x.reshape(I, M)

# 传统交叉操作
def crossover(parent1, parent2):
    if np.random.rand()  < PC:
        point = np.random.randint(1,  I * M - 1)
        child1 = np.concatenate([parent1.flatten()[:point],  parent2.flatten()[point:]]).reshape(I,  M)
        child2 = np.concatenate([parent2.flatten()[:point],  parent1.flatten()[point:]]).reshape(I,  M)
        child1 = repair_solution(child1)
        child2 = repair_solution(child2)
        return child1, child2
    return parent1, parent2

# 传统变异操作
def mutation(individual):
    if np.random.rand() < PM:
        i = np.random.randint(I)
        j = np.random.randint(M)
        individual[i, j] = np.random.uniform(0,  Cap[j] * (1 - R[i, j] / H[i]))
        individual = repair_solution(individual)
    return individual

# 二元锦标赛选择
def binary_tournament(population, objectives, rank, distances):
    idx1, idx2 = np.random.choice(len(population),  size=2, replace=False)
    if rank[idx1] < rank[idx2]:
        return population[idx1], objectives[idx1]
    elif rank[idx2] < rank[idx1]:
        return population[idx2], objectives[idx2]
    else:
        if distances[idx1] > distances[idx2]:
            return population[idx1], objectives[idx1]
        else:
            return population[idx2], objectives[idx2]

# 混合后代生成：结合GM采样和传统遗传操作
def generate_offspring(all_points, gen, population, fronts, objectives, rank):
    offspring = []
    offspring_objectives = []
    # 准备数据并标准化
    data = np.array([point.flatten() for point in all_points])
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    bic_values = []
    for n in range(1, 6):
        gmbic = GaussianMixture(n_components=n, covariance_type='tied', random_state=42).fit(data_scaled)
        bic_values.append(gmbic.bic(data_scaled))
    gmm_components = np.argmin(bic_values) + 1
    gm_model = GaussianMixture(n_components=gmm_components, covariance_type='tied', random_state=42)
    gm_model.fit(data_scaled)

    while len(offspring) < POP_SIZE:
        if np.random.rand() < GMGEN_RATIO:
            # 生成样本并反标准化
            sample = gm_model.sample()[0]
            child = scaler.inverse_transform(sample).reshape(I, M)  # 反标准化到原始范围
            child = repair_solution(child)
            # 计算GM采样后代的目标值
            child_obj = calculate_objectives(child)
            offspring.append(child)
            offspring_objectives.append(child_obj)
        else:
            if len(offspring) < POP_SIZE:
                distances = np.zeros(len(population))
                for front in fronts:
                    front_distances = crowding_distance(front, objectives)
                    for idx, dist in zip(front, front_distances):
                        distances[idx] = dist
                # 使用传统交叉和变异生成后代
                parent1, _ = binary_tournament(population, objectives, rank, distances)
                parent2, _ = binary_tournament(population, objectives, rank, distances)
                child1, child2 = crossover(deepcopy(parent1), deepcopy(parent2))
                child1 = mutation(child1)
                child2 = mutation(child2)
                offspring.append(child1)
                offspring_objectives.append(calculate_objectives(child1))

                offspring.append(child2)
                offspring_objectives.append(calculate_objectives(child2))

    return offspring, offspring_objectives

# 选择操作
def selection(population, objectives, fronts):
    new_population = []
    new_objectives = []
    for front in fronts:
        if len(new_population) + len(front) <= POP_SIZE:
            new_population.extend([population[i] for i in front])
            new_objectives.extend([objectives[i] for i in front])
        else:
            front_distances = crowding_distance(front, objectives)
            sorted_indices = np.argsort(-front_distances)
            remaining_slots = POP_SIZE - len(new_population)
            selected_indices = sorted_indices[:remaining_slots]
            new_population.extend([population[front[i]]  for i in selected_indices])
            new_objectives.extend([objectives[front[i]]  for i in selected_indices])
            break
    return new_population, np.array(new_objectives)

def hybriddataaugmented(population,fronts):
    # 获取Pareto前沿解
    pareto_solutions = [population[i] for i in fronts[0]]

    # 生成扰动点
    perturbed_points = []
    selected_indices = np.random.choice(len(pareto_solutions), int(len(pareto_solutions) * DRATIO), replace=False)
    for idx in selected_indices:
        x = pareto_solutions[idx]
        idx1, idx2 = np.random.choice(len(population), 2, replace=False)
        d = population[idx1] - population[idx2]
        x_prime = x + BETA * d
        x_prime = repair_solution(x_prime)
        perturbed_points.append(x_prime)

    # 合并Pareto解和扰动点用于GM训练
    all_points = pareto_solutions + perturbed_points
    return all_points

# 主NSGA-II函数
def gmnsga(init_pop):
    history = []
    population = deepcopy(init_pop)
    objectives = np.array([calculate_objectives(ind) for ind in population])
    genindex = 1
    for gen in range(GENS):
        fronts, rank = non_dominated_sort(population, objectives)
        print(genindex)
        genindex = genindex + 1

        all_points = hybriddataaugmented(population, fronts)

        # 使用混合策略生成后代
        offspring, offspring_objectives = generate_offspring(all_points, gen, population, fronts, objectives, rank)

        # 合并父代和子代种群
        combined_population = population + offspring
        combined_objectives = np.vstack((objectives,  offspring_objectives))

        # 对合并后的种群进行非支配排序
        fronts, rank = non_dominated_sort(combined_population, combined_objectives)


        # 选择下一代种群
        population, objectives = selection(combined_population, combined_objectives, fronts)

        # 记录历史信息
        fronts_his, _ = non_dominated_sort(population, objectives)
        pareto_obj = objectives[fronts_his[0]]
        history.append({
            'generation': gen,
            'pareto': pareto_obj,
            'all_objectives': objectives.copy()
        })

    # 返回最终Pareto前沿
    fronts, _ = non_dominated_sort(population, objectives)
    pareto_front = fronts[0]
    pareto_solutions = [population[i] for i in pareto_front]
    pareto_objectives = objectives[pareto_front]
    return population, objectives, history
