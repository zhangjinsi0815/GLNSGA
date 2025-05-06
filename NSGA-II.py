import numpy as np
import plotly.graph_objs as go
from copy import deepcopy
from zjs_parameterconfiguration import M,I,Cap,C,P,R,H,D,POP_SIZE,GENS,PC,PM,GMGEN_RATIO,BETA,DRATIO

np.random.seed(42)


PSUM = np.sum(P)
print("Total Production Demand (PSUM):", PSUM)

# 初始化种群
def init_population():
    population = []
    for _ in range(POP_SIZE):
        X = np.zeros((I, M))
        for i in range(I):
            remaining_capacity = Cap * (1 - R[i] / H[i])
            for j in range(M):
                X[i, j] = np.random.uniform(0, remaining_capacity[j])
        X = repair_solution(X)
        population.append(X)
    return population

# 计算目标函数
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

    # 移除最后一个空的前沿
    if not fronts[-1]:
        fronts.pop()

    return fronts, rank

def dominates(obj1, obj2):
    return np.all(obj1 <= obj2) and np.any(obj1 < obj2)

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

# 修复解，确保总生产量满足需求且不超过机器能力
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

# 二元锦标赛选择函数
def binary_tournament(population, objectives, rank, distances):
    idx1, idx2 = np.random.choice(len(population), size=2, replace=False)
    if rank[idx1] < rank[idx2]:
        return population[idx1], objectives[idx1]
    elif rank[idx2] < rank[idx1]:
        return population[idx2], objectives[idx2]
    else:
        if distances[idx1] > distances[idx2]:
            return population[idx1], objectives[idx1]
        else:
            return population[idx2], objectives[idx2]

# 交叉操作
def crossover(parent1, parent2):
    if np.random.rand() < PC:
        point = np.random.randint(1, I * M - 1)  # 交叉点在 1 到 I*M-1 之间，避免切分超出范围
        child1 = np.concatenate([parent1.flatten()[:point], parent2.flatten()[point:]]).reshape(I, M)
        child2 = np.concatenate([parent2.flatten()[:point], parent1.flatten()[point:]]).reshape(I, M)

        # 确保交叉后的子代满足总产量约束
        child1 = repair_solution(child1)
        child2 = repair_solution(child2)

        return child1, child2
    return parent1, parent2

# 变异操作
def mutation(individual):
    if np.random.rand() < PM:
        i = np.random.randint(I)
        j = np.random.randint(M)
        individual[i, j] = np.random.uniform(0, Cap[j] * (1 - R[i, j] / H[i]))  # 在机器的容量范围内随机生成产量

        # 确保变异后的个体满足总产量约束
        individual = repair_solution(individual)
    return individual

# 选择操作，接受预先计算的拥挤距离
def selection(population, objectives, fronts, distances):
    new_population = []
    new_objectives = []
    for front in fronts:
        if len(new_population) + len(front) <= POP_SIZE:
            new_population.extend([population[i] for i in front])
            new_objectives.extend([objectives[i] for i in front])
        else:
            # 使用预先计算的拥挤距离
            front_distances = distances[front]
            sorted_indices = np.argsort(-front_distances)  # 拥挤度降序排序
            remaining_slots = POP_SIZE - len(new_population)
            selected_indices = sorted_indices[:remaining_slots]
            new_population.extend([population[front[i]] for i in selected_indices])
            new_objectives.extend([objectives[front[i]] for i in selected_indices])
            break
    return new_population, np.array(new_objectives)

# 主函数：NSGA-II
def nsga2(init_pop):
    history = []  # 记录每代的帕累托前沿目标值
    population = deepcopy(init_pop)
    objectives = np.array([calculate_objectives(ind) for ind in population])

    for gen in range(GENS):
        fronts, rank = non_dominated_sort(population, objectives)

        # 计算当前种群的拥挤距离
        distances = np.zeros(len(population))
        for front in fronts:
            front_distances = crowding_distance(front, objectives)
            for idx, dist in zip(front, front_distances):
                distances[idx] = dist

        # 生成子代
        offspring = []
        offspring_objectives = []
        while len(offspring) < POP_SIZE:
            # 使用二元锦标赛选择父代，传入拥挤距离
            parent1, parent1_obj = binary_tournament(population, objectives, rank, distances)
            parent2, parent2_obj = binary_tournament(population, objectives, rank, distances)

            # 避免选择同一个个体作为父母
            while np.array_equal(parent1, parent2):
                parent2, parent2_obj = binary_tournament(population, objectives, rank, distances)

            # 交叉和变异
            child1, child2 = crossover(deepcopy(parent1), deepcopy(parent2))
            child1 = mutation(child1)
            child2 = mutation(child2)
            # 计算子代的目标函数值
            child1_obj = calculate_objectives(child1)
            child2_obj = calculate_objectives(child2)

            offspring.extend([child1, child2])
            offspring_objectives.extend([child1_obj, child2_obj])

        # 合并父代和子代，并选择下一代
        combined_population = population + offspring
        combined_objectives = np.vstack((objectives, offspring_objectives))

        # 非支配排序
        fronts, rank = non_dominated_sort(combined_population, combined_objectives)

        # 计算合并种群的拥挤距离
        distances = np.zeros(len(combined_population))
        for front in fronts:
            front_distances = crowding_distance(front, combined_objectives)
            for idx, dist in zip(front, front_distances):
                distances[idx] = dist

        # 选择下一代种群
        population, objectives = selection(combined_population, combined_objectives, fronts, distances)

        # 记录当前代数据
        fronts_his, _ = non_dominated_sort(population, objectives)
        pareto_obj = objectives[fronts_his[0]]
        history.append({
            'generation': gen,
            'pareto': pareto_obj,
            'all_objectives': objectives.copy()
        })

    # 最后一次迭代后，计算帕累托前沿
    fronts, _ = non_dominated_sort(population, objectives)
    pareto_front = fronts[0]
    pareto_solutions = [population[i] for i in pareto_front]
    pareto_objectives = objectives[pareto_front]
    return population, objectives, history

