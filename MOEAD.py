import numpy as np
import plotly.graph_objs as go
from copy import deepcopy
from zjs_parameterconfiguration import M,I,Cap,C,P,R,H,D,PC,PM,POP_SIZE,GENS,GMGEN_RATIO,BETA,DRATIO
np.random.seed(42)

PSUM = np.sum(P)

T = 20  # 邻域大小

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

def initialize_weight_vectors(pop_size, num_objectives):
    weight_vectors = []
    for _ in range(pop_size):
        weight = np.random.dirichlet(np.ones(num_objectives))
        weight_vectors.append(weight)
    return np.array(weight_vectors)

def initialize_neighborhoods(weight_vectors, T):
    neighborhoods = []
    for i in range(len(weight_vectors)):
        distances = np.linalg.norm(weight_vectors - weight_vectors[i], axis=1)
        neighbor_indices = np.argsort(distances)[:T]
        neighborhoods.append(neighbor_indices)
    return np.array(neighborhoods)

def initialize_reference_point(objectives):
    return np.min(objectives, axis=0)

def calculate_tchebycheff(obj, weight, z):
    return np.max(weight * np.abs(obj - z))

def crossover(parent1, parent2):
    if np.random.rand() < PC:
        point = np.random.randint(1, I * M - 1)  # 交叉点在 1 到 I*M-1 之间，避免切分超出范围
        child1 = np.concatenate([parent1.flatten()[:point], parent2.flatten()[point:]]).reshape(I, M)
        child2 = np.concatenate([parent2.flatten()[:point], parent1.flatten()[point:]]).reshape(I, M)

        child1 = repair_solution(child1)
        child2 = repair_solution(child2)

        return child1, child2
    return parent1, parent2

def mutation(individual):
    if np.random.rand() < PM:
        i = np.random.randint(I)
        j = np.random.randint(M)
        individual[i, j] = np.random.uniform(0, Cap[j] * (1 - R[i, j] / H[i]))  # 在机器的容量范围内随机生成产量

        # 确保变异后的个体满足总产量约束
        individual = repair_solution(individual)
    return individual

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

def update_external_population(EP, solution):
    solution_obj = calculate_objectives(solution)
    EP = [s for s in EP if not dominates(solution_obj, calculate_objectives(s))]
    if not any(dominates(calculate_objectives(s), solution_obj) for s in EP):
        EP.append(solution)
    return EP

def dominates(obj1, obj2):
    obj1 = np.array(obj1)
    obj2 = np.array(obj2)
    return np.all(obj1 <= obj2) and np.any(obj1 < obj2)

def normalize_objectives(objectives, z, z_max):
    return (objectives - z) / (z_max - z)

def moead(init_pop):
    # 使用统一初始种群
    population = deepcopy(init_pop)
    objectives = np.array([calculate_objectives(ind) for ind in population])

    # 初始化历史记录
    history = []

    # 初始化MOEA/D特有组件
    weight_vectors = initialize_weight_vectors(POP_SIZE, 3)
    neighborhoods = initialize_neighborhoods(weight_vectors, T)
    z = initialize_reference_point(objectives)
    z_max = np.max(objectives, axis=0)
    EP = []

    for gen in range(GENS):
        for i in range(POP_SIZE):
            neighbors = neighborhoods[i]
            idx1, idx2 = np.random.choice(neighbors, 2, replace=False)
            p1, p2 = population[idx1], population[idx2]

            # 交叉和变异
            child1, child2 = crossover(deepcopy(p1), deepcopy(p2))
            child1 = mutation(child1)
            child2 = mutation(child2)

            # 选择较优子代
            child = child1 if (
                    calculate_tchebycheff(calculate_objectives(child1), weight_vectors[i], z) <
                    calculate_tchebycheff(calculate_objectives(child2), weight_vectors[i], z)
            ) else child2

            # 修复解
            child = repair_solution(child)
            obj_child = calculate_objectives(child)

            # 更新参考点和邻域
            z = np.minimum(z, obj_child)
            z_max = np.maximum(z_max, obj_child)
            for neighbor in neighbors:
                if calculate_tchebycheff(obj_child, weight_vectors[neighbor], z) < calculate_tchebycheff(
                        objectives[neighbor], weight_vectors[neighbor], z):
                    population[neighbor] = child
                    objectives[neighbor] = obj_child

            # 更新外部种群
            EP = update_external_population(EP, child)

        # 记录当前代数据
        current_pareto = np.array([calculate_objectives(ind) for ind in EP])
        history.append({
            'generation': gen,
            'pareto': current_pareto,
            'all_objectives': objectives.copy()
        })

    # 提取最终非支配解
    pareto_solutions = EP
    pareto_objectives = np.array([calculate_objectives(ind) for ind in EP])
    return population, objectives, history

