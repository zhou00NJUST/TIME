import matplotlib.pyplot as plt
import numpy as np
import torch

def compute_curvature_importance(trajectories):
    """
    计算基于曲率的时间点重要性指标。

    参数：
    trajectories: numpy 数组，形状为 [N, 12, 2]

    返回：
    importance_scores: numpy 数组，形状为 [N, 12]，每个时间点的曲率重要性
    """
    N, T, _ = trajectories.shape
    # 初始化重要性分数
    importance_scores = np.zeros((N, T))
    # 计算速度向量（相邻点之差）
    v_prev = trajectories[:, 1:-1, :] - trajectories[:, :-2, :]
    v_next = trajectories[:, 2:, :] - trajectories[:, 1:-1, :]
    # 计算速度向量的模（长度）
    v_prev_norm = np.linalg.norm(v_prev, axis=2)
    v_next_norm = np.linalg.norm(v_next, axis=2)
    # 计算速度向量之间的点积
    dot_product = np.sum(v_prev * v_next, axis=2)
    # 计算夹角的余弦值，防止除以零
    cos_theta = dot_product / (v_prev_norm * v_next_norm + 1e-8)
    # 将余弦值裁剪到 [-1, 1] 范围内，避免数值误差
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    # 计算夹角 theta
    theta = np.arccos(cos_theta)
    # 将 theta 的绝对值作为重要性指标
    importance_scores[:, 1:-1] = np.abs(theta)
    return importance_scores

def compute_perpendicular_distance_importance(trajectories):
    """
    计算基于垂直距离的时间点重要性指标。

    参数：
    trajectories: numpy 数组，形状为 [N, 12, 2]

    返回：
    importance_scores: numpy 数组，形状为 [N, 12]，每个时间点的垂直距离重要性
    """
    N, T, _ = trajectories.shape
    # 初始化重要性分数
    importance_scores = np.zeros((N, T))
    # 起点和终点
    start_points = trajectories[:, 0, :]      # 形状 [N, 2]
    end_points = trajectories[:, -1, :]       # 形状 [N, 2]
    # 起点到终点的向量
    line_vec = end_points - start_points      # 形状 [N, 2]
    # 计算线段长度的平方，防止除以零
    line_length = np.linalg.norm(line_vec, axis=1, keepdims=True) + 1e-8
    # 对于每个中间时间点，计算到起终点连线的垂直距离
    for t in range(1, T - 1):
        point = trajectories[:, t, :]         # 形状 [N, 2]
        # 向量从起点到当前点
        vec = point - start_points            # 形状 [N, 2]
        # 计算叉积
        cross_prod = np.abs(line_vec[:, 0] * vec[:, 1] - line_vec[:, 1] * vec[:, 0])
        # 垂直距离
        distance = cross_prod / line_length.squeeze()
        importance_scores[:, t] = distance
    return importance_scores

def keep_most_important_points(trajectories):
    """
    选择每条轨迹中最重要的四个时间点：起点、终点、曲率指标最重要的点和垂直距离指标最重要的点。

    参数：
    trajectories: numpy 数组，形状为 [N, 12, 2]

    返回：
    simplified_trajectories: numpy 数组，形状为 [N, 4, 2]，每条轨迹选取的四个关键点的坐标
    """
    N, T, _ = trajectories.shape

    # 计算曲率和垂直距离的重要性指标
    curvature_scores = compute_curvature_importance(trajectories)
    distance_scores = compute_perpendicular_distance_importance(trajectories)

    # 排除起点和终点，防止选中它们作为关键点
    curvature_scores[:, [0, T - 1]] = -np.inf
    distance_scores[:, [0, T - 1]] = -np.inf

    # 初始化选定的时间点索引数组
    selected_indices = np.zeros((N, 4), dtype=int)
    selected_indices[:, 0] = 0           # 起点索引
    selected_indices[:, -1] = T - 1      # 终点索引

    # 获取每条轨迹曲率指标最大的时间点索引
    curvature_idx = np.argmax(curvature_scores, axis=1)
    # 获取每条轨迹垂直距离指标最大的时间点索引
    distance_idx = np.argmax(distance_scores, axis=1)

    # 确保曲率和垂直距离选取的时间点不重合
    for i in range(N):
        if curvature_idx[i] == distance_idx[i]:
            # 如果重合，找到曲率指标的次大值索引
            curvature_scores_i = curvature_scores[i].copy()
            curvature_scores_i[curvature_idx[i]] = -np.inf  # 排除已选中的点
            curvature_idx[i] = np.argmax(curvature_scores_i)
            # 检查新的索引是否仍与 distance_idx 重合
            if curvature_idx[i] == distance_idx[i]:
                # 如果仍然重合，找到垂直距离指标的次大值索引
                distance_scores_i = distance_scores[i].copy()
                distance_scores_i[distance_idx[i]] = -np.inf  # 排除已选中的点
                distance_idx[i] = np.argmax(distance_scores_i)

    # 将选定的时间点索引加入数组
    selected_indices[:, 1] = curvature_idx
    selected_indices[:, 2] = distance_idx

    # 对每条轨迹的时间点索引进行排序，保持时间顺序
    sorted_indices = np.sort(selected_indices, axis=1)

    # 根据选定的时间点索引，提取对应的坐标
    simplified_trajectories = trajectories[np.arange(N)[:, None], sorted_indices, :]

    return simplified_trajectories

    

if __name__ == '__main__':
    x = np.arange(0, 1, 0.1)
    
    traj2 = np.stack([x, x ** 2], axis=1)
    traj3 = np.stack([x, x ** 3], axis=1)
    traj = np.stack([traj2, traj3], axis=0)
    curvature_scores = compute_curvature_importance(traj)
    
    for i in range(2):
        plt.figure(i)

        plt.subplot(2, 1, 1)
        plt.plot(traj[i, :, 0], traj[i, :, 1], 'o-')
        plt.title('original trajectory')

        plt.subplot(2, 1, 2)
        kept_traj = keep_most_important_points(traj)
        plt.plot(kept_traj[i, :, 0], kept_traj[i, :, 1], 'o-')
        plt.title('important points')
      
    

    plt.show()