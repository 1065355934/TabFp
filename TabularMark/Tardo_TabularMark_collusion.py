import numpy as np
import pandas as pd
from scipy.stats import norm

# 1. 生成模拟数据集（数值型）
np.random.seed(42)
data = pd.DataFrame({
    'Feature1': np.random.normal(0, 10, 1000),
    'Feature2': np.random.randint(0, 10, 1000),
    'Target': np.random.normal(50, 5, 1000)
})


# 2. 水印嵌入函数
def embed_watermark(data, target_col, p=2.0, n_w=50, k=100):
    original_data = data.copy()
    key_cells = np.random.choice(data.index, n_w, replace=False)
    seeds = [hash((idx, target_col)) % (2 ** 32) for idx in key_cells]

    for idx, seed in zip(key_cells, seeds):
        np.random.seed(seed)
        # 分区扰动范围 [-p, p]
        intervals = np.linspace(-p, p, k + 1)
        green = np.random.choice([0, 1], k, replace=True).astype(bool)
        green_intervals = intervals[:-1][green]
        # 从绿色区域随机选择扰动值
        noise = np.random.choice(green_intervals)
        data.loc[idx, target_col] += noise

    return original_data, data, key_cells, seeds


# 3. 水印检测函数
def detect_watermark(suspect_data, original_data, target_col, key_cells, seeds, alpha=1.96):
    n_w = len(key_cells)
    n_g = 0

    for idx, seed in zip(key_cells, seeds):
        original_val = original_data.loc[idx, target_col]
        suspect_val = suspect_data.loc[idx, target_col]
        delta = suspect_val - original_val

        np.random.seed(seed)
        intervals = np.linspace(-p, p, k + 1)
        green = np.random.choice([0, 1], k, replace=True).astype(bool)
        green_intervals = intervals[:-1][green]

        # 判断偏差是否在绿色区域
        # in_green = any((delta >= left) & (delta <= right) for left, right in zip(green_intervals, intervals[1:][green]))
        in_green = any(((delta >= left) & (delta <= right)).any() for left, right in zip(green_intervals, intervals[1:][green]))
        if in_green:
            n_g += 1

    z_score = (2 * (n_g - 0.5 * n_w)) / np.sqrt(n_w)
    return z_score > alpha, z_score



class UserFingerprint:
    def __init__(self, user_id, p=2.0, n_w=50, k=100):
        self.user_id = user_id
        self.seed = hash(user_id) % (2 ** 32)  # 用户唯一种子
        self.p = p
        self.n_w = n_w
        self.k = k

    def embed(self, data, target_col):
        np.random.seed(self.seed)
        key_cells = np.random.choice(data.index, self.n_w, replace=False)
        watermarked_data = data.copy()

        for idx in key_cells:
            np.random.seed(hash((self.seed, idx)) % (2 ** 32))  # 单元格级别种子
            intervals = np.linspace(-self.p, self.p, self.k + 1)
            green = np.random.choice([0, 1], self.k, replace=True).astype(bool)
            green_intervals = intervals[:-1][green]
            noise = np.random.choice(green_intervals)
            watermarked_data.loc[idx, target_col] += noise

        return watermarked_data, key_cells

def collusion_attack(datasets, method='average'):
    attacked_data = datasets[0].copy()
    for col in attacked_data.columns:
        if col == 'Target':  # 假设仅对目标列攻击
            stacked = np.stack([d[col].values for d in datasets])
            if method == 'average':
                attacked_data[col] = stacked.mean(axis=0)
            elif method == 'majority':
                attacked_data[col] = np.apply_along_axis(lambda x: np.bincount(x.astype(int)).argmax(), 0, stacked)
            elif method == 'random':
                # attacked_data[col] = np.random.choice(stacked.T, size=len(attacked_data))
                attacked_data[col] = [np.random.choice(stacked[:,i], size=len(stacked[:,i])) for  i in range(len(attacked_data))]
    return attacked_data


import numpy as np
from scipy.stats import ortho_group


class TardosFingerprint:
    def __init__(self, user_id, n_w, m):
        self.user_id = user_id
        self.n_w = n_w  # 指纹长度
        self.m = m  # 最大合谋用户数
        np.random.seed(hash(user_id) % (2 ** 32))
        # 生成正交编码矩阵（简化实现）
        self.code = np.random.choice([-1, 1], size=n_w)

    def get_perturbation(self, p=2.0):
        noise = np.random.uniform(-p, p, self.n_w)
        return self.code * noise


def embed_watermark_with_tardos(data, user_fingerprint, target_col):
    n_w = user_fingerprint.n_w
    key_cells = np.random.choice(data.index, n_w, replace=False)
    perturbations = user_fingerprint.get_perturbation()

    watermarked_data = data.copy()
    for idx, delta in zip(key_cells, perturbations):
        watermarked_data.loc[idx, target_col] += delta
    return watermarked_data, key_cells


def detect_collusion(suspect_data, original_data, users, target_col, alpha=1.96):
    detected_users = []
    for user in users:
        # 恢复用户的关键单元格和编码
        np.random.seed(hash(user.user_id) % (2 ** 32))
        key_cells = np.random.choice(original_data.index, user.n_w, replace=False)
        code = user.code

        # 计算统计量 T_i
        T = 0
        for j, idx in enumerate(key_cells):
            delta_suspect = suspect_data.loc[idx, target_col] - original_data.loc[idx, target_col]
            T += code[j] * delta_suspect
        T /= user.n_w

        # 假设检验
        z_score = T * np.sqrt(user.n_w)
        if abs(z_score) >= alpha:
            detected_users.append(user.user_id)
    return detected_users


p=2.0
k=100
original_df, watermarked_df, key_cells, seeds = embed_watermark(data, 'Target', p=2.0, n_w=50, k=100)

# 5. 执行水印检测
is_watermarked, z = detect_watermark(watermarked_df, original_df, 'Target', key_cells, seeds)
print(f"检测到水印: {is_watermarked}, Z分数: {z:.2f}")

# 6. 模拟攻击（添加随机噪声）
attacked_df = watermarked_df.copy()
attacked_df['Target'] += np.random.normal(0, 1, len(data))
is_watermarked_attacked, z_attacked = detect_watermark(attacked_df, original_df, 'Target', key_cells, seeds)
print(f"攻击后检测到水印: {is_watermarked_attacked}, Z分数: {z_attacked:.2f}")





# 生成多个用户的水印数据集
users = [UserFingerprint(f'user_{i}') for i in range(5)]
original_data = data  # 假设已生成原始数据
watermarked_datasets = [user.embed(original_data, 'Target')[0] for user in users]

# 模拟合谋攻击
attacked_avg = collusion_attack(watermarked_datasets, 'average')
attacked_majority = collusion_attack(watermarked_datasets, 'majority')
attacked_random = collusion_attack(watermarked_datasets, 'random')

# 检测攻击后的数据集
def batch_detect(attacked_data, users, original_data):
    results = {}
    for user in users:
        _, key_cells = user.embed(original_data, 'Target')  # 获取该用户的关键单元格
        seeds = [hash((user.seed, idx)) % (2**32) for idx in key_cells]
        detected, z = detect_watermark(attacked_data, original_data, 'Target', key_cells, seeds)
        results[user.user_id] = (detected, z)
    return results

# 输出检测结果
print("平均攻击检测结果:", batch_detect(attacked_avg, users, original_data))
print("多数投票攻击检测结果:", batch_detect(attacked_majority, users, original_data))
print("随机攻击检测结果:", batch_detect(attacked_random, users, original_data))


#---------------------with tardo--------------------------
# 生成5个用户的合谋数据集
users = [TardosFingerprint(f'user_{i}', n_w=100, m=5) for i in range(5)]
original_data = data # 原始数据
watermarked_datasets = [embed_watermark_with_tardos(original_data, user, 'Target') for user in users]

# 模拟平均攻击
attacked_data = original_data.copy()
for idx in original_data.index:
    values = [watermarked_datasets[i][0].loc[idx, 'Target'] for i in range(5)]
    attacked_data.loc[idx, 'Target'] = np.mean(values)

# 检测合谋用户
detected = detect_collusion(attacked_data, original_data, users, 'Target')
print(f"检测到的合谋用户: {detected}")