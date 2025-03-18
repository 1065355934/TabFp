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
        in_green = any((delta >= left) & (delta <= right) for left, right in zip(green_intervals, intervals[1:][green]))
        if in_green:
            n_g += 1

    z_score = (2 * (n_g - 0.5 * n_w)) / np.sqrt(n_w)
    return z_score > alpha, z_score


# 4. 执行水印嵌入
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