import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd  # 用于格式化输出误差结果

# ===================== 1. 基础配置 =====================
plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文乱码
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.alpha'] = 0.2  # 降低网格干扰，突出粗线条
plt.rcParams['legend.framealpha'] = 0.98  # 图例高透明度

# ===================== 2. 核心配置（线条整体加粗） =====================
model_folders = [
    'OPSDCHformer',
    'OPSDautoformer',
    'OPSDtransformer',
    'OPSDInformer',
    'OPSDLSTM',
    'OPSDTCN'
]
model_names = [
    'CHformer（提出模型）',
    'Autoformer',
    'Transformer',
    'Informer',
    'LSTM',
    'TCN',
]
true_file = 'OPSDCHformer/true.npy'
styles = {
    # CHformer：大红色 + 超粗线（2.5）→ 视觉核心
    'CHformer（提出模型）': {'color': '#FF0000', 'linestyle': '-', 'lw': 2.5, 'alpha': 0.95},
    # 真实值：纯黑色 + 超粗线（2.5）→ 基准参考
    '真实值': {'color': '#000000', 'linestyle': '-', 'lw': 2.5, 'alpha': 0.9},
    # 其他模型：同步加粗（2.1/2.0/1.9/1.8）+ 高对比色
    'Autoformer': {'color': '#0066FF', 'linestyle': '-', 'lw': 2.1, 'alpha': 0.9},
    'Transformer': {'color': '#00CC66', 'linestyle': '-', 'lw': 2.0, 'alpha': 0.9},
    'GRU': {'color': '#FF9900', 'linestyle': '-', 'lw': 1.9, 'alpha': 0.9},
    'Informer': {'color': '#9933FF', 'linestyle': '-', 'lw': 1.9, 'alpha': 0.9},
    'LSTM': {'color': '#00CCCC', 'linestyle': '-', 'lw': 1.8, 'alpha': 0.9},
    'ResNet': {'color': '#005000', 'linestyle': '-', 'lw': 2.0, 'alpha': 0.9},
    'TCN': {'color': '#CC9900', 'linestyle': '-', 'lw': 1.8, 'alpha': 0.9}
}


# ===================== 3. 误差计算函数 =====================
def calculate_metrics(pred, true):
    """
    计算MAE和RMSE
    :param pred: 预测值数组（一维）
    :param true: 真实值数组（一维）
    :return: mae, rmse
    """
    # 确保输入维度一致且无NaN值
    mask = ~np.isnan(pred) & ~np.isnan(true)
    pred_clean = pred[mask]
    true_clean = true[mask]

    mae = np.mean(np.abs(pred_clean - true_clean))
    rmse = np.sqrt(np.mean((pred_clean - true_clean) ** 2))
    return round(mae, 4), round(rmse, 4)


# 初始化误差结果存储
error_results = []

# ===================== 4. 读取数据 & 计算误差 =====================
for V in range(3):
    pred_data = {}
    for folder, name in zip(model_folders, model_names):
        pred_path = os.path.join(folder, 'pred.npy')
        if os.path.exists(pred_path):  # 增加文件存在性检查
            data = np.load(pred_path)
            data = data[:, :, V]
            pred_data[name] = data
        else:
            print(f"⚠️ {pred_path} 文件不存在，跳过 {name}")

    # 读取真实值
    if os.path.exists(true_file):
        true_data = np.load(true_file)
        true_data = true_data[:, :, V]
    else:
        print(f"❌ 真实值文件 {true_file} 不存在！")
        continue
    # ===================== 计算并保存误差 =====================
    for name in model_names:
        if name in pred_data:
            mae, rmse = calculate_metrics(pred_data[name][:2176, :], true_data[:2176, :])
            error_results.append({
                '维度V': V,
                '模型名称': name,
                'MAE': mae,
                'RMSE': rmse
            })
            print(f"📊 V={V} | {name} | MAE={mae} | RMSE={rmse}")
    # 数据维度处理
    true_1d = true_data[:, 1]
    all_data = [true_1d]
    for name in pred_data:
        pred_1d = pred_data[name][1:, 1]
        pred_data[name] = pred_1d
        all_data.append(pred_1d)

    # 截断到指定长度
    show_length = 24 * 10  # 24×5=120个值
    start = 240
    min_total_length = min([len(d) for d in all_data])
    final_length = min(show_length, min_total_length)
    true_final = true_1d[start:start + final_length]
    for name in pred_data:
        pred_data[name] = pred_data[name][start:start + final_length]
    x = np.arange(final_length)



    # ===================== 6. 绘图（加粗线条） =====================
    fig, ax = plt.subplots(figsize=(18, 8))

    # 先画真实值（黑色基准，底层）
    ax.plot(x, true_final, label='真实值', **styles['真实值'])

    # 再画所有模型（CHformer最后画，确保在顶层）
    for name in model_names:
        if name in pred_data:
            ax.plot(x, pred_data[name], label=name, **styles[name])

    # 优化坐标轴
    ax.tick_params(axis='both', labelsize=11, colors='#333333')
    ax.set_xticks(np.arange(0, final_length + 1, 24))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#DDDDDD')
    ax.spines['bottom'].set_color('#DDDDDD')

    # 调整布局
    plt.tight_layout()
    # 保存+显示
    plt.savefig('OPSD' + str(V) + '.png', dpi=300, bbox_inches='tight')
    plt.close()  # 关闭画布释放内存

# ===================== 结果汇总输出 =====================
# 转换为DataFrame方便查看和保存
error_df = pd.DataFrame(error_results)
print("\n========== 所有模型误差汇总 ==========")
print(error_df.to_string(index=False))

# 可选：保存误差结果到CSV文件
error_df.to_csv('OPSDmetirc.csv', index=False, encoding='utf-8-sig')
print("\n✅ 误差结果已保存到「模型误差结果.csv」文件")