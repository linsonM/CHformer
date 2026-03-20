import numpy as np
import os
import pandas as pd  # 用于格式化输出误差结果

# ===================== 1. 核心配置（仅保留文件路径相关） =====================
model_folders = [
    'OPSDCHformer',
    'OPSDCHformer_ATTN',
    'OPSDCHformer_HATTN',
    'OPSDCHformer_EM',
    'OPSDCHformer_MSE',
]
model_names = [
    'C1',
    'C2',
    'C3',
    'C4',
    'C5',
]
true_file = 'OPSDCHformer/true.npy'  # 真实值文件路径


# ===================== 2. 误差计算函数（核心保留） =====================
def calculate_metrics(pred, true):
    """
    计算MAE（平均绝对误差）和RMSE（均方根误差）
    :param pred: 预测值数组（一维/多维，函数内自动清洗NaN）
    :param true: 真实值数组（维度需与pred匹配）
    :return: 四舍五入后的mae, rmse（保留4位小数）
    """
    # 确保输入维度一致且过滤NaN值（避免计算误差）
    mask = ~np.isnan(pred) & ~np.isnan(true)
    pred_clean = pred[mask]
    true_clean = true[mask]

    mae = np.mean(np.abs(pred_clean - true_clean))
    rmse = np.sqrt(np.mean((pred_clean - true_clean) ** 2))
    return round(mae, 4), round(rmse, 4)


# ===================== 3. 初始化误差结果存储 =====================
error_results = []

# ===================== 4. 读取数据 & 批量计算误差 =====================
# 遍历3个维度（V=0,1,2）
for V in range(3):
    print(f"\n===== 开始计算 V={V} 维度的模型误差 =====")

    # 读取各模型预测值
    pred_data = {}
    for folder, name in zip(model_folders, model_names):
        pred_path = os.path.join(folder, 'pred.npy')
        if os.path.exists(pred_path):
            # 读取预测值并提取对应维度V的数据
            data = np.load(pred_path)
            pred_data[name] = data[:, :, V]
            print(f"✅ 成功读取 {name} 的预测数据（V={V}）")
        else:
            print(f"⚠️ 跳过 {name}：文件 {pred_path} 不存在")

    # 读取真实值并提取对应维度V的数据
    if not os.path.exists(true_file):
        print(f"❌ 终止计算 V={V}：真实值文件 {true_file} 不存在！")
        continue
    true_data = np.load(true_file)[:, :, V]

    # 计算每个模型的误差并存储结果
    for name in model_names:
        if name in pred_data:
            mae, rmse = calculate_metrics(pred_data[name][:2176, :], true_data[:2176, :])
            error_results.append({
                '维度V': V,
                '模型名称': name,
                'MAE': mae,
                'RMSE': rmse
            })
            print(f"📊 {name} | MAE={mae} | RMSE={rmse}")

# ===================== 5. 结果汇总 & 保存 =====================
# 转换为DataFrame，方便查看和后续分析
error_df = pd.DataFrame(error_results)

# 打印汇总结果
print("\n========== 所有模型误差汇总（按维度+模型排序） ==========")
print(error_df.to_string(index=False))

# 保存误差结果到CSV文件（支持中文，utf-8-sig编码）
csv_path = 'OPSDmetric_ab.csv'  # 注意原代码拼写错误：metirc → metric
error_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
print(f"\n✅ 误差结果已保存到：{csv_path}")
