from sklearn.metrics import mean_squared_error
import numpy as np

# 计算均方根误差（RMSE）
def compute_rmse(predicted_ratings, true_ratings):
    return np.sqrt(mean_squared_error(true_ratings, predicted_ratings))

# 其他评估指标可以根据需求添加
