#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简单测试优化后的模型预测正确率，避免特征名称不匹配问题
"""

import sys
import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

# 添加项目路径到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from qmt_strategy.strategy.model_train import ModelTrainer

# 设置日志
from qmt_strategy.utils.logger import setup_logger
logger = setup_logger()

def test_model_accuracy():
    """测试优化后的模型预测正确率"""
    logger.info("=== 开始测试模型预测正确率 ===")
    
    # 1. 创建简单的测试数据（与训练时特征名称一致）
    logger.info("创建测试数据")
    
    # 生成模拟数据，特征名称与训练时一致
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    # 创建特征，使用与训练时相同的特征名称
    feature_names = [f'feature_{i}' for i in range(n_features)]
    X = np.random.randn(n_samples, n_features)
    
    # 创建标签（模拟涨停数据，10%涨停概率）
    y = np.random.choice([0, 1], size=n_samples, p=[0.9, 0.1])
    
    # 转换为DataFrame
    X_df = pd.DataFrame(X, columns=feature_names)
    
    logger.info(f"测试数据样本数量：{len(X_df)}")
    logger.info(f"涨停样本数量：{sum(y)}")
    logger.info(f"非涨停样本数量：{len(y) - sum(y)}")
    
    # 2. 初始化模型训练器
    logger.info("初始化模型训练器")
    model_trainer = ModelTrainer()
    
    # 3. 加载预训练模型
    logger.info("加载预训练模型")
    if not model_trainer.load_model():
        logger.error("无法加载模型，测试失败")
        return False
    
    # 4. 模型预测
    logger.info("进行模型预测")
    predictions = model_trainer.predict(X_df)
    if predictions is None:
        logger.error("模型预测失败")
        return False
    
    y_pred = predictions['predictions']
    y_prob = predictions['probabilities']
    
    # 5. 计算预测指标
    logger.info("计算预测指标")
    metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred, zero_division=0),
        'recall': recall_score(y, y_pred, zero_division=0),
        'f1_score': f1_score(y, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y, y_prob)
    }
    
    # 6. 输出预测结果
    logger.info("=== 模型预测正确率测试结果 ===")
    logger.info(f"准确率 (Accuracy): {metrics['accuracy']:.4f}")
    logger.info(f"精确率 (Precision): {metrics['precision']:.4f}")
    logger.info(f"召回率 (Recall): {metrics['recall']:.4f}")
    logger.info(f"F1分数 (F1 Score): {metrics['f1_score']:.4f}")
    logger.info(f"AUC-ROC分数: {metrics['roc_auc']:.4f}")
    
    # 7. 保存测试结果
    result_file = f"simple_test_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
    result_df = pd.DataFrame([metrics])
    result_df.to_csv(result_file, index=False)
    logger.info(f"测试结果保存到：{result_file}")
    
    # 8. 分析预测结果分布
    logger.info("=== 预测结果分布 ===")
    pred_counts = pd.Series(y_pred).value_counts()
    logger.info(f"预测为涨停的数量：{pred_counts.get(1, 0)}")
    logger.info(f"预测为非涨停的数量：{pred_counts.get(0, 0)}")
    
    # 9. 分析概率分布
    logger.info("=== 涨停概率分布 ===")
    prob_mean = np.mean(y_prob)
    prob_std = np.std(y_prob)
    prob_max = np.max(y_prob)
    prob_min = np.min(y_prob)
    logger.info(f"平均涨停概率：{prob_mean:.4f}")
    logger.info(f"涨停概率标准差：{prob_std:.4f}")
    logger.info(f"最大涨停概率：{prob_max:.4f}")
    logger.info(f"最小涨停概率：{prob_min:.4f}")
    
    return True

def main():
    """主函数"""
    success = test_model_accuracy()
    
    if success:
        logger.info("=== 模型预测正确率测试完成 ===")
        return 0
    else:
        logger.error("=== 模型预测正确率测试失败 ===")
        return 1

if __name__ == "__main__":
    sys.exit(main())