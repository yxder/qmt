#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
优化模型测试脚本，提高预测涨停股票的正确率并统计预测为涨停但未涨停股票的平均涨幅
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
# 导入配置文件
import qmt_strategy.config as config

# 设置日志
from qmt_strategy.utils.logger import setup_logger
logger = setup_logger()

def generate_realistic_data(n_samples=10000, n_features=10, seed=42):
    """生成更真实的模拟数据，考虑特征与涨停的相关性"""
    np.random.seed(seed)
    
    # 创建特征，其中前半部分特征与涨停正相关，后半部分特征与涨停负相关
    feature_names = [f'feature_{i}' for i in range(n_features)]
    
    # 生成特征数据
    X = np.random.randn(n_samples, n_features)
    
    # 调整特征分布，使得涨停样本的特征更具区分性
    # 计算综合得分，用于生成更真实的标签
    # 前半部分特征正相关，后半部分特征负相关
    half_features = n_features // 2
    weights = np.concatenate([np.ones(half_features) * 0.5, np.ones(n_features - half_features) * -0.3])
    scores = X @ weights + np.random.randn(n_samples) * 0.5
    
    # 使用sigmoid函数转换为概率
    probs = 1 / (1 + np.exp(-scores))
    # 调整概率，使得涨停样本比例约为5%
    probs = 0.05 * probs / np.mean(probs)
    probs = np.clip(probs, 0, 0.9)
    
    # 生成标签
    y = np.random.binomial(1, probs)
    
    # 生成模拟的实际涨幅数据，涨停样本涨幅为10%，非涨停样本涨幅为随机
    # 生成一个基础涨幅，与综合得分相关
    base_gain = scores * 0.02 + np.random.randn(n_samples) * 0.03
    # 涨停样本涨幅固定为10%
    actual_gain = np.where(y == 1, 0.1, base_gain)
    # 限制涨幅范围在-10%到15%
    actual_gain = np.clip(actual_gain, -0.1, 0.15)
    
    # 转换为DataFrame
    X_df = pd.DataFrame(X, columns=feature_names)
    
    logger.info(f"生成数据样本数量：{len(X_df)}")
    logger.info(f"涨停样本数量：{sum(y)}")
    logger.info(f"非涨停样本数量：{len(y) - sum(y)}")
    logger.info(f"涨停样本占比：{sum(y)/len(y):.2%}")
    
    return X_df, y, actual_gain

def optimize_model_threshold(model_trainer, X_test, y_test, actual_gain):
    """优化模型阈值，提高预测涨停股票的正确率"""
    logger.info("=== 开始优化模型阈值 ===")
    
    # 获取模型预测概率
    predictions = model_trainer.predict(X_test)
    if predictions is None:
        logger.error("模型预测失败")
        return None
    
    y_prob = predictions['probabilities']
    
    # 测试不同的阈值
    thresholds = np.arange(0.1, 0.95, 0.05)
    best_threshold = 0.5
    best_precision = 0.0
    best_results = None
    
    results_list = []
    
    for threshold in thresholds:
        # 根据阈值生成预测标签
        y_pred = (y_prob >= threshold).astype(int)
        
        # 计算评估指标
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_prob)
        
        # 统计预测为涨停但未涨停股票的平均涨幅
        false_positive_mask = (y_pred == 1) & (y_test == 0)
        false_positive_count = np.sum(false_positive_mask)
        avg_false_positive_gain = np.mean(actual_gain[false_positive_mask]) if false_positive_count > 0 else 0
        
        # 统计预测为涨停且实际涨停的股票数量
        true_positive_count = np.sum((y_pred == 1) & (y_test == 1))
        
        # 统计预测为涨停的总数量
        predicted_limit_up_count = np.sum(y_pred == 1)
        
        # 保存结果
        result = {
            'threshold': threshold,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'predicted_limit_up_count': predicted_limit_up_count,
            'true_positive_count': true_positive_count,
            'false_positive_count': false_positive_count,
            'avg_false_positive_gain': avg_false_positive_gain
        }
        
        results_list.append(result)
        
        logger.info(f"阈值: {threshold:.2f}, 准确率: {accuracy:.4f}, 精确率: {precision:.4f}, "
                   f"召回率: {recall:.4f}, F1分数: {f1:.4f}, 预测涨停数: {predicted_limit_up_count}, "
                   f"实际涨停数: {true_positive_count}, 误报数: {false_positive_count}, "
                   f"误报平均涨幅: {avg_false_positive_gain:.4f}")
        
        # 更新最佳阈值（以精确率为主要指标，兼顾召回率）
        if precision > best_precision and recall > 0.1:
            best_precision = precision
            best_threshold = threshold
            best_results = result
    
    logger.info(f"=== 最佳阈值优化结果 ===")
    logger.info(f"最佳阈值: {best_threshold:.2f}")
    logger.info(f"精确率: {best_results['precision']:.4f}")
    logger.info(f"召回率: {best_results['recall']:.4f}")
    logger.info(f"F1分数: {best_results['f1_score']:.4f}")
    logger.info(f"预测涨停数: {best_results['predicted_limit_up_count']}")
    logger.info(f"实际涨停数: {best_results['true_positive_count']}")
    logger.info(f"误报数: {best_results['false_positive_count']}")
    logger.info(f"误报平均涨幅: {best_results['avg_false_positive_gain']:.4f}")
    
    return best_threshold, best_results, results_list

def test_model_accuracy():
    """测试优化后的模型预测正确率"""
    logger.info("=== 开始测试模型预测正确率 ===")
    
    # 1. 生成更真实的测试数据
    logger.info("生成更真实的测试数据")
    X, y, actual_gain = generate_realistic_data(n_samples=10000, n_features=10, seed=42)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test, gain_train, gain_test = train_test_split(
        X, y, actual_gain, test_size=0.3, random_state=42, stratify=y
    )
    
    logger.info(f"训练集样本数量：{len(X_train)}")
    logger.info(f"测试集样本数量：{len(X_test)}")
    logger.info(f"测试集涨停样本数量：{sum(y_test)}")
    logger.info(f"测试集非涨停样本数量：{len(y_test) - sum(y_test)}")
    
    # 2. 初始化模型训练器
    logger.info("初始化模型训练器")
    model_trainer = ModelTrainer()
    
    # 3. 直接训练一个新的模型，确保特征名称匹配
    logger.info(f"直接训练新的{config.MODEL_TYPE}模型")
    # 创建带标签的训练数据
    train_data = X_train.copy()
    train_data['label'] = y_train
    # 训练模型
    if not model_trainer.train(train_data):
        logger.error("模型训练失败，测试失败")
        return False
    
    # 4. 优化模型阈值
    logger.info("优化模型阈值")
    best_threshold, best_results, results_list = optimize_model_threshold(
        model_trainer, X_test, y_test, gain_test
    )
    
    if best_results is None:
        logger.error("阈值优化失败")
        return False
    
    # 5. 保存测试结果
    result_file = f"optimized_test_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
    results_df = pd.DataFrame(results_list)
    results_df.to_csv(result_file, index=False)
    logger.info(f"测试结果保存到：{result_file}")
    
    # 6. 分析不同模型的表现
    logger.info("=== 模型表现分析 ===")
    logger.info(f"最佳阈值下预测涨停准确率：{best_results['precision']:.4f}")
    logger.info(f"预测为涨停但未涨停股票的平均涨幅：{best_results['avg_false_positive_gain']:.4f}")
    logger.info(f"在最佳阈值下，每预测100只涨停股票，实际涨停{int(best_results['precision'] * 100)}只")
    
    # 7. 测试不同阈值下的误报平均涨幅
    logger.info("=== 不同阈值下的误报平均涨幅分析 ===")
    for result in results_list:
        if result['false_positive_count'] > 0:
            logger.info(f"阈值: {result['threshold']:.2f}, 误报平均涨幅: {result['avg_false_positive_gain']:.4f}")
    
    return True

def main():
    """主函数"""
    success = test_model_accuracy()
    
    if success:
        logger.info("=== 模型预测正确率优化测试完成 ===")
        return 0
    else:
        logger.error("=== 模型预测正确率优化测试失败 ===")
        return 1

if __name__ == "__main__":
    sys.exit(main())