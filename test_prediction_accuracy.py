#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试优化后的模型预测正确率
"""

import sys
import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# 添加项目路径到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from qmt_strategy.strategy.feature_engineer import FeatureEngineer
from qmt_strategy.strategy.model_train import ModelTrainer

# 设置日志
from qmt_strategy.utils.logger import setup_logger
logger = setup_logger()

def load_test_data():
    """加载测试数据"""
    logger.info("加载测试数据")
    
    # 加载之前生成的模拟数据
    data_file = "data/raw_data/history/history_20251101_20251231.pkl"
    if not os.path.exists(data_file):
        logger.error(f"测试数据文件不存在：{data_file}")
        return None
    
    import pickle
    with open(data_file, 'rb') as f:
        historical_data = pickle.load(f)
    
    logger.info(f"加载了{len(historical_data)}只股票的测试数据")
    return historical_data

def prepare_test_data(historical_data):
    """准备测试数据"""
    logger.info("准备测试数据")
    
    test_data_list = []
    
    for stock, data in historical_data.items():
        try:
            # 转换数据格式
            df = pd.DataFrame({
                'time': pd.to_datetime(data['time']),
                'open': data['open'],
                'close': data['close'],
                'high': data['high'],
                'low': data['low'],
                'volume': data['volume'],
                'stock_code': stock
            })
            
            # 计算涨跌幅
            df['return'] = (df['close'] - df['open']) / df['open']
            df['label'] = (df['return'] >= 0.095).astype(int)  # 当日涨幅>=9.5%标记为涨停
            
            # 添加必要的特征列（模拟集合竞价数据）
            # 注意：这里使用收盘价作为模拟的竞价数据，实际应使用真实竞价数据
            df['bid_price_915'] = df['open'] * (1 + np.random.randn(len(df)) * 0.01)
            df['bid_price_920'] = df['open'] * (1 + np.random.randn(len(df)) * 0.005)
            df['bid_price_925'] = df['open'] * (1 + np.random.randn(len(df)) * 0.002)
            df['bid_volume_915'] = np.random.randint(1000, 10000, len(df))
            df['bid_volume_920'] = np.random.randint(1000, 10000, len(df))
            df['bid_volume_925'] = np.random.randint(1000, 10000, len(df))
            df['prev_close'] = df['close'].shift(1)
            df['prev_close'].fillna(df['open'], inplace=True)
            
            test_data_list.append(df)
        except Exception as e:
            logger.error(f"处理股票{stock}时出错：{e}")
            continue
    
    # 合并所有测试数据
    if test_data_list:
        test_data = pd.concat(test_data_list, ignore_index=True)
        return test_data
    else:
        logger.error("没有生成有效的测试数据")
        return None

def test_model_accuracy():
    """测试模型预测正确率"""
    logger.info("=== 开始测试模型预测正确率 ===")
    
    # 1. 加载测试数据
    historical_data = load_test_data()
    if historical_data is None:
        return False
    
    # 2. 准备测试数据
    test_data = prepare_test_data(historical_data)
    if test_data is None:
        return False
    
    # 3. 初始化模型和特征工程师
    logger.info("初始化模型和特征工程师")
    feature_engineer = FeatureEngineer()
    model_trainer = ModelTrainer()
    
    # 4. 加载预训练模型
    logger.info("加载预训练模型")
    if not model_trainer.load_model():
        logger.error("无法加载模型，测试失败")
        return False
    
    # 5. 提取特征
    logger.info("提取测试数据特征")
    features = feature_engineer.extract_features(test_data)
    if features is None or features.empty:
        logger.error("特征提取失败，测试数据为空")
        return False
    
    # 6. 分离特征和标签
    if 'label' not in features.columns:
        logger.error("测试数据中没有找到标签列")
        return False
    
    X_test = features.drop('label', axis=1)
    y_test = features['label']
    
    logger.info(f"测试数据样本数量：{len(X_test)}")
    logger.info(f"涨停样本数量：{sum(y_test)}")
    logger.info(f"非涨停样本数量：{len(y_test) - sum(y_test)}")
    
    # 7. 模型预测
    logger.info("进行模型预测")
    predictions = model_trainer.predict(X_test)
    if predictions is None:
        logger.error("模型预测失败")
        return False
    
    y_pred = predictions['predictions']
    y_prob = predictions['probabilities']
    
    # 8. 计算预测指标
    logger.info("计算预测指标")
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_prob)
    }
    
    # 9. 输出预测结果
    logger.info("=== 模型预测正确率测试结果 ===")
    logger.info(f"准确率 (Accuracy): {metrics['accuracy']:.4f}")
    logger.info(f"精确率 (Precision): {metrics['precision']:.4f}")
    logger.info(f"召回率 (Recall): {metrics['recall']:.4f}")
    logger.info(f"F1分数 (F1 Score): {metrics['f1_score']:.4f}")
    logger.info(f"AUC-ROC分数: {metrics['roc_auc']:.4f}")
    
    # 10. 保存测试结果
    result_file = f"test_prediction_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
    result_df = pd.DataFrame([metrics])
    result_df.to_csv(result_file, index=False)
    logger.info(f"测试结果保存到：{result_file}")
    
    # 11. 分析预测结果分布
    logger.info("=== 预测结果分布 ===")
    pred_counts = pd.Series(y_pred).value_counts()
    logger.info(f"预测为涨停的数量：{pred_counts.get(1, 0)}")
    logger.info(f"预测为非涨停的数量：{pred_counts.get(0, 0)}")
    
    # 12. 分析概率分布
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