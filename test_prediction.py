#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试模型预测结果脚本：检查模型对回测数据的预测结果
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pickle
import pandas as pd
import numpy as np
from qmt_strategy.strategy.model_train import ModelTrainer
from qmt_strategy.strategy.feature_engineer import FeatureEngineer
from qmt_strategy.utils.logger import setup_logger
import qmt_strategy.config as config

logger = setup_logger()

def load_historical_data():
    """加载历史数据"""
    logger.info("加载历史数据")
    
    data_file = "data/raw_data/history/history_20251101_20251231.pkl"
    if not os.path.exists(data_file):
        logger.error(f"数据文件不存在：{data_file}")
        return None
    
    with open(data_file, 'rb') as f:
        historical_data = pickle.load(f)
    
    logger.info(f"加载了{len(historical_data)}只股票的数据")
    return historical_data

def prepare_test_data(historical_data):
    """准备测试数据"""
    logger.info("准备测试数据")
    
    feature_engineer = FeatureEngineer()
    all_features = []
    
    # 只处理前10只股票，便于快速测试
    stocks = list(historical_data.keys())[:10]
    
    for stock in stocks:
        data = historical_data[stock]
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
            
            # 计算涨跌幅作为标签
            df['return'] = (df['close'] - df['open']) / df['open']
            df['label'] = (df['return'] >= 0.095).astype(int)  # 当日涨幅>=9.5%标记为涨停
            
            # 生成特征
            features = feature_engineer.extract_features(df)
            if features is not None and not features.empty:
                features['stock_code'] = stock
                all_features.append(features)
        except Exception as e:
            logger.error(f"处理股票{stock}时出错：{e}")
            continue
    
    if not all_features:
        logger.error("没有生成任何特征数据")
        return None
    
    # 合并所有股票的特征数据
    features_df = pd.concat(all_features, ignore_index=True)
    logger.info(f"生成了{len(features_df)}条测试特征数据")
    
    return features_df

def test_model_prediction(features_df):
    """测试模型预测结果"""
    logger.info("测试模型预测结果")
    
    if features_df is None or features_df.empty:
        logger.error("测试数据为空，无法进行预测测试")
        return
    
    # 只保留数值特征
    numeric_features = features_df.select_dtypes(include=[np.number])
    if 'label' in numeric_features.columns:
        numeric_features = numeric_features.drop('label', axis=1)
    
    logger.info(f"用于预测的特征数量：{len(numeric_features.columns)}")
    
    # 加载模型
    model_trainer = ModelTrainer()
    model_trainer.load_model()
    
    # 进行预测
    predictions = model_trainer.predict(numeric_features)
    
    if predictions is not None:
        logger.info("=== 模型预测结果 ===")
        logger.info(f"预测数量：{len(predictions['predictions'])}")
        logger.info(f"预测值分布：{np.unique(predictions['predictions'], return_counts=True)}")
        logger.info(f"概率分布：最小值={np.min(predictions['probabilities'])}, 最大值={np.max(predictions['probabilities'])}, 平均值={np.mean(predictions['probabilities'])}")
        
        # 打印部分预测结果
        for i in range(min(5, len(predictions['predictions']))):
            logger.info(f"样本{i+1}: 预测={predictions['predictions'][i]}, 概率={predictions['probabilities'][i]:.4f}")
    else:
        logger.error("模型预测失败")

def main():
    """主函数"""
    logger.info("=== 开始模型预测测试 ===")
    
    # 1. 加载历史数据
    historical_data = load_historical_data()
    if historical_data is None:
        return 1
    
    # 2. 准备测试数据
    features_df = prepare_test_data(historical_data)
    if features_df is None:
        return 1
    
    # 3. 测试模型预测
    test_model_prediction(features_df)
    
    logger.info("=== 模型预测测试完成 ===")
    return 0

if __name__ == "__main__":
    sys.exit(main())
