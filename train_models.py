#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练三种不同的机器学习模型：Logistic Regression、Random Forest、XGBoost
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pickle
import pandas as pd
import numpy as np
from qmt_strategy.strategy.feature_engineer import FeatureEngineer
from qmt_strategy.strategy.model_train import ModelTrainer
from qmt_strategy.utils.logger import setup_logger
import qmt_strategy.config as config

logger = setup_logger()

def load_and_prepare_data():
    """加载并准备训练数据"""
    logger.info("加载并准备训练数据")
    
    # 加载生成的模拟数据
    data_file = "data/raw_data/history/history_20251101_20251231.pkl"
    if not os.path.exists(data_file):
        logger.error(f"数据文件不存在：{data_file}")
        return None
    
    with open(data_file, 'rb') as f:
        historical_data = pickle.load(f)
    
    logger.info(f"加载了{len(historical_data)}只股票的数据")
    
    # 使用特征工程模块处理数据
    feature_engineer = FeatureEngineer()
    all_features = []
    
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
    logger.info(f"生成了{len(features_df)}条特征数据")
    
    return features_df

def train_model(model_type):
    """训练指定类型的模型"""
    logger.info(f"=== 开始训练{model_type}模型 ===")
    
    # 保存原始配置
    original_model_type = config.MODEL_TYPE
    
    try:
        # 修改配置
        config.MODEL_TYPE = model_type
        
        # 加载数据
        features_df = load_and_prepare_data()
        if features_df is None:
            return False
        
        # 初始化模型训练器
        model_trainer = ModelTrainer()
        
        # 训练模型
        model = model_trainer.train(features_df)
        
        if model is not None:
            logger.info(f"✓ {model_type}模型训练成功")
        else:
            logger.error(f"✗ {model_type}模型训练失败")
            return False
        
    except Exception as e:
        logger.error(f"训练{model_type}模型时出错：{e}")
        return False
    finally:
        # 恢复原始配置
        config.MODEL_TYPE = original_model_type
    
    return True

def main():
    """主函数：训练三种模型"""
    logger.info("=== 开始训练三种机器学习模型 ===")
    
    # 定义要训练的模型类型
    model_types = ["logistic", "random_forest", "xgboost"]
    
    # 依次训练每种模型
    results = {}
    for model_type in model_types:
        success = train_model(model_type)
        results[model_type] = success
    
    # 输出训练结果
    logger.info("=== 模型训练结果 ===")
    for model_type, success in results.items():
        status = "成功" if success else "失败"
        logger.info(f"{model_type}: {status}")
    
    # 检查是否所有模型都训练成功
    all_success = all(results.values())
    if all_success:
        logger.info("✓ 所有模型训练成功！")
    else:
        logger.warning("✗ 部分模型训练失败")
    
    return all_success

if __name__ == "__main__":
    main()
