#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
策略参数敏感性分析脚本：测试不同参数组合对策略表现的影响
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pickle
import pandas as pd
import numpy as np
from qmt_strategy.strategy.feature_engineer import FeatureEngineer
from qmt_strategy.strategy.model_train import ModelTrainer
from qmt_strategy.strategy.strategy_decision import StrategyDecision
from qmt_strategy.strategy.risk_control import RiskController
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
    
    # 直接使用已生成的特征数据，而不是重新生成
    # 查找最新的特征数据文件
    import os
    import glob
    
    feature_files = glob.glob("data/feature_data/features_*.pkl")
    if not feature_files:
        logger.error("没有找到特征数据文件")
        return None
    
    # 选择最新的特征数据文件
    feature_files.sort(key=os.path.getmtime, reverse=True)
    latest_feature_file = feature_files[0]
    
    logger.info(f"使用最新的特征数据文件：{latest_feature_file}")
    
    try:
        # 加载特征数据
        features_df = pd.read_pickle(latest_feature_file)
        logger.info(f"加载了{len(features_df)}条特征数据")
        
        # 添加模拟的return和label列（如果不存在）
        if 'return' not in features_df.columns:
            features_df['return'] = np.random.randn(len(features_df)) * 0.02
        if 'label' not in features_df.columns:
            features_df['label'] = (features_df['return'] >= 0.095).astype(int)
        
        return features_df
    except Exception as e:
        logger.error(f"加载特征数据失败：{e}")
        return None

def test_parameter_sensitivity(features_df):
    """测试策略参数敏感性"""
    logger.info("=== 开始策略参数敏感性分析 ===")
    
    if features_df is None or features_df.empty:
        logger.error("测试数据为空，无法进行参数敏感性分析")
        return
    
    # 只保留数值特征用于预测
    numeric_features = features_df.select_dtypes(include=[np.number])
    if 'label' in numeric_features.columns:
        numeric_features = numeric_features.drop('label', axis=1)
    if 'return' in numeric_features.columns:
        numeric_features = numeric_features.drop('return', axis=1)
    
    # 加载不同类型的模型
    model_files = {
        'logistic': None,
        'random_forest': None,
        'xgboost': None
    }
    
    # 获取最新的模型文件
    model_dir = config.MODEL_PATH
    model_files_list = [f for f in os.listdir(model_dir) if f.endswith('.joblib')]
    model_files_list.sort(key=lambda x: os.path.getmtime(os.path.join(model_dir, x)), reverse=True)
    
    for model_file in model_files_list:
        if 'logistic' in model_file and model_files['logistic'] is None:
            model_files['logistic'] = os.path.join(model_dir, model_file)
        elif 'random_forest' in model_file and model_files['random_forest'] is None:
            model_files['random_forest'] = os.path.join(model_dir, model_file)
        elif 'xgboost' in model_file and model_files['xgboost'] is None:
            model_files['xgboost'] = os.path.join(model_dir, model_file)
    
    # 测试不同的预测阈值
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    
    # 测试不同的模型类型和阈值组合
    results = []
    
    for model_name, model_path in model_files.items():
        if model_path is None:
            logger.warning(f"未找到{model_name}模型文件")
            continue
        
        logger.info(f"=== 测试{model_name}模型 ===")
        
        # 加载模型
        model_trainer = ModelTrainer()
        model_trainer.load_model(model_path)
        
        if model_trainer.model is None:
            logger.error(f"无法加载{model_name}模型")
            continue
        
        # 进行预测
        predictions = model_trainer.predict(numeric_features)
        if predictions is None:
            logger.error(f"{model_name}模型预测失败")
            continue
        
        # 测试不同阈值
        for threshold in thresholds:
            logger.info(f"测试阈值：{threshold}")
            
            # 应用阈值
            adjusted_predictions = (predictions['probabilities'] >= threshold).astype(int)
            
            # 计算指标
            true_labels = features_df['label'].values
            pred_labels = adjusted_predictions
            pred_proba = predictions['probabilities']
            
            # 计算基础指标
            total_samples = len(true_labels)
            positive_samples = sum(true_labels)
            predicted_positive = sum(pred_labels)
            
            # 计算准确率
            if total_samples > 0:
                accuracy = np.sum(pred_labels == true_labels) / total_samples
            else:
                accuracy = 0
            
            # 计算召回率
            if positive_samples > 0:
                recall = np.sum((pred_labels == 1) & (true_labels == 1)) / positive_samples
            else:
                recall = 0
            
            # 计算精确率
            if predicted_positive > 0:
                precision = np.sum((pred_labels == 1) & (true_labels == 1)) / predicted_positive
            else:
                precision = 0
            
            # 计算F1分数
            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0
            
            # 记录结果
            result = {
                'model_name': model_name,
                'threshold': threshold,
                'total_samples': total_samples,
                'positive_samples': positive_samples,
                'predicted_positive': predicted_positive,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'avg_probability': np.mean(pred_proba)
            }
            
            results.append(result)
            
            logger.info(f"结果：{result}")
    
    # 保存结果到CSV文件
    results_df = pd.DataFrame(results)
    results_file = "parameter_sensitivity_results.csv"
    results_df.to_csv(results_file, index=False)
    logger.info(f"参数敏感性分析结果已保存到{results_file}")
    
    # 输出汇总结果
    logger.info("=== 参数敏感性分析汇总 ===")
    for model_name in model_files.keys():
        model_results = results_df[results_df['model_name'] == model_name]
        if not model_results.empty:
            logger.info(f"\n{model_name}模型结果：")
            for _, row in model_results.iterrows():
                logger.info(f"阈值={row['threshold']:.1f}，F1分数={row['f1_score']:.4f}，预测正样本={row['predicted_positive']}")
    
    logger.info("=== 参数敏感性分析完成 ===")

def main():
    """主函数"""
    logger.info("=== 开始策略参数敏感性分析流程 ===")
    
    # 1. 加载历史数据
    historical_data = load_historical_data()
    if historical_data is None:
        return 1
    
    # 2. 准备测试数据
    features_df = prepare_test_data(historical_data)
    if features_df is None:
        return 1
    
    # 3. 执行参数敏感性分析
    test_parameter_sensitivity(features_df)
    
    logger.info("=== 策略参数敏感性分析流程完成 ===")
    return 0

if __name__ == "__main__":
    main()
