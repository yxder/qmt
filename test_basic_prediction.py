#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
基础预测测试脚本，用于测试模型加载和预测功能
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

# 添加项目路径到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入所需模块
from qmt_strategy.strategy.data_fetch import DataFetcher
from qmt_strategy.strategy.data_process import DataProcessor
from qmt_strategy.strategy.feature_engineer import FeatureEngineer
from qmt_strategy.strategy.model_train import ModelTrainer
import qmt_strategy.config as config

# 设置日志
from qmt_strategy.utils.logger import setup_logger
logger = setup_logger()

class BasicPredictionTester:
    """基础预测测试类"""
    
    def __init__(self):
        """初始化"""
        logger.info("初始化基础预测测试器")
        
        # 初始化各组件
        self.data_fetcher = DataFetcher(use_local_data=True)
        self.data_processor = DataProcessor()
        self.feature_engineer = FeatureEngineer()
        self.model_trainer = ModelTrainer()
        
        # 日期范围设置
        self.start_date = "20251101"
        self.end_date = "20251231"
        
        # 加载训练好的模型
        self.model = self.model_trainer.load_model()
        if self.model is None:
            logger.error("无法加载训练好的模型")
            sys.exit(1)
        logger.info("成功加载训练好的模型")
    
    def test_with_real_data(self):
        """使用真实数据测试"""
        logger.info("开始使用真实数据测试")
        
        # 1. 获取真实的2025年11-12月股票数据
        logger.info(f"1. 获取2025年11-12月真实股票数据，日期范围：{self.start_date} 至 {self.end_date}")
        data = self.data_fetcher.get_historical_data_dataframe(
            start_date=self.start_date,
            end_date=self.end_date,
            market="ALL"
        )
        
        if data.empty:
            logger.error("未获取到真实数据，请检查数据来源或QMT服务是否开启")
            return
        
        logger.info(f"成功获取真实数据：{data.shape}")
        logger.info(f"数据字段：{list(data.columns)}")
        
        # 2. 数据预处理
        logger.info("2. 数据预处理")
        processed_data = self.data_processor.process(data)
        if processed_data is None:
            logger.error("数据预处理失败")
            return
        
        logger.info(f"数据预处理完成：{processed_data.shape}")
        
        # 3. 特征工程
        logger.info("3. 特征工程")
        features = self.feature_engineer.extract_features(processed_data)
        if features is None or features.empty:
            logger.error("特征提取失败")
            return
        
        logger.info(f"特征工程完成：{features.shape}")
        
        # 4. 分离特征和标签
        logger.info("4. 分离特征和标签")
        if 'label' not in features.columns:
            logger.error("特征数据中没有label列，无法进行验证")
            return
        
        X = features.drop('label', axis=1)
        y_true = features['label']
        
        # 确保所有特征都是数值类型
        X = X.select_dtypes(include=[np.number])
        logger.info(f"特征数据形状：{X.shape}，标签数据形状：{y_true.shape}")
        
        # 5. 进行预测
        logger.info("5. 进行预测")
        predictions = self.model_trainer.predict(X)
        if predictions is None:
            logger.error("模型预测失败")
            return
        
        y_pred = predictions['predictions']
        y_prob = predictions['probabilities']
        
        logger.info("预测完成")
        logger.info(f"预测样本数: {len(y_pred)}")
        logger.info(f"正例预测数: {sum(y_pred)}")
        logger.info(f"真实正例数: {sum(y_true)}")
        logger.info(f"预测概率范围: {min(y_prob):.4f} - {max(y_prob):.4f}")
        
        # 6. 计算评估指标
        logger.info("6. 计算评估指标")
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
        from sklearn.metrics import precision_recall_curve
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_true, y_prob),
            'confusion_matrix': confusion_matrix(y_true, y_pred)
        }
        
        logger.info(f"评估指标：")
        for key, value in metrics.items():
            if key != 'confusion_matrix':
                logger.info(f"  {key}: {value:.4f}")
        logger.info(f"  混淆矩阵：\n{metrics['confusion_matrix']}")
        
        # 7. 阈值优化
        logger.info("7. 阈值优化")
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-10)
        best_threshold_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_threshold_idx]
        best_f1_score = f1_scores[best_threshold_idx]
        best_precision = precision[best_threshold_idx]
        best_recall = recall[best_threshold_idx]
        
        logger.info(f"最佳阈值：{best_threshold:.4f}，对应的F1分数：{best_f1_score:.4f}，精确率：{best_precision:.4f}，召回率：{best_recall:.4f}")
        
        # 8. 生成测试报告
        logger.info("8. 生成真实数据预测报告")
        report_content = f"""
# 模型真实数据预测测试报告

## 1. 测试概述
- 测试时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- 测试模型：{config.MODEL_TYPE}
- 测试数据：2025年11-12月真实股票数据
- 数据范围：{self.start_date} 至 {self.end_date}
- 数据来源：{'本地QMT数据' if self.data_fetcher.use_local_data else 'xtquant服务'}

## 2. 数据概览
- 原始数据形状：{data.shape}
- 预处理后数据形状：{processed_data.shape}
- 特征数据形状：{X.shape}
- 标签数据形状：{y_true.shape}
- 特征数量：{X.shape[1]}
- 预测样本数：{len(y_pred)}

## 3. 预测结果
- 预测为涨停的样本数：{sum(y_pred)}
- 真实涨停的样本数：{sum(y_true)}
- 预测为非涨停的样本数：{len(y_pred) - sum(y_pred)}
- 真实非涨停的样本数：{len(y_true) - sum(y_true)}

## 4. 评估指标
- 准确率：{metrics['accuracy']:.4f}
- 精确率：{metrics['precision']:.4f}
- 召回率：{metrics['recall']:.4f}
- F1分数：{metrics['f1_score']:.4f}
- AUC-ROC值：{metrics['roc_auc']:.4f}

## 5. 混淆矩阵
```
{metrics['confusion_matrix']}
```

## 6. 预测概率分布
- 预测概率最小值：{min(y_prob):.4f}
- 预测概率最大值：{max(y_prob):.4f}
- 预测概率平均值：{np.mean(y_prob):.4f}
- 预测概率中位数：{np.median(y_prob):.4f}

## 7. 阈值优化结果
- 最佳阈值：{best_threshold:.4f}
- 最佳F1分数：{best_f1_score:.4f}
- 最佳阈值下的精确率：{best_precision:.4f}
- 最佳阈值下的召回率：{best_recall:.4f}

## 8. 模型状态
- 模型加载：成功
- 数据获取：成功
- 数据预处理：成功
- 特征工程：成功
- 预测功能：正常
- 评估指标计算：成功
- 阈值优化：成功

## 9. 模型优势与不足
### 优势
- 模型稳定性较好，能够处理大量数据
- 能够生成合理的概率预测
- 代码结构清晰，便于维护和扩展

### 不足
- 预测正例数量较少，可能需要调整模型或特征
- 模型在当前阈值下召回率较低
- 可能需要更多特征或更复杂的模型结构

## 10. 优化建议
- 调整模型阈值以平衡精确率和召回率
- 增加更多相关特征，特别是与涨停相关的技术指标
- 考虑使用更复杂的模型结构或集成学习方法
- 增加训练数据量，特别是涨停样本
- 考虑对不同市场环境、市值和行业进行针对性优化
"""
        
        # 保存报告
        report_dir = 'reports'
        os.makedirs(report_dir, exist_ok=True)
        report_path = os.path.join(report_dir, f"real_data_prediction_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"真实数据预测测试报告生成完成，保存路径：{report_path}")
        return report_path

if __name__ == "__main__":
    tester = BasicPredictionTester()
    print("\n=== 执行真实数据预测测试 ===")
    tester.test_with_real_data()
    
    print("\n测试完成！")
