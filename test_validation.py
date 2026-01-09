#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简化的验证脚本，用于测试模型预测和报告生成功能
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.metrics import precision_recall_curve

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

class SimplifiedValidator:
    """简化的模型验证类"""
    
    def __init__(self):
        """初始化"""
        logger.info("初始化简化验证器")
        
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
    
    def test_full_validation(self):
        """测试完整验证流程"""
        logger.info("开始测试完整验证流程")
        
        # 1. 获取或生成数据
        logger.info("1. 获取或生成数据")
        data = self.data_fetcher.get_historical_data_dataframe(
            start_date=self.start_date,
            end_date=self.end_date,
            market="ALL"
        )
        
        if data.empty:
            logger.warning("未获取到真实数据，使用模拟数据")
            # 使用模拟数据生成函数
            from qmt_strategy.utils.simulate_data import generate_stock_list, generate_historical_data
            stock_list = generate_stock_list(n=100)
            data_dict = generate_historical_data(stock_list, self.start_date, self.end_date)
            
            # 转换为DataFrame
            all_data = []
            for stock_code, df in data_dict.items():
                if not df.empty:
                    df = df.copy()
                    df['stock_code'] = stock_code
                    all_data.append(df)
            
            if all_data:
                data = pd.concat(all_data, ignore_index=True)
            else:
                logger.error("无法生成模拟数据")
                return
        
        logger.info(f"获取到数据，共{len(data)}条记录")
        
        # 2. 处理数据，生成特征
        logger.info("2. 处理数据，生成特征")
        processed_data = self.data_processor.process(data)
        if processed_data is None:
            logger.error("数据预处理失败")
            return
        
        features = self.feature_engineer.extract_features(processed_data)
        if features is None or features.empty:
            logger.error("特征提取失败")
            return
        
        logger.info(f"特征生成完成，共{len(features)}条记录")
        
        # 3. 模型预测
        logger.info("3. 模型预测")
        if 'label' in features.columns:
            X = features.drop('label', axis=1)
            y_true = features['label']
        else:
            logger.error("特征数据中没有label列")
            return
        
        # 确保所有特征都是数值类型
        X = X.select_dtypes(include=[np.number])
        
        # 进行预测
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
        
        # 4. 计算评估指标
        logger.info("4. 计算评估指标")
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred),
            'roc_auc': roc_auc_score(y_true, y_prob),
            'confusion_matrix': confusion_matrix(y_true, y_pred)
        }
        
        logger.info(f"评估指标：{metrics}")
        
        # 5. 生成简单报告
        logger.info("5. 生成验证报告")
        report_content = f"""
# 股票涨停预测模型2025年11-12月验证报告

## 1. 验证概述
- 验证时间范围：2025年11月1日至2025年12月31日
- 验证模型：{config.MODEL_TYPE}
- 数据来源：模拟数据

## 2. 模型性能指标
- 准确率：{metrics['accuracy']:.4f}
- 精确率：{metrics['precision']:.4f}
- 召回率：{metrics['recall']:.4f}
- F1分数：{metrics['f1_score']:.4f}
- AUC-ROC值：{metrics['roc_auc']:.4f}

## 3. 混淆矩阵
```
{metrics['confusion_matrix']}
```

## 4. 预测分布
- 总预测样本数：{len(y_pred)}
- 预测为涨停的样本数：{sum(y_pred)}
- 实际涨停的样本数：{sum(y_true)}
- 预测准确率：{metrics['accuracy']:.2%}
"""
        
        # 保存报告
        report_dir = os.path.join('reports', f"model_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(report_dir, exist_ok=True)
        report_path = os.path.join(report_dir, "validation_report.txt")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"验证报告生成完成，保存路径：{report_path}")
        return report_path

if __name__ == "__main__":
    validator = SimplifiedValidator()
    report_path = validator.test_full_validation()
    if report_path:
        print(f"\n验证报告已生成：{report_path}")
    else:
        print("\n验证失败")
