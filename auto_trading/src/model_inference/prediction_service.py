#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型预测服务，用于加载保存的ModelWrapper实例并进行股票涨停预测
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from qmt_strategy.utils.logger import setup_logger

logger = setup_logger()


class PredictionService:
    """
    模型预测服务，用于加载保存的ModelWrapper实例并进行股票涨停预测
    """
    
    def __init__(self, model_path=None):
        """
        初始化预测服务
        
        Args:
            model_path: 模型文件路径，如果为None则使用最新的模型
        """
        logger.info("初始化模型预测服务")
        self.model_wrapper = None
        self.model_path = model_path
        self._load_model()
    
    def _load_model(self):
        """
        加载模型
        """
        try:
            if not self.model_path:
                # 如果没有提供模型路径，使用最新的模型
                model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models")
                # 检查是否有模型文件
                if not os.path.exists(model_dir):
                    logger.warning(f"模型目录不存在: {model_dir}")
                    # 尝试使用父项目的模型目录
                    model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "models")
                    if not os.path.exists(model_dir):
                        logger.error(f"模型目录不存在: {model_dir}")
                        return
                
                # 获取所有模型文件
                model_files = [f for f in os.listdir(model_dir) if f.endswith('.joblib')]
                if not model_files:
                    logger.error(f"模型目录中没有模型文件: {model_dir}")
                    return
                
                # 按修改时间排序，获取最新的模型
                model_files.sort(key=lambda x: os.path.getmtime(os.path.join(model_dir, x)), reverse=True)
                self.model_path = os.path.join(model_dir, model_files[0])
            
            logger.info(f"加载模型: {self.model_path}")
            self.model_wrapper = joblib.load(self.model_path)
            logger.info(f"模型加载成功，模型版本: {getattr(self.model_wrapper, 'version', '未知')}")
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            self.model_wrapper = None
    
    def predict(self, bid_data):
        """
        使用模型进行预测
        
        Args:
            bid_data: 竞价数据DataFrame
            
        Returns:
            pd.DataFrame: 预测结果，包含股票代码、预测概率、预测标签等
        """
        if self.model_wrapper is None:
            logger.error("模型未加载，无法进行预测")
            return pd.DataFrame()
        
        if bid_data.empty:
            logger.warning("竞价数据为空，无法进行预测")
            return pd.DataFrame()
        
        try:
            logger.info(f"开始预测，共{len(bid_data)}条数据")
            
            # 使用模型包装器进行预测
            predictions = self.model_wrapper.predict(bid_data)
            
            # 构建预测结果DataFrame
            result = pd.DataFrame({
                'stock_code': bid_data['stock_code'],
                'date': bid_data['date'],
                'prediction': predictions['predictions'],
                'probability': predictions['probabilities']
            })
            
            # 按照预测概率降序排序
            result = result.sort_values('probability', ascending=False)
            
            # 添加预测时间
            result['predict_time'] = datetime.now().isoformat()
            
            # 添加模型信息
            result['model_version'] = getattr(self.model_wrapper, 'version', '未知')
            result['model_path'] = self.model_path.split(os.sep)[-1] if self.model_path else '未知'
            
            logger.info(f"预测完成，共预测{len(result)}条数据")
            return result
        except Exception as e:
            logger.error(f"预测失败: {e}")
            return pd.DataFrame()
    
    def filter_prediction_results(self, prediction_results, threshold=0.5):
        """
        过滤预测结果，只保留概率大于等于阈值的结果
        
        Args:
            prediction_results: 预测结果DataFrame
            threshold: 概率阈值
            
        Returns:
            pd.DataFrame: 过滤后的预测结果
        """
        if prediction_results.empty:
            logger.warning("预测结果为空，无法过滤")
            return pd.DataFrame()
        
        filtered_results = prediction_results[prediction_results['probability'] >= threshold]
        logger.info(f"过滤前: {len(prediction_results)}条，过滤后: {len(filtered_results)}条，阈值: {threshold}")
        return filtered_results
    
    def save_prediction_results(self, prediction_results, file_path=None, format='csv'):
        """
        保存预测结果
        
        Args:
            prediction_results: 预测结果DataFrame
            file_path: 保存路径，如果为None则使用默认路径
            format: 保存格式，支持'csv'和'json'
            
        Returns:
            str: 实际保存的文件路径
        """
        if prediction_results.empty:
            logger.warning("预测结果为空，不保存")
            return None
        
        try:
            if not file_path:
                # 使用默认路径
                output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "output")
                os.makedirs(output_dir, exist_ok=True)
                
                # 生成文件名
                today = datetime.now().strftime("%Y%m%d")
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                file_name = f"prediction_results_{today}_{timestamp}.{format}"
                file_path = os.path.join(output_dir, file_name)
            
            # 确保目录存在
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            if format == 'csv':
                prediction_results.to_csv(file_path, index=False, encoding='utf-8-sig')
                logger.info(f"预测结果保存为CSV: {file_path}")
            elif format == 'json':
                prediction_results.to_json(file_path, orient='records', force_ascii=False, indent=2)
                logger.info(f"预测结果保存为JSON: {file_path}")
            else:
                logger.error(f"不支持的保存格式: {format}")
                return None
            
            return file_path
        except Exception as e:
            logger.error(f"保存预测结果失败: {e}")
            return None
    
    def get_model_info(self):
        """
        获取模型信息
        
        Returns:
            dict: 模型信息
        """
        if not self.model_wrapper:
            return {
                'loaded': False,
                'model_path': self.model_path,
                'error': '模型未加载'
            }
        
        return {
            'loaded': True,
            'model_path': self.model_path,
            'version': getattr(self.model_wrapper, 'version', '未知'),
            'created_at': getattr(self.model_wrapper, 'created_at', '未知'),
            'config': getattr(self.model_wrapper, 'config', {})
        }


if __name__ == "__main__":
    # 测试代码
    from data_acquisition.qmt_bid_data import QMTBidDataFetcher
    
    # 初始化数据获取器
    data_fetcher = QMTBidDataFetcher()
    
    # 获取当天的竞价数据
    bid_data = data_fetcher.get_today_bid_data()
    
    if not bid_data.empty:
        # 初始化预测服务
        prediction_service = PredictionService()
        
        # 获取模型信息
        model_info = prediction_service.get_model_info()
        print("模型信息:", model_info)
        
        # 进行预测
        prediction_results = prediction_service.predict(bid_data)
        print(f"预测结果: {len(prediction_results)}条")
        print(prediction_results.head())
        
        # 过滤预测结果
        filtered_results = prediction_service.filter_prediction_results(prediction_results, threshold=0.5)
        print(f"过滤后结果: {len(filtered_results)}条")
        print(filtered_results.head())
        
        # 保存预测结果
        if not prediction_results.empty:
            prediction_service.save_prediction_results(prediction_results, format='csv')
            prediction_service.save_prediction_results(prediction_results, format='json')
    else:
        print("未获取到竞价数据，无法进行预测")
