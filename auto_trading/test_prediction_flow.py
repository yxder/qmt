#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试预测流程的完整脚本
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime
from src.data_acquisition.qmt_bid_data import QMTBidDataFetcher
from src.model_inference.prediction_service import PredictionService
from src.scheduler.scheduler_service import SchedulerService
from qmt_strategy.utils.logger import setup_logger

logger = setup_logger()


def test_prediction_flow():
    """
    测试完整的预测流程
    """
    logger.info("=== 开始测试预测流程 ===")
    
    try:
        # 1. 初始化预测服务
        logger.info("1. 初始化预测服务")
        prediction_service = PredictionService()
        
        # 获取模型信息
        model_info = prediction_service.get_model_info()
        logger.info(f"模型信息: {model_info}")
        
        # 2. 生成模拟竞价数据
        logger.info("2. 生成模拟竞价数据")
        mock_bid_data = generate_mock_bid_data()
        logger.info(f"生成了{len(mock_bid_data)}条模拟竞价数据")
        
        # 3. 进行预测
        logger.info("3. 进行预测")
        prediction_results = prediction_service.predict(mock_bid_data)
        logger.info(f"预测结果: {len(prediction_results)}条")
        
        # 4. 过滤预测结果
        logger.info("4. 过滤预测结果")
        filtered_results = prediction_service.filter_prediction_results(prediction_results, threshold=0.5)
        logger.info(f"过滤后结果: {len(filtered_results)}条")
        
        # 5. 保存预测结果
        logger.info("5. 保存预测结果")
        csv_path = prediction_service.save_prediction_results(prediction_results, format='csv')
        json_path = prediction_service.save_prediction_results(prediction_results, format='json')
        
        if not filtered_results.empty:
            filtered_csv_path = prediction_service.save_prediction_results(
                filtered_results, 
                file_path=csv_path.replace('.csv', '_filtered.csv') if csv_path else None,
                format='csv'
            )
            filtered_json_path = prediction_service.save_prediction_results(
                filtered_results, 
                file_path=json_path.replace('.json', '_filtered.json') if json_path else None,
                format='json'
            )
        
        logger.info("=== 预测流程测试完成 ===")
        logger.info(f"原始预测结果: {csv_path}, {json_path}")
        if not filtered_results.empty:
            logger.info(f"过滤后结果: {filtered_csv_path}, {filtered_json_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"测试预测流程时发生错误: {e}")
        import traceback
        logger.error(f"错误堆栈: {traceback.format_exc()}")
        return False


def generate_mock_bid_data():
    """
    生成模拟竞价数据
    
    Returns:
        pd.DataFrame: 模拟竞价数据
    """
    # 模拟股票代码
    stock_codes = [f"0000{i:02d}.SZ" for i in range(1, 11)] + [f"6000{i:02d}.SH" for i in range(1, 11)]
    
    # 今天的日期
    today = datetime.now().strftime("%Y%m%d")
    
    # 生成模拟数据
    data = []
    for stock_code in stock_codes:
        # 生成随机竞价数据
        bid_open = np.random.uniform(5, 50)
        bid_high = bid_open * np.random.uniform(1.0, 1.05)
        bid_low = bid_open * np.random.uniform(0.95, 1.0)
        bid_price = np.random.uniform(bid_low, bid_high)
        bid_volume = np.random.randint(10000, 1000000)
        bid_amount = bid_price * bid_volume
        
        data.append({
            'stock_code': stock_code,
            'date': today,
            'bid_open': bid_open,
            'bid_high': bid_high,
            'bid_low': bid_low,
            'bid_price': bid_price,
            'bid_volume': bid_volume,
            'bid_amount': bid_amount,
            'datetime': datetime.now()
        })
    
    return pd.DataFrame(data)


def test_scheduler_service():
    """
    测试调度服务
    """
    logger.info("=== 开始测试调度服务 ===")
    
    try:
        scheduler_service = SchedulerService()
        
        # 立即执行一次预测任务（使用模拟数据）
        logger.info("立即执行一次预测任务")
        scheduler_service.run_once()
        
        logger.info("=== 调度服务测试完成 ===")
        return True
        
    except Exception as e:
        logger.error(f"测试调度服务时发生错误: {e}")
        import traceback
        logger.error(f"错误堆栈: {traceback.format_exc()}")
        return False


if __name__ == "__main__":
    # 测试预测流程
    logger.info("测试预测流程...")
    prediction_success = test_prediction_flow()
    
    # 测试调度服务
    logger.info("\n测试调度服务...")
    scheduler_success = test_scheduler_service()
    
    # 输出测试结果
    logger.info("\n=== 测试结果汇总 ===")
    logger.info(f"预测流程测试: {'成功' if prediction_success else '失败'}")
    logger.info(f"调度服务测试: {'成功' if scheduler_success else '失败'}")
    
    if prediction_success and scheduler_success:
        logger.info("所有测试都成功了！")
    else:
        logger.error("部分测试失败了！")
