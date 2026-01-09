#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
定时任务调度服务，用于每天9:26分自动启动预测流程
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from datetime import datetime
import traceback
from qmt_strategy.utils.logger import setup_logger

logger = setup_logger()


class SchedulerService:
    """
    定时任务调度服务，用于每天9:26分自动启动预测流程
    """
    
    def __init__(self):
        """
        初始化调度服务
        """
        logger.info("初始化定时任务调度服务")
        self.scheduler = BlockingScheduler()
        self._setup_jobs()
    
    def _setup_jobs(self):
        """
        设置定时任务
        """
        # 添加每天9:26分执行的预测任务
        logger.info("添加每天9:26分执行的预测任务")
        self.scheduler.add_job(
            func=self._run_prediction,
            trigger=CronTrigger(hour=9, minute=26, second=0),
            id='daily_prediction',
            name='每日股票涨停预测',
            replace_existing=True
        )
        
        # 可以添加一个测试任务，每分钟执行一次，用于调试
        # self.scheduler.add_job(
        #     func=self._run_prediction,
        #     trigger=CronTrigger(minute='*'),
        #     id='test_prediction',
        #     name='测试预测任务',
        #     replace_existing=True
        # )
    
    def _run_prediction(self):
        """
        执行预测流程
        """
        logger.info("=== 开始执行每日股票涨停预测任务 ===")
        
        try:
            # 动态导入，避免循环依赖
            from src.data_acquisition.qmt_bid_data import QMTBidDataFetcher
            from src.model_inference.prediction_service import PredictionService
            
            # 记录开始时间
            start_time = datetime.now()
            
            # 1. 获取竞价数据
            logger.info("1. 获取当天的股票竞价数据")
            data_fetcher = QMTBidDataFetcher()
            bid_data = data_fetcher.get_today_bid_data()
            
            if bid_data.empty:
                logger.error("未获取到竞价数据，预测任务终止")
                return
            
            logger.info(f"获取到{len(bid_data)}条竞价数据")
            
            # 2. 进行模型预测
            logger.info("2. 使用训练好的模型进行预测")
            prediction_service = PredictionService()
            prediction_results = prediction_service.predict(bid_data)
            
            if prediction_results.empty:
                logger.error("预测结果为空，预测任务终止")
                return
            
            logger.info(f"预测完成，共预测{len(prediction_results)}条数据")
            
            # 3. 过滤预测结果
            logger.info("3. 过滤预测结果")
            filtered_results = prediction_service.filter_prediction_results(prediction_results, threshold=0.5)
            logger.info(f"过滤后剩余{len(filtered_results)}条结果")
            
            # 4. 保存预测结果
            logger.info("4. 保存预测结果")
            
            # 保存原始预测结果
            csv_path = prediction_service.save_prediction_results(prediction_results, format='csv')
            json_path = prediction_service.save_prediction_results(prediction_results, format='json')
            
            # 保存过滤后的预测结果
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
            
            # 记录结束时间
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            logger.info(f"=== 预测任务执行完成 ===")
            logger.info(f"开始时间: {start_time}")
            logger.info(f"结束时间: {end_time}")
            logger.info(f"执行时长: {duration:.2f}秒")
            logger.info(f"原始预测结果: {csv_path}, {json_path}")
            if not filtered_results.empty:
                logger.info(f"过滤后结果: {filtered_csv_path}, {filtered_json_path}")
            
        except Exception as e:
            logger.error(f"执行预测任务时发生错误: {e}")
            logger.error(f"错误堆栈: {traceback.format_exc()}")
    
    def run(self):
        """
        启动调度服务
        """
        try:
            logger.info("启动定时任务调度服务")
            logger.info(f"当前时间: {datetime.now()}")
            logger.info("调度服务将在每天9:26分执行预测任务")
            
            # 启动调度器
            self.scheduler.start()
        except KeyboardInterrupt:
            logger.info("调度服务被用户中断")
        except Exception as e:
            logger.error(f"调度服务启动失败: {e}")
            logger.error(f"错误堆栈: {traceback.format_exc()}")
        finally:
            # 关闭调度器
            if self.scheduler.running:
                self.scheduler.shutdown()
            logger.info("调度服务已关闭")
    
    def run_once(self):
        """
        立即执行一次预测任务（用于测试）
        """
        logger.info("立即执行一次预测任务")
        self._run_prediction()


if __name__ == "__main__":
    # 测试代码
    scheduler_service = SchedulerService()
    
    # 可以选择立即执行一次测试，或者启动调度服务
    import argparse
    
    parser = argparse.ArgumentParser(description='股票涨停预测定时调度服务')
    parser.add_argument('--test', action='store_true', help='立即执行一次预测任务')
    
    args = parser.parse_args()
    
    if args.test:
        # 立即执行一次预测任务
        scheduler_service.run_once()
    else:
        # 启动调度服务
        scheduler_service.run()
