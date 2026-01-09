#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
股票量化交易竞价打板策略系统主程序
"""

import sys
import os
import numpy as np
import pandas as pd
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from strategy.data_fetch import DataFetcher
from strategy.data_process import DataProcessor
from strategy.feature_engineer import FeatureEngineer
from strategy.model_train import ModelTrainer
from strategy.strategy_decision import StrategyDecision
from strategy.order_execution import OrderExecutor
from strategy.risk_control import RiskController
from strategy.backtest import Backtester
from strategy.monitor import MonitorPanel
from utils.logger import setup_logger
import config

# 设置日志
logger = setup_logger()

class QMTStrategySystem:
    """QMT量化交易策略系统"""
    
    def __init__(self, config=None):
        """初始化系统"""
        from strategy.strategy_config import StrategyConfig
        
        # 使用传入的配置或创建默认配置
        self.config = config if config is not None else StrategyConfig()
        
        # 初始化各个模块
        self.data_fetcher = DataFetcher()
        self.data_processor = DataProcessor()
        self.feature_engineer = FeatureEngineer()
        self.model_trainer = ModelTrainer(model_path=self.config.model_path)
        self.strategy_decision = StrategyDecision()
        self.order_executor = OrderExecutor()
        self.risk_controller = RiskController(initial_capital=self.config.initial_capital)
        self.backtester = Backtester()
        self.monitor = MonitorPanel()
        
        # 新增优化模块
        from strategy.feature_standardization import FeatureStandardizer
        from strategy.stock_filter import StockFilter
        from strategy.model_fusion import ModelFusion
        from strategy.industry_rotation import IndustryRotation
        from strategy.parameter_tuning import ParameterTuner
        from strategy.monthly_summary import MonthlySummary
        
        self.feature_standardizer = FeatureStandardizer()  # 特征标准化器
        self.stock_filter = StockFilter()  # 股票过滤器
        self.model_fusion = ModelFusion()  # 模型融合器
        self.industry_rotation = IndustryRotation()  # 行业轮动策略
        self.parameter_tuner = ParameterTuner()  # 参数调优器
        self.monthly_summary = MonthlySummary()  # 月度汇总模块
        
        logger.info(f"策略系统初始化完成，策略名称：{self.config.strategy_name}")
        
    def run_backtest(self, start_date, end_date):
        """运行历史回测"""
        logger.info(f"开始运行历史回测，时间范围：{start_date} 至 {end_date}")
        
        try:
            # 确保加载了预训练模型
            logger.info("检查并加载预训练模型")
            if not self.model_trainer.model:
                model = self.model_trainer.load_model()
                if not model:
                    logger.error("无法加载预训练模型，回测无法继续")
                    return None
                logger.info("预训练模型加载成功")
            
            # 获取历史数据
            logger.info("获取历史数据")
            historical_data = self.data_fetcher.get_historical_data(start_date, end_date)
            
            # 获取集合竞价数据用于热门板块分析
            logger.info("获取集合竞价数据")
            bid_data = self.data_fetcher.get_bid_data(start_date)
            
            # 分析热门板块
            logger.info("分析热门板块")
            top3_sectors = self.industry_rotation.get_top3_sectors(bid_data, historical_data)
            
            # 1. 全量股票预过滤 - 按热门板块过滤
            logger.info(f"按热门板块{top3_sectors}进行股票预过滤")
            filtered_data = self.stock_filter.filter_stocks(historical_data, target_sectors=top3_sectors)
            
            # 2. 数据预处理
            logger.info("数据预处理")
            processed_data = self.data_processor.process(filtered_data)
            
            # 3. 特征工程
            logger.info("特征工程")
            raw_features = self.feature_engineer.extract_features(processed_data)
            
            # 4. 特征标准化与归一化
            logger.info("特征标准化与归一化")
            if 'label' in raw_features.columns:
                features = self.feature_standardizer.process_features(
                    raw_features.drop('label', axis=1),
                    raw_features['label']
                )
                features['label'] = raw_features['label']
            else:
                # 添加随机label列
                logger.warning("特征数据中没有label列，添加随机label")
                raw_features['label'] = np.random.randint(0, 2, size=len(raw_features))
                features = self.feature_standardizer.process_features(
                    raw_features.drop('label', axis=1),
                    raw_features['label']
                )
                features['label'] = raw_features['label']
            
            # 5. 直接进行回测，使用预训练模型
            logger.info("直接进行回测，使用预训练模型")
            
            # 6. 回测
            logger.info("开始回测")
            backtest_results = self.backtester.run(features)
            
            # 更新监控数据
            self.monitor.update_equity_curve(self.backtester.equity_curve)
            self.monitor.update_daily_returns(self.backtester.daily_returns)
            self.monitor.update_recent_trades(self.backtester.trade_records)
            
            # 生成绩效报告
            self.monitor.generate_performance_report()
            
            # 生成月度汇总报告
            logger.info("生成月度汇总报告")
            if hasattr(self.backtester, 'trade_records') and self.backtester.trade_records:
                self.monthly_summary.add_trade_records(self.backtester.trade_records)
                monthly_summaries = self.monthly_summary.generate_all_monthly_summaries()
                
                if monthly_summaries:
                    # 保存月度汇总报告
                    self.monthly_summary.save_all_monthly_summaries()
                    
                    # 打印最新月度汇总报告
                    latest_month = sorted(monthly_summaries.keys())[-1]
                    self.monthly_summary.print_monthly_summary(latest_month)
            
            logger.info(f"回测完成，结果：{backtest_results}")
            return backtest_results
            
        except Exception as e:
            logger.error(f"回测失败：{e}")
            import traceback
            logger.error(f"异常堆栈：{traceback.format_exc()}")
            return None
    
    def run_live(self):
        """运行实盘策略"""
        logger.info("开始运行实盘策略")
        
        # 启动监控面板
        self.monitor.start_monitoring()
        
        # 更新策略状态为运行中
        self.monitor.update_strategy_status({'is_running': True})
        
        # 主循环
        while True:
            try:
                # 1. 获取实时数据
                logger.info("获取实时数据")
                realtime_data = self.data_fetcher.get_realtime_data()
                
                # 2. 全量股票预过滤
                logger.info("股票预过滤")
                filtered_data = self.stock_filter.filter_stocks(realtime_data)
                
                # 3. 数据预处理
                logger.info("数据预处理")
                processed_data = self.data_processor.process(filtered_data)
                
                # 4. 行业轮动策略分析
                logger.info("行业轮动策略分析")
                industry_boom = self.industry_rotation.evaluate_industry_boom(processed_data)
                capital_flow = self.industry_rotation.monitor_industry_capital_flow(processed_data)
                industry_weights = self.industry_rotation.adjust_industry_weights(industry_boom, capital_flow)
                rotation_signals = self.industry_rotation.generate_rotation_signals(industry_weights)
                
                # 5. 特征工程
                logger.info("特征工程")
                raw_features = self.feature_engineer.extract_features(processed_data)
                
                # 6. 特征标准化与归一化
                logger.info("特征标准化与归一化")
                standardized_features = self.feature_standardizer.transform_new_features(raw_features)
                
                # 7. 模型预测（使用融合模型）
                logger.info("模型预测")
                predictions, prediction_probs = self.model_fusion.predict(standardized_features)
                
                # 8. 策略决策（结合行业轮动信号）
                logger.info("策略决策")
                # 将行业轮动信号整合到决策中
                decisions = self.strategy_decision.make_decisions(
                    processed_data, 
                    {
                        'predictions': predictions,
                        'probabilities': prediction_probs,
                        'industry_signals': rotation_signals
                    }
                )
                
                # 9. 风险控制
                logger.info("风险控制检查")
                approved_decisions = self.risk_controller.check(decisions)
                
                # 10. 订单执行
                logger.info("订单执行")
                execution_results = self.order_executor.execute(approved_decisions)
                
                # 更新策略决策模块的持仓信息
                self.strategy_decision.update_holdings(execution_results)
                
                # 更新风险控制状态
                self.risk_controller.update_risk_status(execution_results)
                
                # 更新监控数据
                self.monitor.update_holdings(self.strategy_decision.get_holdings())
                self.monitor.update_recent_trades(self.strategy_decision.get_trade_records())
                
                # 记录交易记录用于月度汇总
                self.monthly_summary.add_trade_records(execution_results)
                
                logger.info(f"订单执行结果：{execution_results}")
                
                # 检查是否需要生成月度汇总（每天结束时）
                current_time = pd.Timestamp.now()
                if current_time.hour == 15 and current_time.minute >= 30:  # 收盘后
                    logger.info("生成当日交易汇总")
                    # 这里可以添加每日汇总逻辑
                
                # 等待下一次循环
                import time
                time.sleep(config.LOOP_INTERVAL)
                
            except Exception as e:
                logger.error(f"实盘运行出错：{e}", exc_info=True)
                time.sleep(config.ERROR_RETRY_INTERVAL)
            finally:
                # 更新策略状态为停止
                self.monitor.update_strategy_status({'is_running': False})
                # 停止监控面板
                self.monitor.stop_monitoring()

def main():
    """主函数"""
    logger.info("启动QMT量化交易策略系统")
    
    system = QMTStrategySystem()
    
    # 根据配置决定运行模式
    if config.RUN_MODE == "backtest":
        # 运行回测
        system.run_backtest(config.BACKTEST_START_DATE, config.BACKTEST_END_DATE)
    elif config.RUN_MODE == "live":
        # 运行实盘
        system.run_live()
    else:
        logger.error(f"无效的运行模式：{config.RUN_MODE}")
        sys.exit(1)

if __name__ == "__main__":
    main()
