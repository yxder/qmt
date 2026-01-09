#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
股票量化交易竞价打板策略系统主程序
"""

import sys
import os
import numpy as np
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
    
    def __init__(self):
        """初始化系统"""
        self.data_fetcher = DataFetcher()
        self.data_processor = DataProcessor()
        self.feature_engineer = FeatureEngineer()
        self.model_trainer = ModelTrainer()
        self.strategy_decision = StrategyDecision()
        self.order_executor = OrderExecutor()
        self.risk_controller = RiskController()
        self.backtester = Backtester()
        self.monitor = MonitorPanel()
        
        # 新增优化模块
        from strategy.feature_standardization import FeatureStandardizer
        from strategy.stock_filter import StockFilter
        from strategy.model_fusion import ModelFusion
        from strategy.industry_rotation import IndustryRotation
        from strategy.parameter_tuning import ParameterTuner
        
        self.feature_standardizer = FeatureStandardizer()  # 特征标准化器
        self.stock_filter = StockFilter()  # 股票过滤器
        self.model_fusion = ModelFusion()  # 模型融合器
        self.industry_rotation = IndustryRotation()  # 行业轮动策略
        self.parameter_tuner = ParameterTuner()  # 参数调优器
        
    def run_backtest(self, start_date, end_date):
        """运行历史回测"""
        logger.info(f"开始运行历史回测，时间范围：{start_date} 至 {end_date}")
        
        # 获取历史数据
        logger.info("获取历史数据")
        historical_data = self.data_fetcher.get_historical_data(start_date, end_date)
        
        # 1. 全量股票预过滤
        logger.info("股票预过滤")
        filtered_data = self.stock_filter.filter_stocks(historical_data)
        
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
            features = self.feature_standardizer.process_features(raw_features)
        
        # 5. 模型训练（支持模型融合）
        logger.info("模型训练与融合")
        # 训练基础模型
        self.model_trainer.train(features)
        
        # 训练融合模型
        X = features.drop('label', axis=1) if 'label' in features.columns else features
        y = features['label'] if 'label' in features.columns else None
        if y is not None:
            self.model_fusion.train(X, y)
            
            # 6. 参数调优（针对融合模型）
            logger.info("参数调优")
            # 调优阈值参数
            thresholds = np.arange(0.1, 0.9, 0.05)
            best_threshold = self.parameter_tuner.tune_threshold_params(
                self.model_fusion.fusion_model, X, y, thresholds
            )
            logger.info(f"最佳阈值：{best_threshold}")
        
        # 7. 行业轮动策略集成
        logger.info("行业轮动策略分析")
        industry_analysis = self.industry_rotation.evaluate_industry_boom(processed_data)
        if industry_analysis is not None:
            logger.info(f"行业景气度分析结果：{industry_analysis}")
        
        # 8. 回测
        logger.info("开始回测")
        backtest_results = self.backtester.run(features)
        
        # 更新监控数据
        self.monitor.update_equity_curve(self.backtester.equity_curve)
        self.monitor.update_daily_returns(self.backtester.daily_returns)
        self.monitor.update_recent_trades(self.backtester.trade_records)
        
        # 生成绩效报告
        self.monitor.generate_performance_report()
        
        logger.info(f"回测完成，结果：{backtest_results}")
        return backtest_results
    
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
                
                logger.info(f"订单执行结果：{execution_results}")
                
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
