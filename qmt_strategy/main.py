#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
股票量化交易竞价打板策略系统主程序
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from qmt_strategy.strategy.data_fetch import DataFetcher
from qmt_strategy.strategy.data_process import DataProcessor
from qmt_strategy.strategy.feature_engineer import FeatureEngineer
from qmt_strategy.strategy.model_train import ModelTrainer
from qmt_strategy.strategy.strategy_decision import StrategyDecision
from qmt_strategy.strategy.order_execution import OrderExecutor
from qmt_strategy.strategy.risk_control import RiskController
from qmt_strategy.strategy.backtest import Backtester
from qmt_strategy.strategy.monitor import MonitorPanel
from qmt_strategy.utils.logger import setup_logger
import qmt_strategy.config as config

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
        
    def run_backtest(self, start_date, end_date):
        """运行历史回测"""
        logger.info(f"开始运行历史回测，时间范围：{start_date} 至 {end_date}")
        
        # 获取历史数据
        logger.info("获取历史数据")
        historical_data = self.data_fetcher.get_historical_data(start_date, end_date)
        
        # 数据预处理
        logger.info("数据预处理")
        processed_data = self.data_processor.process(historical_data)
        
        # 特征工程
        logger.info("特征工程")
        features = self.feature_engineer.extract_features(processed_data)
        
        # 模型训练
        logger.info("模型训练")
        self.model_trainer.train(features)
        
        # 回测
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
                # 获取实时数据
                logger.info("获取实时数据")
                realtime_data = self.data_fetcher.get_realtime_data()
                
                # 数据预处理
                logger.info("数据预处理")
                processed_data = self.data_processor.process(realtime_data)
                
                # 特征工程
                logger.info("特征工程")
                features = self.feature_engineer.extract_features(processed_data)
                
                # 策略决策
                logger.info("策略决策")
                decisions = self.strategy_decision.make_decisions(features)
                
                # 风险控制
                logger.info("风险控制检查")
                approved_decisions = self.risk_controller.check(decisions)
                
                # 订单执行
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
