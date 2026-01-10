#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
回测运行脚本
使用2025年9月A股数据测试竞价打板策略表现
"""

import os
import sys
import pandas as pd
import numpy as np
from utils.logger import setup_logger
from data.data_collector import DataCollector
from data.data_source import DataSource
from strategy.sector_recognizer import SectorRecognizer
from strategy.stock_filter import StockFilter
from strategy.model_predictor import ModelPredictor
from strategy.risk_evaluator import RiskEvaluator
from strategy.decision_generator import DecisionGenerator
from strategy.backtest import Backtester

# 设置日志
logger = setup_logger()


def prepare_backtest_data():
    """准备回测数据"""
    logger.info("开始准备回测数据")
    
    # 创建数据源对象
    data_source = DataSource()
    
    # 获取股票列表
    stock_list = data_source.get_stock_list()
    logger.info(f"获取到{len(stock_list)}只股票")
    
    # 生成模拟的2025年9月数据
    # 实际应用中应从真实数据源获取
    backtest_data = generate_simulated_data(stock_list)
    
    logger.info("回测数据准备完成")
    return backtest_data


def generate_simulated_data(stock_list):
    """生成模拟的回测数据"""
    logger.info("生成模拟的回测数据")
    
    # 生成2025年9月的交易日列表
    trading_dates = pd.bdate_range(start='2025-09-01', end='2025-09-30')
    logger.info(f"生成{len(trading_dates)}个交易日的数据")
    
    all_data = []
    
    for date in trading_dates:
        logger.info(f"生成{date.strftime('%Y-%m-%d')}的数据")
        
        # 为每只股票生成当日数据
        for stock_code in stock_list[:50]:  # 只取前50只股票作为示例
            # 模拟竞价数据
            prev_close = np.random.rand() * 100 + 10
            bid_price = prev_close * (1 + np.random.rand() * 0.15 - 0.05)
            bid_change = (bid_price - prev_close) / prev_close
            
            # 模拟其他数据
            data = {
                'date': date,
                'stock_code': stock_code,
                'price_bid_price': bid_price,
                'price_bid_change': bid_change,
                'price_prev_close': prev_close,
                'price_open': bid_price * (1 + np.random.rand() * 0.02 - 0.01),
                'volume_bid_volume': np.random.randint(10000, 1000000),
                'volume_bid_volume_ratio': np.random.rand() * 5 + 0.5,
                'volume_bid_turnover_rate': np.random.rand() * 0.05 + 0.001,
                'order_book_bid_order_amount': np.random.randint(10000000, 500000000),
                'order_book_bid_order_ratio': np.random.rand() * 0.3 + 0.05,
                'order_book_ask_order_amount': np.random.randint(5000000, 400000000),
                'order_book_order_imbalance': np.random.rand() * 0.5 - 0.25,
                'fund_flow_fund_inflow': np.random.randint(1000000, 100000000),
                'fund_flow_fund_outflow': np.random.randint(500000, 80000000),
                'fund_flow_net_flow': np.random.randint(-50000000, 50000000),
                'fund_flow_large_order_flow': np.random.randint(500000, 50000000),
                'sector': np.random.choice(['金融', '医药', '科技', '消费', '新能源', '地产', '化工', '有色', '机械', '通信']),
                'industry': np.random.choice(['银行', '券商', '医药生物', '电子', '食品饮料', '新能源汽车', '房地产', '化工', '有色金属', '机械设备'])
            }
            
            all_data.append(data)
    
    # 转换为DataFrame
    df = pd.DataFrame(all_data)
    
    # 添加label列（模拟涨停标签）
    # 随机生成10%的涨停样本
    df['label'] = np.random.choice([0, 1], size=len(df), p=[0.9, 0.1])
    
    logger.info(f"生成了{len(df)}行模拟数据")
    logger.info(f"涨停样本数量：{df['label'].sum()}")
    
    return df


def run_backtest():
    """运行回测"""
    logger.info("开始运行回测")
    
    # 1. 准备回测数据
    backtest_data = prepare_backtest_data()
    
    # 2. 创建回测器对象
    backtester = Backtester()
    
    # 3. 运行回测
    backtest_results = backtester.run(backtest_data)
    
    # 4. 输出回测结果
    logger.info("回测结果：")
    for key, value in backtest_results.items():
        if isinstance(value, float):
            logger.info(f"{key}: {value:.4f}")
        else:
            logger.info(f"{key}: {value}")
    
    # 5. 生成回测报告
    logger.info("生成回测报告")
    backtester._generate_backtest_report()
    
    logger.info("回测运行完成")
    return backtest_results


def run_strategy_pipeline():
    """运行完整的策略流程"""
    logger.info("开始运行完整的策略流程")
    
    # 1. 准备数据
    backtest_data = prepare_backtest_data()
    
    # 2. 获取第一个交易日的数据作为示例
    sample_date = backtest_data['date'].iloc[0]
    sample_data = backtest_data[backtest_data['date'] == sample_date]
    logger.info(f"使用{sample_date.strftime('%Y-%m-%d')}的数据作为示例")
    
    # 3. 运行数据采集模块
    data_collector = DataCollector()
    bidding_data = data_collector.collect_bidding_data("09:15", "09:25")
    
    # 如果采集的数据为空，使用模拟数据
    if bidding_data.empty:
        logger.warning("数据采集模块返回空结果，使用模拟数据")
        bidding_data = sample_data.copy()
    
    # 4. 运行板块识别模块
    sector_recognizer = SectorRecognizer()
    top_sectors = sector_recognizer.evaluate_sectors(bidding_data)
    logger.info(f"识别出{len(top_sectors)}个热门板块")
    
    # 5. 运行个股筛选模块
    stock_filter = StockFilter()
    candidate_stocks = stock_filter.filter_stocks(bidding_data, top_sectors)
    logger.info(f"筛选出{len(candidate_stocks)}只候选股票")
    
    # 6. 运行模型预测模块
    model_predictor = ModelPredictor()
    predicted_stocks = model_predictor.predict(candidate_stocks)
    logger.info(f"完成{len(predicted_stocks)}只股票的预测")
    
    # 7. 运行风险评估模块
    risk_evaluator = RiskEvaluator()
    risk_evaluated_stocks = risk_evaluator.evaluate_risk(predicted_stocks)
    logger.info(f"完成{len(risk_evaluated_stocks)}只股票的风险评估")
    
    # 8. 运行决策建议生成模块
    decision_generator = DecisionGenerator()
    decisions = decision_generator.generate_decisions(risk_evaluated_stocks)
    logger.info(f"生成{len(decisions)}个决策建议")
    
    # 9. 筛选TOP决策
    top_decisions = decision_generator.filter_top_decisions(decisions, top_n=5)
    logger.info(f"筛选出{len(top_decisions)}个TOP决策建议")
    
    # 10. 输出TOP决策
    logger.info("TOP决策建议：")
    for idx, row in top_decisions.iterrows():
        logger.info(f"优先级{row['buy_priority']}: 股票{row['stock_code']}，买入区间[{row['buy_price_lower']:.2f}, {row['buy_price_upper']:.2f}]，仓位{row['position_ratio']:.2%}")
    
    logger.info("完整策略流程运行完成")
    return top_decisions


def main():
    """主函数"""
    logger.info("="*50)
    logger.info("开始运行竞价打板策略回测")
    
    try:
        # 运行完整的策略流程
        run_strategy_pipeline()
        
        # 运行回测
        run_backtest()
        
        logger.info("竞价打板策略回测运行完成")
    except Exception as e:
        logger.error(f"回测运行过程中发生错误: {e}", exc_info=True)
    finally:
        logger.info("="*50)


if __name__ == "__main__":
    main()