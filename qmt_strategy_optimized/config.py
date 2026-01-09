#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
股票量化交易竞价打板策略系统配置文件
"""

# 运行模式：backtest（回测）或 live（实盘）
RUN_MODE = "backtest"

# 回测配置
BACKTEST_START_DATE = "20250101"
BACKTEST_END_DATE = "20251231"

# 实盘配置
LOOP_INTERVAL = 60  # 循环间隔，单位：秒
ERROR_RETRY_INTERVAL = 300  # 错误重试间隔，单位：秒

# 数据配置
STOCK_LIST = []  # 股票列表，为空则表示所有沪深A股
HISTORY_DATA_PATH = "data/raw_data/history/"
REALTIME_DATA_PATH = "data/raw_data/realtime/"
PROCESSED_DATA_PATH = "data/processed_data/"
FEATURE_DATA_PATH = "data/feature_data/"

# 模型配置
MODEL_PATH = "models/trained_models/"
MODEL_TYPE = "xgboost"  # 模型类型：logistic, random_forest, xgboost
MODEL_FILE_NAME = "model_xgboost_20260108_164001.joblib"  # 具体模型文件名
TRAIN_DATA_START_DATE = "20251101"  # 模型训练数据开始日期
TRAIN_DATA_END_DATE = "20251231"  # 模型训练数据结束日期
TRAIN_VALID_RATIO = 0.7  # 训练集与验证集比例
MODEL_ACCURACY_THRESHOLD = 0.7  # 模型准确率阈值，低于此值触发模型更新
RANDOM_SEED = 42  # 随机种子，确保结果可复现

# 策略配置
# 动态买点决策参数
BUY_THRESHOLD = 0.1  # 模型预测概率阈值，高于此值考虑买入
BID_INTENSITY_WEIGHT = 0.3  # 竞价强度权重
BUY_ORDER_SIZE_WEIGHT = 0.3  # 封单量权重
MARKET_SENTIMENT_WEIGHT = 0.4  # 市场情绪权重

# 自适应卖点策略参数
PROFIT_TARGET = 0.1  # 目标收益率
STOP_LOSS_RATIO = 0.05  # 止损比例
TRAILING_STOP_RATIO = 0.03  # 跟踪止损比例

# 风险控制配置
MAX_POSITION_PER_STOCK = 0.2  # 单票最大仓位比例
MAX_TOTAL_POSITION = 0.8  # 总仓位最大比例
MAX_DAILY_LOSS = 0.03  # 单日最大亏损比例
MAX_CONSECUTIVE_LOSSES = 5  # 连续亏损最大次数
MAX_DRAWDOWN = 0.1  # 最大回撤比例

# 日志配置
LOG_LEVEL = "DEBUG"  # 日志级别：DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FILE = "logs/strategy.log"  # 日志文件路径
LOG_ROTATION = "daily"  # 日志轮转方式：daily, hourly, size
LOG_BACKUP_COUNT = 30  # 日志备份数量

# 回测配置
INITIAL_CAPITAL = 1000000  # 初始资金，单位：元
COMMISSION_RATE = 0.0003  # 佣金率
SLIPPAGE_RATE = 0.0001  # 滑点率

# 监控配置
MONITOR_PORT = 8080  # 监控面板端口
MONITOR_REFRESH_INTERVAL = 5  # 监控面板刷新间隔，单位：秒

# 报告配置
REPORT_PATH = "reports"  # 报告保存路径
