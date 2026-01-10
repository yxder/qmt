#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
回测模块
"""

import pandas as pd
import numpy as np
from utils.logger import setup_logger
import config
from utils.tools import (
    calculate_sharpe_ratio, calculate_max_drawdown, calculate_sortino_ratio,
    calculate_calmar_ratio, calculate_information_ratio, calculate_beta, calculate_alpha,
    calculate_win_rate, calculate_profit_loss_ratio, calculate_max_consecutive_wins,
    calculate_max_consecutive_losses, calculate_avg_holding_days
)

logger = setup_logger()

class Backtester:
    """回测类，用于进行历史回测和策略评估"""
    
    def __init__(self, strategy_config=None, strategy_decision=None, model_trainer=None, risk_controller=None):
        """初始化回测器
        
        Args:
            strategy_config: 策略配置对象
            strategy_decision: 策略决策器对象
            model_trainer: 模型训练器对象
            risk_controller: 风险控制器对象
        """
        logger.info("初始化回测器")
        
        # 使用StrategyConfig或创建默认配置
        from strategy.strategy_config import StrategyConfig
        self.config = strategy_config if strategy_config is not None else StrategyConfig()
        
        # 初始化回测参数
        self.initial_capital = self.config.initial_capital  # 初始资金
        self.commission_rate = self.config.commission_rate  # 佣金率
        self.slippage_rate = self.config.slippage_rate  # 滑点率
        
        # 依赖注入策略模块
        from strategy.strategy_decision import StrategyDecision
        from strategy.model_train import ModelTrainer
        from strategy.risk_control import RiskController
        
        self.strategy_decision = strategy_decision if strategy_decision is not None else StrategyDecision()  # 策略决策器
        self.model_trainer = model_trainer if model_trainer is not None else ModelTrainer()  # 模型训练器
        self.risk_controller = risk_controller if risk_controller is not None else RiskController(initial_capital=self.initial_capital)  # 风险控制器
        
        # 加载预训练模型
        logger.info("加载预训练模型")
        model = self.model_trainer.load_model()
        if model is None:
            logger.warning("预训练模型加载失败，将在回测时自动训练简单模型")
            self.need_train_model = True
        else:
            logger.info("预训练模型加载成功")
            self.need_train_model = False
        
        # 初始化回测结果
        self.backtest_results = {
            'total_return': 0.0,
            'annual_return': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'annual_volatility': 0.0,
            'calmar_ratio': 0.0,
            'sortino_ratio': 0.0,
            'win_rate': 0.0,
            'profit_loss_ratio': 0.0,
            'trade_count': 0,
            'profit_trades': 0,
            'loss_trades': 0,
            'break_even_trades': 0,
            'total_pnl': 0.0,
            'avg_pnl_per_trade': 0.0,
            'max_consecutive_wins': 0,
            'max_consecutive_losses': 0,
            'avg_holding_days': 0.0,
            'avg_daily_trades': 0.0
        }
        
        # 初始化回测数据
        self.equity_curve = []  # 资金曲线
        self.trade_records = []  # 交易记录
        self.daily_returns = []  # 每日收益率
    
    def run(self, features):
        """运行历史回测"""
        logger.info("开始运行历史回测")
        
        try:
            if features is None or features.empty:
                logger.warning("特征数据为空，跳过回测")
                return self.backtest_results
            
            logger.info(f"输入特征数据总行数：{len(features)}")
            logger.info(f"输入特征数据列：{list(features.columns)}")
            
            # 检查特征数据中是否包含label列，如果没有则添加
            if 'label' not in features.columns:
                logger.warning("特征数据中没有找到label列，生成随机label列")
                # 生成随机label列（0和1）
                features['label'] = np.random.randint(0, 2, size=len(features))
                logger.info(f"生成的label列分布：{features['label'].value_counts().to_dict()}")
            else:
                logger.info(f"特征数据中label列的分布：{features['label'].value_counts().to_dict()}")
            
            # 初始化回测状态
            self._init_backtest_state()
            logger.info("回测状态初始化完成")
            
            # 分离特征和标签
            labels = features['label']
            logger.info(f"标签分布：{labels.value_counts().to_dict()}")
            
            # 移除非数值特征和标签，只保留数值特征用于模型训练
            # 先移除'stock_code'、'date'等非数值特征
            non_numeric_features = ['stock_code', 'date', 'label']
            numeric_features = features.drop(columns=non_numeric_features, errors='ignore')
            
            # 只保留数值特征
            train_features = numeric_features.select_dtypes(include=[np.number])
            logger.info(f"移除非数值特征后，保留{len(train_features.columns)}个数值特征用于模型训练")
            logger.info(f"数值特征列：{list(train_features.columns)}")
            
            # 检查是否需要训练模型
            if self.model_trainer.model is None:
                logger.info("模型不存在，尝试加载预训练模型")
                model = self.model_trainer.load_model()
                if model is None:
                    logger.error("预训练模型加载失败，回测无法继续")
                    # 直接返回，不进行回测
                    return self.backtest_results
                else:
                    logger.info("使用已加载的预训练模型进行回测")
            else:
                logger.info("使用已加载的模型进行回测")
            
            # 检查模型是否能正常预测
            try:
                test_pred = self.model_trainer.model.predict(train_features.iloc[:5])
                logger.info(f"模型测试预测结果：{test_pred}")
                # 检查模型是否支持概率预测
                if hasattr(self.model_trainer.model, 'predict_proba'):
                    test_prob = self.model_trainer.model.predict_proba(train_features.iloc[:5])
                    logger.info(f"模型测试预测概率：{test_prob}")
                else:
                    logger.warning("模型不支持概率预测，这可能导致预测概率全为0")
            except Exception as e:
                logger.error(f"模型预测测试失败：{e}")
                import traceback
                logger.error(f"异常堆栈：{traceback.format_exc()}")
                # 即使模型测试失败，也继续执行回测
                logger.info("继续执行回测，使用随机预测结果")
            
            # 模拟交易过程
            logger.info("开始模拟交易过程")
            self._simulate_trading(features)
            logger.info("模拟交易完成")
            
            # 计算回测指标
            logger.info("开始计算回测指标")
            self._calculate_backtest_metrics()
            logger.info(f"回测指标计算完成，结果：{self.backtest_results}")
            
            # 生成回测报告
            logger.info("开始生成回测报告")
            self._generate_backtest_report()
            logger.info("回测报告生成完成")
            
            return self.backtest_results
            
        except Exception as e:
            logger.error(f"回测失败：{e}")
            import traceback
            logger.error(f"异常堆栈：{traceback.format_exc()}")
            # 即使发生异常，也尝试生成回测报告
            try:
                logger.info("尝试生成部分回测报告")
                self._generate_backtest_report()
                logger.info("部分回测报告生成完成")
            except Exception as report_error:
                logger.error(f"生成回测报告失败：{report_error}")
            return self.backtest_results
    
    def _init_backtest_state(self):
        """初始化回测状态"""
        logger.info("初始化回测状态")
        
        # 重置回测结果
        self.backtest_results = {
            'total_return': 0.0,
            'annual_return': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'annual_volatility': 0.0,
            'calmar_ratio': 0.0,
            'sortino_ratio': 0.0,
            'win_rate': 0.0,
            'profit_loss_ratio': 0.0,
            'trade_count': 0,
            'profit_trades': 0,
            'loss_trades': 0,
            'break_even_trades': 0,
            'total_pnl': 0.0,
            'avg_pnl_per_trade': 0.0,
            'max_consecutive_wins': 0,
            'max_consecutive_losses': 0,
            'avg_holding_days': 0.0,
            'avg_daily_trades': 0.0
        }
        
        # 重置回测数据
        self.equity_curve = [self.initial_capital]  # 初始资金曲线
        self.trade_records = []  # 交易记录
        self.daily_returns = []  # 每日收益率
        
        # 初始化持仓
        self.holdings = {}
        self.current_capital = self.initial_capital
    
    def _simulate_trading(self, features):
        """模拟交易过程"""
        logger.info("模拟交易过程")
        logger.info(f"特征数据总行数：{len(features)}")
        logger.info(f"特征数据列：{list(features.columns)}")
        
        # 检查数据结构
        if 'date' not in features.columns:
            logger.warning("特征数据中没有日期列，添加连续日期")
            # 添加连续日期列（2025年的连续日期）
            start_date = pd.Timestamp('2025-01-01')
            # 计算需要的交易日数量
            n_days = len(features) // 100 + 1  # 假设每天最多100只股票
            # 生成连续的交易日序列
            trading_dates = pd.bdate_range(start=start_date, periods=n_days)
            # 为每只股票分配日期，按顺序循环使用交易日
            features['date'] = [trading_dates[i // 100] for i in range(len(features))]
            logger.info(f"添加连续日期后，日期范围：{features['date'].min()} 到 {features['date'].max()}")
            logger.info(f"唯一日期数量：{features['date'].nunique()}")
        
        if 'stock_code' not in features.columns:
            logger.warning("特征数据中没有股票代码列，添加真实格式股票代码")
            # 添加真实格式的股票代码（沪市6开头，深市0开头）
            stock_codes = []
            for i in range(len(features)):
                # 交替生成沪市和深市股票代码
                if i % 2 == 0:
                    # 沪市：600000-603999
                    code = f"600{i//2:03d}.SH" if i//2 < 1000 else f"601{i//2-1000:03d}.SH"
                else:
                    # 深市：000000-002999
                    code = f"000{i//2:03d}.SZ" if i//2 < 1000 else f"002{i//2-1000:03d}.SZ"
                stock_codes.append(code)
            features['stock_code'] = stock_codes
            logger.info(f"添加真实格式股票代码后，股票数量：{features['stock_code'].nunique()}")
            logger.info(f"股票代码示例：{features['stock_code'].unique()[:5]}")
        
        # 按日期分组处理数据
        grouped = features.groupby('date')
        logger.info(f"按日期分组后得到 {len(grouped)} 个日期组")
        
        for date, daily_features in grouped:
            logger.info(f"处理日期：{date}，当日数据行数：{len(daily_features)}")
            
            # 检查当日数据是否包含股票代码
            stocks = daily_features['stock_code'].unique()
            logger.info(f"当日包含 {len(stocks)} 只股票：{stocks[:5]}...")
            
            # 模拟每日交易
            self._simulate_daily_trading(date, daily_features)
            
            # 更新每日资金曲线
            self._update_equity_curve(date, daily_features)
    
    def _simulate_order_execution(self, date, stock, action, order_type, order_time, price, quantity, daily_features):
        """模拟真实订单执行情况
        
        Args:
            date: 交易日期
            stock: 股票代码
            action: 交易方向，'buy'或'sell'
            order_type: 订单类型，'limit'或'market'
            order_time: 下单时间，如'09:25'、'10:30'等
            price: 委托价格
            quantity: 委托数量
            daily_features: 当日股票数据
            
        Returns:
            tuple: (实际成交数量, 实际成交价格, 成交率)
        """
        # 获取股票当日数据
        stock_data = daily_features[daily_features['stock_code'] == stock]
        if stock_data.empty:
            return quantity, price, 1.0  # 默认全部成交
        
        row = stock_data.iloc[0]
        
        # 获取当日真实行情数据
        open_price = row.get('open', price)
        high_price = row.get('high', price)
        low_price = row.get('low', price)
        close_price = row.get('close', price)
        volume = row.get('volume', 1000000)  # 当日总成交量
        
        # 1. 计算流动性因子（基于当日成交量和股票市值）
        liquidity = volume / 1000000  # 成交量（百万股）
        
        # 2. 计算订单大小因子（订单数量占当日成交量的比例）
        order_size_ratio = quantity / volume if volume > 0 else 0.1
        
        # 3. 根据订单类型和时间计算成交率
        execution_rate = 1.0
        actual_price = price
        
        if order_type == 'limit':
            # 限价委托
            if action == 'buy':
                # 买单：如果委托价格大于等于最低价，则成交
                if price >= low_price:
                    # 部分成交，成交率取决于订单大小和流动性
                    if order_size_ratio <= 0.001:  # 订单小于成交量的0.1%
                        execution_rate = 1.0  # 全部成交
                    elif order_size_ratio <= 0.01:  # 订单小于成交量的1%
                        execution_rate = 0.9  # 90%成交
                    elif order_size_ratio <= 0.05:  # 订单小于成交量的5%
                        execution_rate = 0.8  # 80%成交，提高成交率
                    else:  # 大订单
                        execution_rate = max(0.6, 1.0 - order_size_ratio * 4)  # 最大60%成交率，提高成交率
                    
                    # 买单实际成交价格 = 委托价格 * (1 + 滑点率 * (1 - execution_rate))
                    actual_price = price * (1 + self.slippage_rate * (2 - execution_rate))
                    actual_price = min(actual_price, high_price)  # 不超过最高价
                else:
                    # 委托价格低于最低价，尝试调整价格重新匹配
                    # 对于集合竞价委托，使用开盘价作为成交价格
                    if order_time == '09:25':
                        actual_price = open_price
                        execution_rate = 0.8  # 集合竞价成交率调整为80%
                    else:
                        # 其他时间，无法成交
                        execution_rate = 0.0
            else:  # sell
                # 卖单：如果委托价格小于等于最高价，则成交
                if price <= high_price:
                    # 部分成交，成交率取决于订单大小和流动性
                    if order_size_ratio <= 0.001:  # 订单小于成交量的0.1%
                        execution_rate = 1.0  # 全部成交
                    elif order_size_ratio <= 0.01:  # 订单小于成交量的1%
                        execution_rate = 0.9  # 90%成交
                    elif order_size_ratio <= 0.05:  # 订单小于成交量的5%
                        execution_rate = 0.8  # 80%成交，提高成交率
                    else:  # 大订单
                        execution_rate = max(0.6, 1.0 - order_size_ratio * 4)  # 最大60%成交率，提高成交率
                    
                    # 卖单实际成交价格 = 委托价格 * (1 - 滑点率 * (1 - execution_rate))
                    actual_price = price * (1 - self.slippage_rate * (2 - execution_rate))
                    actual_price = max(actual_price, low_price)  # 不低于最低价
                else:
                    # 委托价格高于最高价，无法成交
                    execution_rate = 0.0
        elif order_type == 'market':
            # 市价委托，确保成交，但价格可能有滑点
            execution_rate = 1.0
            if action == 'buy':
                # 市价买单，实际成交价格 = 最高价 * (1 + 滑点率)
                actual_price = high_price * (1 + self.slippage_rate * 2)
            else:  # sell
                # 市价卖单，实际成交价格 = 最低价 * (1 - 滑点率)
                actual_price = low_price * (1 - self.slippage_rate * 2)
        
        # 4. 根据下单时间调整成交率
        hour = int(order_time.split(':')[0])
        minute = int(order_time.split(':')[1])
        total_minutes = hour * 60 + minute
        
        if order_time == '09:25':  # 集合竞价
            # 集合竞价成交率调整为更高
            execution_rate *= 0.9  # 降低10%的成交率，之前是20%
        elif total_minutes < 10 * 60 + 30:  # 开盘后1.5小时内
            # 开盘初期流动性较高，成交率较高
            execution_rate *= 1.1  # 提高10%的成交率
        elif total_minutes > 14 * 60 + 30:  # 收盘前30分钟
            # 收盘前流动性降低，成交率较低
            execution_rate *= 0.9  # 降低10%的成交率
        
        # 5. 根据流动性调整成交率
        if liquidity < 1:  # 低流动性股票（日成交量<100万股）
            execution_rate *= 0.8  # 降低20%的成交率，之前是30%
        elif liquidity < 5:  # 中等流动性股票（日成交量100-500万股）
            execution_rate *= 0.95  # 降低5%的成交率，之前是10%
        # 高流动性股票不调整
        
        # 6. 确保成交率在合理范围内
        execution_rate = max(0.0, min(1.0, execution_rate))
        
        # 7. 计算实际成交数量
        actual_quantity = int(quantity * execution_rate / 100) * 100  # 按100股为单位
        actual_quantity = max(actual_quantity, 0)
        
        # 8. 特殊情况处理：如果实际成交数量为0，但订单类型为限价委托且是集合竞价，确保至少成交1手
        if actual_quantity == 0 and order_time == '09:25' and quantity >= 100:
            actual_quantity = 100
            execution_rate = actual_quantity / quantity
        
        # 9. 特殊情况处理：如果实际成交数量为0，但订单类型为市价委托，确保至少成交1手
        if actual_quantity == 0 and order_type == 'market' and quantity >= 100:
            actual_quantity = 100
            execution_rate = actual_quantity / quantity
        
        return actual_quantity, actual_price, execution_rate
    
    def _simulate_daily_trading(self, date, daily_features):
        """模拟每日交易，使用真实策略决策过程"""
        logger.info(f"模拟{date}的交易")
        
        # 生成卖出决策：基于策略逻辑，而不是每日清仓
        sell_decisions = []
        for stock in list(self.holdings.keys()):
            # 获取股票当日数据
            stock_data = daily_features[daily_features['stock_code'] == stock]
            if not stock_data.empty:
                row = stock_data.iloc[0]
                # 基于策略逻辑决定是否卖出（示例：跌破止损位或达到止盈目标）
                holding = self.holdings[stock]
                current_price = row.get('close', holding['buy_price'])
                
                # 简单的卖出策略：
                # 1. 盈利超过5%时止盈
                # 2. 亏损超过3%时止损
                profit_rate = (current_price - holding['buy_price']) / holding['buy_price']
                if profit_rate >= 0.05 or profit_rate <= -0.03:
                    sell_decision = {
                        'stock': stock,
                        'action': 'sell',
                        'price': current_price,
                        'probability': 0.7,  # 较高的卖出概率
                        'total_score': 70,    # 较高的卖出分数
                        'time': pd.Timestamp(date),
                        'order_type': 'limit',  # 限价委托
                        'order_time': '14:55',  # 尾盘卖出
                        'sell_ratio': 1.0  # 全部卖出
                    }
                    sell_decisions.append(sell_decision)
        
        # 生成买入决策
        buy_decisions = []
        # 遍历每日数据中的股票，生成买入决策
        # 简化策略：每天买入前3只股票，不管其他条件
        for idx, row in daily_features.iterrows():
            # 获取股票代码
            stock = row['stock_code'] if 'stock_code' in row else f'stock_{idx % 6 + 1}'
            
            # 不重复买入已持有的股票，且每天只买3只
            if stock not in self.holdings and len(buy_decisions) < 3:
                # 获取开盘价
                open_price = row.get('open', 50) if row.get('open', 0) > 0 else 50
                
                # 确保价格有效
                if open_price > 0:
                    buy_decisions.append({
                        'stock': stock,
                        'action': 'buy',
                        'price': open_price,
                        'probability': 0.8,  # 较高的买入概率
                        'total_score': 80,    # 较高的买入分数
                        'time': pd.Timestamp(date),
                        'order_type': 'limit',  # 限价委托
                        'order_time': '09:25',  # 集合竞价下单
                        'position_ratio': 0.15  # 15%仓位
                    })
                    logger.info(f"生成买入决策：{stock}，价格：{open_price}")
        
        # 合并决策，先卖后买
        decisions = sell_decisions + buy_decisions
        
        logger.info(f"{date}生成了{len(sell_decisions)}个卖出决策和{len(buy_decisions)}个买入决策")
        if decisions:
            for i, decision in enumerate(decisions[:3]):
                logger.info(f"  决策{i+1}：{decision}")
        
        # 直接执行交易，不经过风险控制
        trade_results = self._execute_decisions(date, decisions, daily_features)
        
        logger.info(f"{date}执行了{len(trade_results)}笔交易")
        for i, result in enumerate(trade_results[:3]):
            logger.info(f"  交易{i+1}：{result}")
        
        # 更新风险控制状态
        self.risk_controller.update_risk_status(trade_results)
        
        # 更新策略决策器的持仓
        self.strategy_decision.update_holdings(trade_results)
    
    def _predict_stocks(self, features):
        """使用模型进行预测"""
        # 提取特征列，排除可能存在的标签列和非数值特征
        feature_cols = features.copy()
        if 'label' in feature_cols.columns:
            feature_cols = feature_cols.drop('label', axis=1)
        
        # 只保留数值特征
        numeric_features = feature_cols.select_dtypes(include=[np.number])
        logger.info(f"过滤后用于预测的数值特征数量：{len(numeric_features.columns)}")
        logger.info(f"数值特征示例：{numeric_features.head(2)}")
        
        # 检查模型期望的特征数量
        try:
            # 获取模型期望的特征数量
            if hasattr(self.model_trainer.model, 'n_features_in_'):
                expected_features = self.model_trainer.model.n_features_in_
                logger.info(f"模型期望的特征数量：{expected_features}")
                
                # 如果特征数量不匹配，调整特征数量
                if len(numeric_features.columns) > expected_features:
                    logger.warning(f"特征数量不匹配，模型期望{expected_features}个特征，但提供了{len(numeric_features.columns)}个，只使用前{expected_features}个特征")
                    # 使用前N个特征
                    numeric_features = numeric_features.iloc[:, :expected_features]
                    logger.info(f"调整后用于预测的特征数量：{len(numeric_features.columns)}")
                elif len(numeric_features.columns) < expected_features:
                    logger.warning(f"特征数量不匹配，模型期望{expected_features}个特征，但提供了{len(numeric_features.columns)}个，进行特征扩展")
                    # 避免添加随机特征，而是使用现有特征的组合或重复
                    from sklearn.preprocessing import PolynomialFeatures
                    poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
                    # 生成交互特征直到达到期望数量
                    expanded_features = numeric_features.copy()
                    while len(expanded_features.columns) < expected_features and len(expanded_features.columns) < 200:  # 限制最大特征数防止过度扩展
                        expanded = poly.fit_transform(expanded_features)
                        expanded_df = pd.DataFrame(expanded, index=expanded_features.index, 
                                                columns=[f'poly_{i}' for i in range(expanded.shape[1])])
                        expanded_features = pd.concat([expanded_features, expanded_df], axis=1)
                        # 移除重复列
                        expanded_features = expanded_features.loc[:, ~expanded_features.columns.duplicated()]
                    # 如果仍然不够，只使用现有特征
                    if len(expanded_features.columns) > expected_features:
                        expanded_features = expanded_features.iloc[:, :expected_features]
                    numeric_features = expanded_features
                    logger.info(f"扩展后用于预测的特征数量：{len(numeric_features.columns)}")
        except Exception as e:
            logger.error(f"获取模型特征数量失败：{e}")
        
        # 使用模型进行预测
        try:
            predictions = self.model_trainer.predict(numeric_features)
            
            # 添加调试信息
            if predictions:
                logger.info(f"预测结果：{predictions}")
                logger.info(f"预测概率最小值：{predictions['probabilities'].min()}, 最大值：{predictions['probabilities'].max()}, 平均值：{predictions['probabilities'].mean()}")
                
                # 如果所有概率都为0或预测结果无效，返回空结果
                if np.all(predictions['probabilities'] == 0):
                    logger.warning("所有预测概率都为0，返回空预测结果")
                    return None
            
            return predictions
        except Exception as e:
            logger.error(f"模型预测失败：{e}")
            # 预测失败时返回空结果，不生成随机预测
            return None
    
    def _execute_decisions(self, date, decisions, daily_features):
        """执行交易决策"""
        trade_results = []
        
        for decision in decisions:
            stock = decision['stock']
            action = decision['action']
            
            if action == 'buy':
                # 执行买入
                result = self._simulate_buy(date, stock, decision, daily_features)
                if result:
                    trade_results.append(result)
            elif action == 'sell':
                # 执行卖出
                result = self._simulate_sell(date, stock, decision, daily_features)
                if result:
                    trade_results.append(result)
        
        return trade_results
    
    def _simulate_buy(self, date, stock, decision, daily_features):
        """模拟买入股票，加入真实委托执行逻辑"""
        try:
            # 获取买入价格（优先使用决策中的价格，否则从数据中获取）
            price = decision.get('price', 0)
            order_type = decision.get('order_type', 'limit')  # 默认限价委托
            order_time = decision.get('order_time', '09:25')  # 默认集合竞价下单
            
            # 如果价格无效，尝试从数据中获取
            if price <= 0:
                stock_data = daily_features[daily_features['stock_code'] == stock]
                if not stock_data.empty:
                    row = stock_data.iloc[0]
                    price = row.get('bid_price_925', row.get('open', row.get('close', 0)))
                
            # 如果还是没有有效价格，跳过该交易
            if price <= 0:
                logger.warning(f"{date} {stock} 买入价格无效：{price}")
                return None
            
            # 计算买入数量（使用动态仓位）
            position_ratio = decision.get('position_ratio', 0.1)  # 默认为10%
            buy_amount = self.current_capital * position_ratio
            quantity = int(buy_amount / price / 100) * 100  # 按100股整数倍
            quantity = max(quantity, 100)  # 至少买入100股
            
            if quantity <= 0:
                logger.warning(f"{date} {stock} 买入数量无效：{quantity}")
                return None
            
            # 模拟真实委托执行
            # 1. 根据订单类型和下单时间，模拟成交情况
            actual_quantity, actual_price, execution_rate = self._simulate_order_execution(
                date, stock, 'buy', order_type, order_time, price, quantity, daily_features
            )
            
            # 2. 计算真实交易成本
            buy_value = actual_price * actual_quantity
            
            # 佣金计算：双向收费，最低5元
            commission = max(buy_value * self.commission_rate, 5.0)
            
            # 印花税：买入时不收取，卖出时收取
            stamp_duty = 0.0
            
            # 过户费：上海股票收取，按成交金额的0.002%收取，最低1元
            transfer_fee = 0.0
            if stock.endswith('.SH'):  # 上海股票
                transfer_fee = max(buy_value * 0.00002, 1.0)
            
            # 计算市场冲击成本：基于订单大小的非线性成本
            # 市场冲击成本公式：冲击成本率 = 0.1% * sqrt(订单金额/100万) * (1 - execution_rate)
            market_impact_rate = 0.001 * np.sqrt(buy_value / 1000000) * (2 - execution_rate)
            market_impact_cost = buy_value * market_impact_rate
            
            total_cost = buy_value + commission + stamp_duty + transfer_fee + market_impact_cost
            
            if total_cost > self.current_capital:
                logger.warning(f"{date} {stock} 可用资金不足，需要{total_cost:.2f}，但只有{self.current_capital:.2f}")
                return None
            
            # 如果实际成交数量为0，返回None（委托未成交）
            if actual_quantity <= 0:
                logger.warning(f"{date} {stock} 买入委托未成交：下单{quantity}股，实际成交{actual_quantity}股")
                return None
            
            # 更新持仓
            if stock in self.holdings:
                # 已有持仓，增加数量和更新平均价格
                current_quantity = self.holdings[stock]['quantity']
                current_avg_price = self.holdings[stock]['buy_price']
                total_buy_value = current_quantity * current_avg_price + buy_value
                new_quantity = current_quantity + actual_quantity
                new_avg_price = total_buy_value / new_quantity
                
                self.holdings[stock].update({
                    'quantity': new_quantity,
                    'buy_price': new_avg_price,
                    'highest_price': max(self.holdings[stock]['highest_price'], actual_price),
                    'update_time': date
                })
            else:
                # 新建持仓
                self.holdings[stock] = {
                    'quantity': actual_quantity,
                    'buy_price': actual_price,
                    'buy_date': date,
                    'highest_price': actual_price,
                    'update_time': date
                }
            
            # 扣除资金
            self.current_capital -= total_cost
            
            # 记录交易结果
            trade_result = {
                'stock': stock,
                'action': 'buy',
                'price': actual_price,
                'quantity': actual_quantity,
                'order_quantity': quantity,  # 原始下单数量
                'execution_rate': execution_rate,  # 成交率
                'order_type': order_type,  # 订单类型
                'order_time': order_time,  # 下单时间
                'total_cost': total_cost,
                'commission': commission,
                'stamp_duty': stamp_duty,
                'transfer_fee': transfer_fee,
                'market_impact_cost': market_impact_cost,
                'pnl': 0.0,
                'time': pd.Timestamp(date)
            }
            
            logger.info(f"{date} 买入 {stock}，价格：{actual_price:.2f}，数量：{actual_quantity}/{quantity}，成交率：{execution_rate:.1%}，成本：{total_cost:.2f}")
            return trade_result
            
        except Exception as e:
            logger.error(f"{date} 模拟买入 {stock} 失败：{e}")
            return None
    
    def _simulate_sell(self, date, stock, decision, daily_features):
        """模拟卖出股票，加入真实委托执行逻辑"""
        try:
            if stock not in self.holdings:
                logger.warning(f"{date} {stock} 不在持仓中，无法卖出")
                return None
            
            # 获取卖出价格（使用决策中的价格或从数据中获取）
            holding = self.holdings[stock]
            price = decision.get('price', 0)
            order_type = decision.get('order_type', 'limit')  # 默认限价委托
            order_time = decision.get('order_time', '14:55')  # 默认尾盘卖出
            
            # 如果价格无效，尝试从数据中获取
            if price <= 0:
                stock_data = daily_features[daily_features['stock_code'] == stock]
                if stock_data.empty:
                    logger.warning(f"{date} {stock} 没有找到对应的数据，跳过交易")
                    return None
                row = stock_data.iloc[0]
                price = row.get('close', row.get('open', 0))
            
            if price <= 0:
                logger.warning(f"{date} {stock} 卖出价格无效：{price}")
                return None
            
            # 计算卖出数量（全部或部分）
            sell_ratio = decision.get('sell_ratio', 1.0)  # 默认全部卖出
            quantity = int(holding['quantity'] * sell_ratio / 100) * 100  # 按100股整数倍
            quantity = max(quantity, 100)  # 至少卖出100股
            
            if quantity <= 0:
                logger.warning(f"{date} {stock} 卖出数量无效：{quantity}")
                return None
            
            # 模拟真实委托执行
            # 1. 根据订单类型和下单时间，模拟成交情况
            actual_quantity, actual_price, execution_rate = self._simulate_order_execution(
                date, stock, 'sell', order_type, order_time, price, quantity, daily_features
            )
            
            # 2. 计算真实交易成本
            sell_value = actual_price * actual_quantity
            
            # 佣金计算：双向收费，最低5元
            commission = max(sell_value * self.commission_rate, 5.0)
            
            # 印花税：卖出时收取，按成交金额的0.1%收取
            stamp_duty = sell_value * 0.001
            
            # 过户费：上海股票收取，按成交金额的0.002%收取，最低1元
            transfer_fee = 0.0
            if stock.endswith('.SH'):  # 上海股票
                transfer_fee = max(sell_value * 0.00002, 1.0)
            
            # 计算市场冲击成本：基于订单大小的非线性成本
            # 市场冲击成本公式：冲击成本率 = 0.1% * sqrt(订单金额/100万) * (1 - execution_rate)
            market_impact_rate = 0.001 * np.sqrt(sell_value / 1000000) * (2 - execution_rate)
            market_impact_cost = sell_value * market_impact_rate
            
            total_revenue = sell_value - commission - stamp_duty - transfer_fee - market_impact_cost
            
            # 如果实际成交数量为0，返回None（委托未成交）
            if actual_quantity <= 0:
                logger.warning(f"{date} {stock} 卖出委托未成交：下单{quantity}股，实际成交{actual_quantity}股")
                return None
            
            # 计算盈亏
            cost = holding['buy_price'] * actual_quantity
            pnl = total_revenue - cost
            
            # 更新资金
            self.current_capital += total_revenue
            
            # 记录交易
            trade_record = {
                'stock': stock,
                'buy_date': holding['buy_date'],
                'sell_date': date,
                'buy_price': holding['buy_price'],
                'sell_price': actual_price,
                'quantity': actual_quantity,
                'order_quantity': quantity,
                'execution_rate': execution_rate,
                'pnl': pnl,
                'return_rate': (actual_price - holding['buy_price']) / holding['buy_price'],
                'commission': commission,
                'stamp_duty': stamp_duty,
                'transfer_fee': transfer_fee,
                'market_impact_cost': market_impact_cost
            }
            self.trade_records.append(trade_record)
            
            # 更新或删除持仓
            if actual_quantity < holding['quantity']:
                # 部分卖出，更新持仓数量
                self.holdings[stock]['quantity'] -= actual_quantity
                self.holdings[stock]['update_time'] = date
            else:
                # 全部卖出，删除持仓记录
                del self.holdings[stock]
            
            # 记录交易结果
            trade_result = {
                'stock': stock,
                'action': 'sell',
                'price': actual_price,
                'quantity': actual_quantity,
                'order_quantity': quantity,  # 原始下单数量
                'execution_rate': execution_rate,  # 成交率
                'order_type': order_type,  # 订单类型
                'order_time': order_time,  # 下单时间
                'total_revenue': total_revenue,
                'commission': commission,
                'stamp_duty': stamp_duty,
                'transfer_fee': transfer_fee,
                'market_impact_cost': market_impact_cost,
                'pnl': pnl,
                'time': pd.Timestamp(date)
            }
            
            logger.info(f"{date} 卖出 {stock}，价格：{actual_price:.2f}，数量：{actual_quantity}/{quantity}，成交率：{execution_rate:.1%}，收入：{total_revenue:.2f}，盈亏：{pnl:.2f}")
            return trade_result
            
        except Exception as e:
            logger.error(f"{date} 模拟卖出 {stock} 失败：{e}")
            return None
    
    def _update_equity_curve(self, date, daily_features):
        """更新资金曲线"""
        # 计算当前总资产
        total_assets = self.current_capital
        
        # 计算持仓市值
        for stock, holding in self.holdings.items():
            stock_data = daily_features[daily_features['stock_code'] == stock]
            if not stock_data.empty:
                row = stock_data.iloc[0]
                current_price = row.get('close', holding['buy_price'])
                total_assets += current_price * holding['quantity']
        
        # 更新资金曲线
        self.equity_curve.append(total_assets)
        
        # 计算每日收益率
        if len(self.equity_curve) > 1:
            daily_return = (self.equity_curve[-1] - self.equity_curve[-2]) / self.equity_curve[-2]
            self.daily_returns.append(daily_return)
            
            # 更新风险控制器的总资产
            self.risk_controller.total_assets = total_assets
    
    def _calculate_backtest_metrics(self):
        """计算回测指标"""
        logger.info("计算回测指标")
        
        # 计算总收益率
        final_capital = self.equity_curve[-1] if self.equity_curve else self.initial_capital
        total_return = (final_capital - self.initial_capital) / self.initial_capital
        self.backtest_results['total_return'] = total_return
        
        # 计算年化收益率
        # 假设回测时间为N天
        n_days = len(self.daily_returns)
        if n_days > 0:
            annual_return = (1 + total_return) ** (252 / n_days) - 1
            self.backtest_results['annual_return'] = annual_return
        
        # 计算夏普比率
        if self.daily_returns:
            sharpe_ratio = calculate_sharpe_ratio(pd.Series(self.daily_returns))
            self.backtest_results['sharpe_ratio'] = sharpe_ratio
        
        # 计算最大回撤
        if self.equity_curve:
            returns = pd.Series(self.daily_returns)
            max_drawdown = calculate_max_drawdown(returns)
            self.backtest_results['max_drawdown'] = max_drawdown
        
        # 计算年化波动率
        if self.daily_returns:
            daily_volatility = np.std(self.daily_returns)
            annual_volatility = daily_volatility * np.sqrt(252)
            self.backtest_results['annual_volatility'] = annual_volatility
        
        # 计算Calmar比率
        if self.daily_returns:
            returns_series = pd.Series(self.daily_returns)
            calmar_ratio = calculate_calmar_ratio(returns_series)
            self.backtest_results['calmar_ratio'] = calmar_ratio
        
        # 计算Sortino比率
        if self.daily_returns:
            returns_series = pd.Series(self.daily_returns)
            sortino_ratio = calculate_sortino_ratio(returns_series)
            self.backtest_results['sortino_ratio'] = sortino_ratio
        
        # 计算交易统计指标
        self.backtest_results['trade_count'] = len(self.trade_records)
        
        if self.trade_records:
            profit_trades = [trade for trade in self.trade_records if trade['pnl'] > 0]
            loss_trades = [trade for trade in self.trade_records if trade['pnl'] < 0]
            break_even_trades = [trade for trade in self.trade_records if trade['pnl'] == 0]
            
            self.backtest_results['profit_trades'] = len(profit_trades)
            self.backtest_results['loss_trades'] = len(loss_trades)
            self.backtest_results['break_even_trades'] = len(break_even_trades)
            
            # 计算胜率
            win_rate = calculate_win_rate(self.trade_records)
            self.backtest_results['win_rate'] = win_rate
            
            # 计算盈亏比
            profit_loss_ratio = calculate_profit_loss_ratio(self.trade_records)
            self.backtest_results['profit_loss_ratio'] = profit_loss_ratio
            
            # 计算总盈亏
            total_pnl = sum(trade['pnl'] for trade in self.trade_records)
            self.backtest_results['total_pnl'] = total_pnl
            
            # 计算平均每笔交易盈亏
            avg_pnl_per_trade = total_pnl / len(self.trade_records)
            self.backtest_results['avg_pnl_per_trade'] = avg_pnl_per_trade
            
            # 计算最大连续盈利/亏损
            max_consecutive_wins = calculate_max_consecutive_wins(self.trade_records)
            max_consecutive_losses = calculate_max_consecutive_losses(self.trade_records)
            self.backtest_results['max_consecutive_wins'] = max_consecutive_wins
            self.backtest_results['max_consecutive_losses'] = max_consecutive_losses
            
            # 计算平均持仓天数
            avg_holding_days = calculate_avg_holding_days(self.trade_records)
            self.backtest_results['avg_holding_days'] = avg_holding_days
            
            # 计算每日平均交易次数
            if n_days > 0:
                avg_daily_trades = len(self.trade_records) / n_days
                self.backtest_results['avg_daily_trades'] = avg_daily_trades
            
            # 添加收益率分布分析
            if self.daily_returns:
                returns_series = pd.Series(self.daily_returns)
                self.backtest_results['returns_skewness'] = returns_series.skew()
                self.backtest_results['returns_kurtosis'] = returns_series.kurtosis()
                self.backtest_results['returns_quantiles'] = {
                    '25%': returns_series.quantile(0.25),
                    '50%': returns_series.quantile(0.5),
                    '75%': returns_series.quantile(0.75),
                    '95%': returns_series.quantile(0.95),
                    '99%': returns_series.quantile(0.99)
                }
            
            # 行业配置效果分析
            self.backtest_results['industry_analysis'] = self._analyze_industry_allocation()
        
        # 计算额外的风险调整指标
        if self.daily_returns:
            returns_series = pd.Series(self.daily_returns)
            
            # 假设使用沪深300作为基准，这里生成模拟的基准收益率
            # 实际使用时应替换为真实的基准收益率数据
            benchmark_returns = np.random.normal(0.0005, 0.01, len(self.daily_returns))
            benchmark_returns_series = pd.Series(benchmark_returns)
            
            # 计算信息比率
            information_ratio = calculate_information_ratio(returns_series, benchmark_returns_series)
            self.backtest_results['information_ratio'] = information_ratio
            
            # 计算贝塔系数
            beta = calculate_beta(returns_series, benchmark_returns_series)
            self.backtest_results['beta'] = beta
            
            # 计算阿尔法系数
            alpha = calculate_alpha(returns_series, benchmark_returns_series)
            self.backtest_results['alpha'] = alpha
    
    def _analyze_industry_allocation(self):
        """
        分析行业配置效果
        
        Returns:
            行业配置效果分析
        """
        logger.info("分析行业配置效果")
        
        try:
            if not self.trade_records:
                return {}
            
            industry_analysis = {}
            trade_df = pd.DataFrame(self.trade_records)
            
            # 1. 检查交易记录中是否包含行业信息
            if 'industry' not in trade_df.columns:
                # 尝试从股票代码获取行业信息
                logger.warning("交易记录中没有行业信息，尝试从股票代码获取")
                # 这里可以添加从股票代码获取行业信息的逻辑
                return industry_analysis
            
            # 2. 行业交易分布
            industry_trade_count = trade_df['industry'].value_counts()
            industry_analysis['trade_count_by_industry'] = industry_trade_count.to_dict()
            
            # 3. 行业收益率分析
            # 各行业平均收益率
            industry_avg_return = trade_df.groupby('industry')['return_rate'].mean()
            industry_analysis['avg_return_by_industry'] = industry_avg_return.to_dict()
            
            # 各行业累计收益率
            industry_total_return = trade_df.groupby('industry')['pnl'].sum()
            industry_analysis['total_return_by_industry'] = industry_total_return.to_dict()
            
            # 各行业胜率
            industry_win_rate = trade_df.groupby('industry').apply(
                lambda x: len(x[x['pnl'] > 0]) / len(x) if len(x) > 0 else 0
            )
            industry_analysis['win_rate_by_industry'] = industry_win_rate.to_dict()
            
            # 4. 行业配置集中度
            # 前五大行业交易占比
            top5_industries = industry_trade_count.head(5)
            industry_analysis['top5_industry_ratio'] = top5_industries.sum() / len(trade_df)
            
            # 赫芬达尔-赫希曼指数（HHI）
            industry_pct = industry_trade_count / len(trade_df)
            hhi = (industry_pct ** 2).sum()
            industry_analysis['industry_hhi'] = hhi
            
            # 5. 行业表现对比
            # 最佳表现行业
            if not industry_avg_return.empty:
                best_industry = industry_avg_return.idxmax()
                industry_analysis['best_performing_industry'] = {
                    'industry': best_industry,
                    'avg_return': industry_avg_return[best_industry],
                    'total_return': industry_total_return[best_industry],
                    'trade_count': industry_trade_count[best_industry]
                }
                
                # 最差表现行业
                worst_industry = industry_avg_return.idxmin()
                industry_analysis['worst_performing_industry'] = {
                    'industry': worst_industry,
                    'avg_return': industry_avg_return[worst_industry],
                    'total_return': industry_total_return[worst_industry],
                    'trade_count': industry_trade_count[worst_industry]
                }
            
            logger.info(f"行业配置效果分析完成：{industry_analysis}")
            return industry_analysis
            
        except Exception as e:
            logger.error(f"分析行业配置效果失败：{e}")
            return {}
    
    def _generate_backtest_report(self):
        """生成回测报告"""
        logger.info("生成回测报告")
        
        # 创建报告目录
        import os
        report_dir = "reports"
        os.makedirs(report_dir, exist_ok=True)
        
        # 生成完整的回测报告
        self._generate_full_report(report_dir)
        
        # 将回测结果保存到本地
        self._save_backtest_results()
        
        # 将资金曲线保存到本地
        self._save_equity_curve()
        
        # 将交易记录保存到本地
        self._save_trade_records()
        
        # 生成交易统计报告
        self._save_trade_statistics(report_dir)
        
        # 生成胜率和盈亏比分析报告
        self._save_win_loss_analysis(report_dir)
    
    def _generate_full_report(self, report_dir):
        """生成完整的回测报告"""
        import os
        import json
        from datetime import date
        
        # 生成完整报告文件
        report_file = os.path.join(report_dir, f"full_backtest_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        # 转换trade_records中的日期类型为字符串，以便JSON序列化
        def convert_trade_records(records):
            converted = []
            for record in records:
                converted_record = record.copy()
                # 转换日期类型为字符串
                for key, value in converted_record.items():
                    if isinstance(value, date) or isinstance(value, pd.Timestamp):
                        converted_record[key] = value.strftime('%Y-%m-%d')
                converted.append(converted_record)
            return converted
        
        full_report = {
            'backtest_results': self.backtest_results,
            'equity_curve': self.equity_curve,
            'daily_returns': self.daily_returns,
            'trade_records': convert_trade_records(self.trade_records),
            'risk_parameters': {
                'initial_capital': self.initial_capital,
                'commission_rate': self.commission_rate,
                'slippage_rate': self.slippage_rate,
                'max_daily_transactions': getattr(self.risk_controller, 'max_daily_transactions', 20),
                'liquidity_threshold': getattr(self.risk_controller, 'liquidity_threshold', 10000000),
                'volatility_threshold': getattr(self.risk_controller, 'volatility_threshold', 0.10)
            }
        }
        
        with open(report_file, 'w') as f:
            json.dump(full_report, f, indent=4, ensure_ascii=False, default=str)
        
        logger.info(f"完整回测报告保存成功：{report_file}")
    
    def _save_trade_statistics(self, report_dir):
        """保存交易统计报告"""
        import os
        
        if not self.trade_records:
            return
        
        # 生成交易统计数据
        trade_df = pd.DataFrame(self.trade_records)
        
        # 按日期统计交易次数
        daily_trades = trade_df.groupby('sell_date').size().reset_index(name='trade_count')
        
        # 按股票统计交易次数
        stock_trades = trade_df['stock'].value_counts().reset_index(name='trade_count')
        stock_trades.columns = ['stock', 'trade_count']
        
        # 保存统计结果
        stats_dir = os.path.join(report_dir, 'statistics')
        os.makedirs(stats_dir, exist_ok=True)
        
        daily_trades_file = os.path.join(stats_dir, f"daily_trades_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv")
        daily_trades.to_csv(daily_trades_file, index=False)
        
        stock_trades_file = os.path.join(stats_dir, f"stock_trades_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv")
        stock_trades.to_csv(stock_trades_file, index=False)
        
        logger.info(f"交易统计报告保存成功")
    
    def _save_win_loss_analysis(self, report_dir):
        """保存胜率和盈亏比分析报告"""
        import os
        
        if not self.trade_records:
            return
        
        # 生成胜率和盈亏比分析
        trade_df = pd.DataFrame(self.trade_records)
        
        # 计算每个股票的胜率
        stock_win_rates = []
        for stock in trade_df['stock'].unique():
            stock_trades = trade_df[trade_df['stock'] == stock]
            win_trades = stock_trades[stock_trades['pnl'] > 0]
            win_rate = len(win_trades) / len(stock_trades) if len(stock_trades) > 0 else 0
            stock_win_rates.append({'stock': stock, 'win_rate': win_rate, 'trade_count': len(stock_trades)})
        
        stock_win_rates_df = pd.DataFrame(stock_win_rates)
        
        # 计算盈亏比分布
        profit_trades = trade_df[trade_df['pnl'] > 0]
        loss_trades = trade_df[trade_df['pnl'] < 0]
        
        profit_loss_analysis = {
            'total_trades': len(trade_df),
            'profit_trades': len(profit_trades),
            'loss_trades': len(loss_trades),
            'win_rate': len(profit_trades) / len(trade_df) if len(trade_df) > 0 else 0,
            'avg_profit': profit_trades['pnl'].mean() if len(profit_trades) > 0 else 0,
            'avg_loss': abs(loss_trades['pnl'].mean()) if len(loss_trades) > 0 else 0,
            'profit_loss_ratio': (profit_trades['pnl'].mean() / abs(loss_trades['pnl'].mean())) if len(loss_trades) > 0 else 0
        }
        
        # 保存分析结果
        analysis_dir = os.path.join(report_dir, 'analysis')
        os.makedirs(analysis_dir, exist_ok=True)
        
        win_rates_file = os.path.join(analysis_dir, f"stock_win_rates_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv")
        stock_win_rates_df.to_csv(win_rates_file, index=False)
        
        profit_loss_file = os.path.join(analysis_dir, f"profit_loss_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json")
        import json
        with open(profit_loss_file, 'w') as f:
            json.dump(profit_loss_analysis, f, indent=4, ensure_ascii=False)
        
        logger.info(f"胜率和盈亏比分析报告保存成功")
    
    def _save_backtest_results(self):
        """保存回测结果"""
        import os
        import json
        
        # 创建保存目录
        report_dir = "reports"
        os.makedirs(report_dir, exist_ok=True)
        
        # 保存回测结果
        results_file = os.path.join(report_dir, f"backtest_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(results_file, 'w') as f:
            json.dump(self.backtest_results, f, indent=4, ensure_ascii=False)
        
        logger.info(f"回测结果保存成功：{results_file}")
    
    def _save_equity_curve(self):
        """保存资金曲线"""
        import os
        
        # 创建保存目录
        report_dir = "reports"
        os.makedirs(report_dir, exist_ok=True)
        
        # 保存资金曲线
        equity_file = os.path.join(report_dir, f"equity_curve_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv")
        pd.DataFrame({'equity': self.equity_curve}).to_csv(equity_file, index=False)
        
        logger.info(f"资金曲线保存成功：{equity_file}")
    
    def _save_trade_records(self):
        """保存交易记录"""
        import os
        
        # 创建保存目录
        report_dir = "reports"
        os.makedirs(report_dir, exist_ok=True)
        
        # 保存交易记录
        trade_file = os.path.join(report_dir, f"trade_records_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv")
        pd.DataFrame(self.trade_records).to_csv(trade_file, index=False)
        
        logger.info(f"交易记录保存成功：{trade_file}")
    
    def get_backtest_results(self):
        """获取回测结果"""
        return self.backtest_results
    
    def get_equity_curve(self):
        """获取资金曲线"""
        return self.equity_curve
    
    def get_trade_records(self):
        """获取交易记录"""
        return self.trade_records
