#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
回测模块
"""

import pandas as pd
import numpy as np
from utils.logger import setup_logger
import config
from utils.tools import calculate_sharpe_ratio, calculate_max_drawdown

logger = setup_logger()

class Backtester:
    """回测类，用于进行历史回测和策略评估"""
    
    def __init__(self):
        """初始化回测器"""
        logger.info("初始化回测器")
        
        # 初始化回测参数
        self.initial_capital = config.INITIAL_CAPITAL  # 初始资金
        self.commission_rate = config.COMMISSION_RATE  # 佣金率
        self.slippage_rate = config.SLIPPAGE_RATE  # 滑点率
        
        # 初始化策略模块
        from strategy.strategy_decision import StrategyDecision
        from strategy.model_train import ModelTrainer
        from strategy.risk_control import RiskController
        
        self.strategy_decision = StrategyDecision()  # 策略决策器
        self.model_trainer = ModelTrainer()  # 模型训练器
        self.risk_controller = RiskController(initial_capital=self.initial_capital)  # 风险控制器
        
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
        
        if features is None or features.empty:
            logger.warning("特征数据为空，跳过回测")
            return self.backtest_results
        
        logger.info(f"输入特征数据总行数：{len(features)}")
        logger.info(f"输入特征数据列：{list(features.columns)}")
        
        # 检查特征数据中是否包含label列
        if 'label' not in features.columns:
            logger.error("特征数据中没有找到label列，无法进行回测")
            return self.backtest_results
        
        logger.info(f"特征数据中label列的分布：{features['label'].value_counts().to_dict()}")
        
        try:
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
            
            # 检查是否需要训练模型
            if self.need_train_model or self.model_trainer.model is None:
                logger.info("开始训练模型用于回测")
                
                # 训练简单的随机森林模型
                from sklearn.ensemble import RandomForestClassifier
                simple_model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1,
                    class_weight='balanced'  # 处理不平衡数据
                )
                
                # 训练模型
                simple_model.fit(train_features, labels)
                
                # 更新模型训练器的模型
                self.model_trainer.model = simple_model
                logger.info("模型训练完成")
            else:
                logger.info("使用已加载的模型进行回测")
                
                # 检查模型是否能正常预测
                try:
                    test_pred = self.model_trainer.model.predict(train_features.iloc[:5])
                    logger.info(f"模型测试预测结果：{test_pred}")
                except Exception as e:
                    logger.error(f"模型预测测试失败，重新训练模型：{e}")
                    from sklearn.ensemble import RandomForestClassifier
                    simple_model = RandomForestClassifier(
                        n_estimators=100,
                        max_depth=10,
                        random_state=42,
                        n_jobs=-1,
                        class_weight='balanced'
                    )
                    simple_model.fit(train_features, labels)
                    self.model_trainer.model = simple_model
                    logger.info("重新训练模型完成")
            
            # 模拟交易过程
            self._simulate_trading(features)
            logger.info("模拟交易完成")
            
            # 计算回测指标
            self._calculate_backtest_metrics()
            logger.info(f"回测指标计算完成，结果：{self.backtest_results}")
            
            # 生成回测报告
            self._generate_backtest_report()
            logger.info("回测报告生成完成")
            
            return self.backtest_results
            
        except Exception as e:
            logger.error(f"回测失败：{e}")
            import traceback
            logger.error(f"异常堆栈：{traceback.format_exc()}")
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
        if 'date' in features.columns:
            logger.info(f"日期列存在，日期范围：{features['date'].min()} 到 {features['date'].max()}")
            logger.info(f"唯一日期数量：{features['date'].nunique()}")
            
            # 按日期分组处理数据
            grouped = features.groupby('date')
            logger.info(f"按日期分组后得到 {len(grouped)} 个日期组")
        else:
            # 否则假设整个数据是同一天的
            logger.warning("特征数据中没有日期列，使用当前日期")
            grouped = [(pd.Timestamp.now().date(), features)]
        
        for date, daily_features in grouped:
            logger.info(f"处理日期：{date}，当日数据行数：{len(daily_features)}")
            
            # 检查当日数据是否包含股票代码
            if 'stock_code' in daily_features.columns:
                stocks = daily_features['stock_code'].unique()
                logger.info(f"当日包含 {len(stocks)} 只股票：{stocks[:5]}...")
            
            # 模拟每日交易
            self._simulate_daily_trading(date, daily_features)
            
            # 更新每日资金曲线
            self._update_equity_curve(date, daily_features)
    
    def _simulate_daily_trading(self, date, daily_features):
        """模拟每日交易，使用真实策略决策过程"""
        logger.info(f"模拟{date}的交易")
        
        # 1. 使用模型进行预测
        predictions = self._predict_stocks(daily_features)
        
        if predictions is None:
            logger.warning(f"{date}没有生成预测结果，跳过交易")
            return
        
        logger.info(f"预测结果类型：{type(predictions)}")
        logger.info(f"预测结果内容：{predictions}")
        logger.info(f"预测结果长度：{len(predictions['predictions']) if isinstance(predictions, dict) and 'predictions' in predictions else 'N/A'}")
        
        # 2. 调用策略决策模块生成买卖决策
        decisions = self.strategy_decision.make_decisions(daily_features, predictions)
        
        if decisions is None:
            logger.warning(f"{date}没有生成交易决策，跳过交易")
            return
        
        logger.info(f"{date}生成了{len(decisions)}个交易决策")
        for i, decision in enumerate(decisions[:3]):
            logger.info(f"  决策{i+1}：{decision}")
        
        # 3. 调用风险控制模块检查决策
        approved_decisions = self.risk_controller.check(decisions)
        
        logger.info(f"{date}通过风险控制的决策数量：{len(approved_decisions)}")
        if not approved_decisions:
            logger.info(f"{date}没有通过风险控制的交易决策")
            return
        
        # 4. 执行交易
        trade_results = self._execute_decisions(date, approved_decisions, daily_features)
        
        logger.info(f"{date}执行了{len(trade_results)}笔交易")
        for i, result in enumerate(trade_results[:3]):
            logger.info(f"  交易{i+1}：{result}")
        
        # 5. 更新风险控制状态
        self.risk_controller.update_risk_status(trade_results)
        
        # 6. 更新策略决策器的持仓
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
        
        # 使用模型进行预测
        return self.model_trainer.predict(numeric_features)
    
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
        """模拟买入股票"""
        try:
            # 获取买入价格（使用竞价结束价格或开盘价）
            stock_data = daily_features[daily_features['stock_code'] == stock]
            if stock_data.empty:
                logger.warning(f"{date} {stock} 没有找到对应的数据，跳过交易")
                return None
            row = stock_data.iloc[0]
            price = decision.get('price', row.get('bid_price_925', row.get('open', row.get('close', 0))))
            
            if price <= 0:
                logger.warning(f"{date} {stock} 买入价格无效：{price}")
                return None
            
            # 计算买入数量（根据总资金和仓位限制）
            buy_amount = self.current_capital * 0.01  # 每次买入总资金的1%
            quantity = int(buy_amount / price / 100) * 100  # 按100股整数倍
            quantity = max(quantity, 100)  # 至少买入100股
            
            if quantity <= 0:
                logger.warning(f"{date} {stock} 买入数量无效：{quantity}")
                return None
            
            # 计算买入成本（考虑滑点和佣金）
            actual_price = price * (1 + self.slippage_rate)
            buy_value = actual_price * quantity
            commission = buy_value * self.commission_rate
            total_cost = buy_value + commission
            
            if total_cost > self.current_capital:
                logger.warning(f"{date} {stock} 可用资金不足，需要{total_cost:.2f}，但只有{self.current_capital:.2f}")
                return None
            
            # 更新持仓
            if stock in self.holdings:
                # 已有持仓，增加数量和更新平均价格
                current_quantity = self.holdings[stock]['quantity']
                current_avg_price = self.holdings[stock]['buy_price']
                total_buy_value = current_quantity * current_avg_price + buy_value
                new_quantity = current_quantity + quantity
                new_avg_price = total_buy_value / new_quantity
                
                self.holdings[stock].update({
                    'quantity': new_quantity,
                    'buy_price': new_avg_price,
                    'highest_price': max(self.holdings[stock]['highest_price'], actual_price)
                })
            else:
                # 新建持仓
                self.holdings[stock] = {
                    'quantity': quantity,
                    'buy_price': actual_price,
                    'buy_date': date,
                    'highest_price': actual_price
                }
            
            # 扣除资金
            self.current_capital -= total_cost
            
            # 记录交易结果
            trade_result = {
                'stock': stock,
                'action': 'buy',
                'price': actual_price,
                'quantity': quantity,
                'total_cost': total_cost,
                'pnl': 0.0,
                'time': pd.Timestamp(date)
            }
            
            logger.info(f"{date} 买入 {stock}，价格：{actual_price:.2f}，数量：{quantity}，成本：{total_cost:.2f}")
            return trade_result
            
        except Exception as e:
            logger.error(f"{date} 模拟买入 {stock} 失败：{e}")
            return None
    
    def _simulate_sell(self, date, stock, decision, daily_features):
        """模拟卖出股票"""
        try:
            if stock not in self.holdings:
                logger.warning(f"{date} {stock} 不在持仓中，无法卖出")
                return None
            
            # 获取卖出价格（使用收盘价）
            holding = self.holdings[stock]
            stock_data = daily_features[daily_features['stock_code'] == stock]
            if stock_data.empty:
                logger.warning(f"{date} {stock} 没有找到对应的数据，跳过交易")
                return None
            row = stock_data.iloc[0]
            price = row.get('close', 0)
            
            if price <= 0:
                logger.warning(f"{date} {stock} 卖出价格无效：{price}")
                return None
            
            # 计算卖出数量（全部卖出）
            quantity = holding['quantity']
            
            # 计算卖出收入（考虑滑点和佣金）
            actual_price = price * (1 - self.slippage_rate)
            sell_value = actual_price * quantity
            commission = sell_value * self.commission_rate
            total_revenue = sell_value - commission
            
            # 计算盈亏
            cost = holding['buy_price'] * quantity
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
                'quantity': quantity,
                'pnl': pnl,
                'return_rate': (actual_price - holding['buy_price']) / holding['buy_price']
            }
            self.trade_records.append(trade_record)
            
            # 记录交易结果
            trade_result = {
                'stock': stock,
                'action': 'sell',
                'price': actual_price,
                'quantity': quantity,
                'total_revenue': total_revenue,
                'pnl': pnl,
                'time': pd.Timestamp(date)
            }
            
            # 删除持仓
            del self.holdings[stock]
            
            logger.info(f"{date} 卖出 {stock}，价格：{actual_price:.2f}，数量：{quantity}，收入：{total_revenue:.2f}，盈亏：{pnl:.2f}")
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
        if self.backtest_results.get('annual_return', 0) != 0 and self.backtest_results.get('max_drawdown', 0) != 0:
            calmar_ratio = self.backtest_results['annual_return'] / abs(self.backtest_results['max_drawdown'])
            self.backtest_results['calmar_ratio'] = calmar_ratio
        
        # 计算Sortino比率
        if self.daily_returns:
            negative_returns = [r for r in self.daily_returns if r < 0]
            if negative_returns:
                downside_volatility = np.std(negative_returns)
                if downside_volatility > 0:
                    sortino_ratio = self.backtest_results.get('annual_return', 0) / (downside_volatility * np.sqrt(252))
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
            win_rate = len(profit_trades) / len(self.trade_records)
            self.backtest_results['win_rate'] = win_rate
            
            # 计算盈亏比
            if loss_trades:
                avg_profit = sum(trade['pnl'] for trade in profit_trades) / len(profit_trades) if profit_trades else 0
                avg_loss = abs(sum(trade['pnl'] for trade in loss_trades) / len(loss_trades)) if loss_trades else 1
                profit_loss_ratio = avg_profit / avg_loss if avg_loss > 0 else 0
                self.backtest_results['profit_loss_ratio'] = profit_loss_ratio
            
            # 计算总盈亏
            total_pnl = sum(trade['pnl'] for trade in self.trade_records)
            self.backtest_results['total_pnl'] = total_pnl
            
            # 计算平均每笔交易盈亏
            avg_pnl_per_trade = total_pnl / len(self.trade_records)
            self.backtest_results['avg_pnl_per_trade'] = avg_pnl_per_trade
            
            # 计算最大连续盈利/亏损
            if self.trade_records:
                consecutive_wins = 0
                max_consecutive_wins = 0
                consecutive_losses = 0
                max_consecutive_losses = 0
                
                for trade in self.trade_records:
                    if trade['pnl'] > 0:
                        consecutive_wins += 1
                        consecutive_losses = 0
                        if consecutive_wins > max_consecutive_wins:
                            max_consecutive_wins = consecutive_wins
                    elif trade['pnl'] < 0:
                        consecutive_losses += 1
                        consecutive_wins = 0
                        if consecutive_losses > max_consecutive_losses:
                            max_consecutive_losses = consecutive_losses
                
                self.backtest_results['max_consecutive_wins'] = max_consecutive_wins
                self.backtest_results['max_consecutive_losses'] = max_consecutive_losses
            
            # 计算平均持仓天数
            holding_days = [pd.Timestamp(trade['sell_date']) - pd.Timestamp(trade['buy_date']) for trade in self.trade_records]
            avg_holding_days = np.mean([hd.days for hd in holding_days]) if holding_days else 0
            self.backtest_results['avg_holding_days'] = avg_holding_days
            
            # 计算每日平均交易次数
            if n_days > 0:
                avg_daily_trades = len(self.trade_records) / n_days
                self.backtest_results['avg_daily_trades'] = avg_daily_trades
    
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
        
        # 生成完整报告文件
        report_file = os.path.join(report_dir, f"full_backtest_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        full_report = {
            'backtest_results': self.backtest_results,
            'equity_curve': self.equity_curve,
            'daily_returns': self.daily_returns,
            'trade_records': self.trade_records,
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
            json.dump(full_report, f, indent=4, ensure_ascii=False)
        
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
