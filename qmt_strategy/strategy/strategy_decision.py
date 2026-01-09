#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
策略决策模块
"""

import pandas as pd
import numpy as np
from qmt_strategy.utils.logger import setup_logger
import qmt_strategy.config as config

logger = setup_logger()

class StrategyDecision:
    """策略决策类，用于基于模型预测和市场数据做出交易决策"""
    
    def __init__(self):
        """初始化策略决策器"""
        logger.info("初始化策略决策器")
        
        # 初始化持仓信息
        self.holdings = {}
        
        # 初始化交易记录
        self.trade_records = []
    
    def _calculate_stock_score(self, stock, row, probability):
        """计算股票综合评分"""
        # 1. 模型预测概率权重 (40%)
        probability_score = probability * 40
        
        # 2. 竞价强度评分 (30%)
        bid_intensity = getattr(row, 'bid_intensity', 0) if hasattr(row, 'bid_intensity') else 0
        bid_intensity_score = min(abs(bid_intensity) * 30, 30)
        
        # 3. 基本面评分 (20%)
        fundamental_score = 0
        # 市值规模评分（适中市值更好）
        if hasattr(row, 'circulating_market_cap'):
            market_cap = row.circulating_market_cap
            # 50-200亿之间的市值得分较高
            if 5000000000 <= market_cap <= 20000000000:
                fundamental_score += 10
            elif 1000000000 <= market_cap < 5000000000 or 20000000000 < market_cap <= 50000000000:
                fundamental_score += 5
        
        # 市盈率评分（合理市盈率更好）
        if hasattr(row, 'pe_ratio'):
            pe = row.pe_ratio
            if 10 <= pe <= 30:
                fundamental_score += 10
            elif 5 <= pe < 10 or 30 < pe <= 50:
                fundamental_score += 5
        
        # 4. 技术面评分 (10%)
        technical_score = 0
        if hasattr(row, 'trend_5d'):
            trend_5d = row.trend_5d
            if trend_5d > 0:
                technical_score += 5
        
        if hasattr(row, 'bullish_ma_arrangement'):
            if row.bullish_ma_arrangement == 1:
                technical_score += 5
        
        # 5. 板块热度评分 (10%，额外加分)
        sector_score = 0
        if hasattr(row, 'sector_change'):
            sector_change = row.sector_change
            if sector_change > 0:
                sector_score = min(sector_change * 10, 10)
        
        # 总分计算
        total_score = probability_score + bid_intensity_score + fundamental_score + technical_score + sector_score
        
        return total_score
    
    def make_decisions(self, features, predictions=None):
        """做出交易决策"""
        logger.info("开始做出交易决策")
        
        if features is None or features.empty:
            logger.warning("特征数据为空，跳过交易决策")
            return None
        
        try:
            # 动态买点决策
            buy_decisions = self._make_buy_decisions(features, predictions)
            
            # 自适应卖点决策
            sell_decisions = self._make_sell_decisions(features)
            
            # 合并买卖决策
            all_decisions = buy_decisions + sell_decisions
            
            logger.info(f"交易决策完成，共{len(all_decisions)}个决策")
            return all_decisions
            
        except Exception as e:
            logger.error(f"做出交易决策失败：{e}")
            return None
    
    def _make_buy_decisions(self, features, predictions=None):
        """动态买点决策"""
        logger.info("进行动态买点决策")
        
        buy_decisions = []
        
        if predictions is None:
            # 如果没有提供预测结果，则假设预测结果包含在特征数据中
            if 'prediction' in features.columns and 'probability' in features.columns:
                predictions = {
                    'predictions': features['prediction'].values,
                    'probabilities': features['probability'].values
                }
            else:
                logger.warning("没有提供预测结果，跳过买点决策")
                return buy_decisions
        
        # 获取当前时间
        current_time = pd.Timestamp.now()
        
        # 确保特征数据和预测结果长度一致
        if len(features) != len(predictions['probabilities']):
            logger.error(f"特征数据长度({len(features)})与预测结果长度({len(predictions['probabilities'])})不一致")
            return buy_decisions
        
        # 计算所有股票的综合评分
        stock_scores = []
        for idx, row in enumerate(features.itertuples()):
            try:
                # 获取股票代码
                stock = row.stock_code if hasattr(row, 'stock_code') else getattr(row, '_1', f'stock_{idx}')
                
                # 获取模型预测概率
                probability = predictions['probabilities'][idx]
                prediction = predictions['predictions'][idx]
                
                # 计算综合评分
                total_score = self._calculate_stock_score(stock, row, probability)
                
                stock_scores.append((stock, row, probability, prediction, total_score))
            except Exception as e:
                logger.error(f"处理股票时发生错误：{e}")
                continue
        
        # 按综合评分降序排序
        stock_scores.sort(key=lambda x: x[4], reverse=True)
        
        # 选择评分最高的前N只股票（最多5只）
        top_stocks = stock_scores[:5]
        
        # 遍历排序后的股票，做出买点决策
        for stock, row, probability, prediction, total_score in top_stocks:
            try:
                logger.debug(f"处理股票：{stock}，预测结果：{prediction}，预测概率：{probability}，综合评分：{total_score}")
                
                # 检查模型预测概率是否满足买入条件
                if probability < config.BUY_THRESHOLD:
                    logger.debug(f"股票{stock}不满足买入条件，概率：{probability} < 阈值：{config.BUY_THRESHOLD}")
                    continue
                
                # 获取买入价格
                buy_price = row.bid_price_925 if hasattr(row, 'bid_price_925') else row.open if hasattr(row, 'open') else 0
                
                # 做出买入决策
                buy_decision = {
                    'stock': stock,
                    'action': 'buy',
                    'price': buy_price,
                    'probability': probability,
                    'total_score': total_score,
                    'time': current_time
                }
                buy_decisions.append(buy_decision)
                
                logger.info(f"买入决策：{buy_decision}")
            except Exception as e:
                logger.error(f"处理股票{stock}时发生错误：{e}")
                continue
        
        return buy_decisions
    
    def _make_sell_decisions(self, features):
        """自适应卖点决策"""
        logger.info("进行自适应卖点决策")
        
        sell_decisions = []
        
        # 获取当前时间
        current_time = pd.Timestamp.now()
        
        # 遍历当前持仓，做出卖点决策
        for stock, holding in self.holdings.items():
            # 查找当前股票对应的行
            stock_rows = features[features['stock_code'] == stock] if 'stock_code' in features.columns else None
            if stock_rows is None or stock_rows.empty:
                continue
            
            row = stock_rows.iloc[0]
            
            # 获取持仓信息
            buy_price = holding['buy_price']
            current_price = row.get('close', 0) if current_time.hour >= 15 else row.get('last_price', 0)
            
            # 计算收益率
            return_rate = (current_price - buy_price) / buy_price
            
            # 检查是否达到目标收益率
            if return_rate >= config.PROFIT_TARGET:
                # 达到目标收益率，止盈卖出
                sell_decision = {
                    'stock': stock,
                    'action': 'sell',
                    'price': current_price,
                    'reason': 'profit_target_reached',
                    'return_rate': return_rate,
                    'time': current_time
                }
                sell_decisions.append(sell_decision)
                
                logger.info(f"卖出决策：{sell_decision}")
                continue
            
            # 检查是否达到止损条件
            if return_rate <= -config.STOP_LOSS_RATIO:
                # 达到止损比例，止损卖出
                sell_decisions.append({
                    'stock': stock,
                    'action': 'sell',
                    'price': current_price,
                    'reason': 'stop_loss_reached',
                    'return_rate': return_rate,
                    'time': current_time
                })
                
                logger.info(f"卖出决策：{sell_decision}")
                continue
            
            # 检查是否达到跟踪止损条件
            if 'highest_price' in holding:
                highest_price = holding['highest_price']
                if (highest_price - current_price) / highest_price >= config.TRAILING_STOP_RATIO:
                    # 达到跟踪止损比例，卖出
                    sell_decisions.append({
                        'stock': stock,
                        'action': 'sell',
                        'price': current_price,
                        'reason': 'trailing_stop_reached',
                        'return_rate': return_rate,
                        'time': current_time
                    })
                    
                    logger.info(f"卖出决策：{sell_decision}")
                    continue
            
            # 更新持仓中的最高价
            if current_price > holding.get('highest_price', 0):
                self.holdings[stock]['highest_price'] = current_price
        
        return sell_decisions
    
    def update_holdings(self, trade_results):
        """更新持仓信息"""
        logger.info("更新持仓信息")
        
        if trade_results is None or not trade_results:
            logger.warning("交易结果为空，跳过更新持仓信息")
            return
        
        for result in trade_results:
            stock = result['stock']
            action = result['action']
            price = result['price']
            quantity = result['quantity']
            time = result['time']
            
            if action == 'buy':
                # 买入股票，更新持仓
                if stock in self.holdings:
                    # 如果已经持有该股票，则增加持仓数量和平均买入价格
                    current_quantity = self.holdings[stock]['quantity']
                    current_avg_price = self.holdings[stock]['buy_price']
                    total_cost = current_quantity * current_avg_price + quantity * price
                    new_quantity = current_quantity + quantity
                    new_avg_price = total_cost / new_quantity
                    
                    self.holdings[stock].update({
                        'quantity': new_quantity,
                        'buy_price': new_avg_price,
                        'highest_price': max(self.holdings[stock]['highest_price'], price)
                    })
                else:
                    # 如果没有持有该股票，则新建持仓记录
                    self.holdings[stock] = {
                        'quantity': quantity,
                        'buy_price': price,
                        'highest_price': price,
                        'buy_time': time
                    }
            
            elif action == 'sell':
                # 卖出股票，更新持仓
                if stock in self.holdings:
                    if quantity >= self.holdings[stock]['quantity']:
                        # 全部卖出，移除持仓记录
                        del self.holdings[stock]
                    else:
                        # 部分卖出，减少持仓数量
                        self.holdings[stock]['quantity'] -= quantity
            
            # 记录交易
            self.trade_records.append(result)
    
    def get_holdings(self):
        """获取当前持仓"""
        return self.holdings
    
    def get_trade_records(self):
        """获取交易记录"""
        return self.trade_records
    
    def _calculate_bid_strength(self, row):
        """计算竞价强度"""
        # 竞价强度 = (9:25竞价价格 - 前收盘价) / 前收盘价
        prev_close = row.get('prev_close', 0)
        bid_price_925 = row.get('bid_price_925', 0)
        
        if prev_close == 0:
            return 0
        
        return (bid_price_925 - prev_close) / prev_close
    
    def _calculate_market_sentiment(self, row):
        """计算市场情绪"""
        # 市场情绪 = 涨跌家数比 * 0.5 + 赚钱效应 * 0.5
        up_down_ratio = row.get('up_down_ratio', 0)
        profit_effect = row.get('profit_effect', 0)
        
        return up_down_ratio * 0.5 + profit_effect * 0.5
