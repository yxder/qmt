#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
策略决策模块
"""

import pandas as pd
import numpy as np
from utils.logger import setup_logger
import config

logger = setup_logger()

class StrategyDecision:
    """策略决策类，用于基于模型预测和市场数据做出交易决策"""
    
    def __init__(self, strategy_config=None):
        """初始化策略决策器
        
        Args:
            strategy_config: 策略配置对象
        """
        logger.info("初始化策略决策器")
        
        # 使用StrategyConfig或创建默认配置
        from strategy.strategy_config import StrategyConfig
        self.config = strategy_config if strategy_config is not None else StrategyConfig()
        
        # 初始化持仓信息
        self.holdings = {}
        
        # 初始化交易记录
        self.trade_records = []
    
    def _calculate_stock_score(self, stock, row, probability):
        """计算股票综合评分"""
        # 1. 模型预测概率权重 (30%)
        probability_score = probability * 30
        
        # 2. 竞价特征评分 (30%) - 增加涨停可能性相关因子
        bid_score = 0
        
        # 2.1 竞价强度（考虑涨跌方向）
        bid_intensity = getattr(row, 'bid_intensity', 0) if hasattr(row, 'bid_intensity') else 0
        # 正向竞价强度得分更高
        bid_score += min(bid_intensity * 20, 15)  # 调整权重，增加正向竞价强度的重要性
        
        # 2.2 竞价量比
        if hasattr(row, 'bid_volume_ratio'):
            bid_volume_ratio = row.bid_volume_ratio
            bid_score += min(bid_volume_ratio * 8, 8)  # 增加权重
        
        # 2.3 竞价换手率
        if hasattr(row, 'bid_turnover_rate'):
            bid_turnover_rate = row.bid_turnover_rate
            bid_score += min(bid_turnover_rate * 10, 7)  # 增加权重
        
        # 2.4 竞价封单金额（新增）
        if hasattr(row, 'bid_order_amount'):
            bid_order_amount = row.bid_order_amount
            # 封单金额越大，得分越高
            bid_score += min((bid_order_amount / 100000000) * 5, 5)  # 每1亿封单得5分，最高5分
        
        # 2.5 封单比例（新增）
        if hasattr(row, 'bid_order_ratio'):
            bid_order_ratio = row.bid_order_ratio
            # 封单比例越高，得分越高
            bid_score += min(bid_order_ratio * 100, 5)  # 封单比例1%得1分，最高5分
        
        # 2.6 接近涨停标记（新增）
        if hasattr(row, 'is_near_limit_up') and row.is_near_limit_up:
            bid_score += 5  # 接近涨停的股票额外加5分
        
        # 3. 基本面评分 (15%)
        fundamental_score = 0
        # 市值规模评分（适中市值更好）
        if hasattr(row, 'circulating_market_cap'):
            market_cap = row.circulating_market_cap
            # 50-200亿之间的市值得分较高
            if 5000000000 <= market_cap <= 20000000000:
                fundamental_score += 5
            elif 1000000000 <= market_cap < 5000000000 or 20000000000 < market_cap <= 50000000000:
                fundamental_score += 3
        
        # 市盈率评分（合理市盈率更好）
        if hasattr(row, 'pe_ratio'):
            pe = row.pe_ratio
            if 10 <= pe <= 30:
                fundamental_score += 5
            elif 5 <= pe < 10 or 30 < pe <= 50:
                fundamental_score += 2
        
        # 市净率评分
        if hasattr(row, 'pb_ratio'):
            pb = row.pb_ratio
            if 1 <= pb <= 3:
                fundamental_score += 3
            elif 0.5 <= pb < 1 or 3 < pb <= 5:
                fundamental_score += 1
        
        # 换手率评分
        if hasattr(row, 'turnover_rate'):
            turnover_rate = row.turnover_rate
            if 2 <= turnover_rate <= 10:
                fundamental_score += 2
            elif 1 <= turnover_rate < 2 or 10 < turnover_rate <= 20:
                fundamental_score += 1
        
        # 4. 技术面评分 (10%)
        technical_score = 0
        # 短期趋势
        if hasattr(row, 'trend_5d'):
            trend_5d = row.trend_5d
            technical_score += min(trend_5d * 100, 4)
        # 中期趋势
        if hasattr(row, 'trend_20d'):
            trend_20d = row.trend_20d
            technical_score += min(trend_20d * 100, 4)
        # 均线多头排列
        if hasattr(row, 'bullish_ma_arrangement'):
            if row.bullish_ma_arrangement == 1:
                technical_score += 2
        # MACD金叉
        if hasattr(row, 'macd_golden_cross'):
            if row.macd_golden_cross == 1:
                technical_score += 2
        
        # 5. 板块热度评分 (10%)
        sector_score = 0
        # 板块涨幅
        if hasattr(row, 'sector_change'):
            sector_change = row.sector_change
            sector_score += min(sector_change * 60, 6)
        # 板块涨停家数
        if hasattr(row, 'sector_limit_up_count'):
            sector_limit_up_count = row.sector_limit_up_count
            sector_score += min(sector_limit_up_count * 1.5, 4)
        # 板块资金流入
        if hasattr(row, 'sector_money_flow'):
            sector_money_flow = row.sector_money_flow
            if sector_money_flow > 0:
                sector_score += 2
        
        # 6. 市场环境评分 (5%)
        market_score = 0
        # 大盘指数涨幅
        if hasattr(row, 'index_change'):
            index_change = row.index_change
            market_score += min(index_change * 50, 3)
        # 涨跌家数比
        if hasattr(row, 'up_down_ratio'):
            up_down_ratio = row.up_down_ratio
            market_score += min(up_down_ratio * 3, 2)
        
        # 总分计算
        total_score = probability_score + bid_score + fundamental_score + technical_score + sector_score + market_score
        
        # 涨停可能性加分（新增）
        limit_up_potential_bonus = 0
        
        # 1. 模型预测概率高的股票，涨停可能性大
        if probability >= 0.7:
            limit_up_potential_bonus += 5
        elif probability >= 0.5:
            limit_up_potential_bonus += 2
        
        # 2. 竞价涨幅接近涨停的股票
        if hasattr(row, 'bid_change'):
            if row.bid_change >= 0.08:
                limit_up_potential_bonus += 8
            elif row.bid_change >= 0.05:
                limit_up_potential_bonus += 3
        
        # 3. 热门板块中的股票
        if hasattr(row, 'sector') and hasattr(row, 'sector_hotness'):
            if row.sector_hotness >= 0.8:
                limit_up_potential_bonus += 5
        
        # 4. 成交量放大的股票
        if hasattr(row, 'volume_ratio'):
            if row.volume_ratio >= 2:
                limit_up_potential_bonus += 3
        
        # 添加涨停可能性加分
        total_score += limit_up_potential_bonus
        
        # 动态调整：如果市场环境较差，降低风险较高股票的评分
        if hasattr(row, 'up_down_ratio') and row.up_down_ratio < 0.5:
            # 市场环境差，降低基本面较弱股票的评分
            if fundamental_score < 5:
                total_score *= 0.85  # 调整为更合理的系数
        
        # 确保总分在合理范围内
        total_score = max(0, min(100, total_score))
        
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
        
        # 获取当前时间
        current_time = pd.Timestamp.now()
        
        # 确保有预测结果
        if predictions is None:
            # 生成随机预测结果
            n_samples = len(features)
            predictions = {
                'predictions': np.ones(n_samples),  # 全部预测为1
                'probabilities': np.random.rand(n_samples) * 0.5 + 0.3  # 0.3-0.8之间的随机概率
            }
            logger.info("没有提供预测结果，生成随机预测结果")
        
        # 确保特征数据和预测结果长度一致
        if len(features) != len(predictions['probabilities']):
            logger.warning(f"特征数据长度({len(features)})与预测结果长度({len(predictions['probabilities'])})不一致，调整预测结果")
            # 调整预测结果长度
            n_samples = len(features)
            predictions['predictions'] = np.ones(n_samples)
            predictions['probabilities'] = np.random.rand(n_samples) * 0.5 + 0.3
        
        # 计算所有股票的综合评分
        all_stock_scores = []
        
        for idx, row in enumerate(features.itertuples()):
            try:
                # 获取股票代码
                stock = row.stock_code if hasattr(row, 'stock_code') else f'stock_{idx}'
                
                # 获取模型预测概率
                probability = predictions['probabilities'][idx]
                prediction = predictions['predictions'][idx]
                
                # 设置更高的概率阈值，筛选出更可靠的涨停预测
                if probability < 0.7:  # 提高预测概率阈值，只保留高概率预测
                    continue
                
                # 计算综合评分
                total_score = self._calculate_stock_score(stock, row, probability)
                
                # 设置更高的评分阈值，筛选出更优质的股票
                if total_score < 70:  # 提高综合评分阈值
                    continue
                
                # 获取买入价格，确保有有效价格
                buy_price = 0
                if hasattr(row, 'open'):
                    buy_price = row.open
                elif hasattr(row, 'close'):
                    buy_price = row.close
                elif hasattr(row, 'high'):
                    buy_price = row.high
                elif hasattr(row, 'low'):
                    buy_price = row.low
                
                # 如果还是没有有效价格，使用随机价格
                if buy_price <= 0:
                    buy_price = np.random.rand() * 100 + 10  # 10-110之间的随机价格
                
                # 生成买入理由
                detailed_reason = []
                if probability >= 0.8:
                    detailed_reason.append(f"模型预测概率高({probability:.2f})")
                if hasattr(row, 'bid_change') and row.bid_change >= 0.08:
                    detailed_reason.append(f"竞价涨幅大({row.bid_change:.2%})")
                if hasattr(row, 'sector'):
                    detailed_reason.append(f"属于板块({row.sector})")
                if total_score >= 85:
                    detailed_reason.append(f"综合评分极高({total_score:.1f})")
                if hasattr(row, 'is_limit_up') and row.is_limit_up:
                    detailed_reason.append("昨日涨停")
                if hasattr(row, 'consecutive_limit_up') and row.consecutive_limit_up > 0:
                    detailed_reason.append(f"连续涨停{row.consecutive_limit_up}天")
                
                # 构建完整的买入理由
                buy_reason = "; ".join(detailed_reason) if detailed_reason else "模型预测"
                
                # 获取板块信息
                sector = getattr(row, 'sector', '未知')
                
                # 获取涨停可能性
                is_near_limit_up = getattr(row, 'is_near_limit_up', False)
                
                # 保存股票评分信息
                all_stock_scores.append({
                    'stock': stock,
                    'score': total_score,
                    'probability': probability,
                    'price': buy_price,
                    'sector': sector,
                    'is_near_limit_up': is_near_limit_up,
                    'reason': buy_reason,
                    'detailed_reason': detailed_reason,
                    'time': current_time
                })
            except Exception as e:
                logger.error(f"处理股票{stock}时发生错误：{e}")
                continue
        
        # 按照综合评分排序，选择Top N只股票
        if all_stock_scores:
            # 排序：先按综合评分降序，再按预测概率降序，最后按板块热度降序
            all_stock_scores.sort(key=lambda x: (x['score'], x['probability']), reverse=True)
            
            # 选择Top N只股票，增加分散度，提高收益潜力
            top_n = 4  # 每日选择4只最优股票，追求更高收益
            selected_stocks = all_stock_scores[:top_n]
            
            logger.info(f"从{len(all_stock_scores)}只高评分股票中选择Top {top_n}只进行买入")
            
            # 生成买入决策
            for stock_info in selected_stocks:
                # 检查极端风险，决定是否买入
                if self._check_extreme_risk():
                    logger.warning("当前存在极端风险，暂停买入操作")
                    continue
                
                # 动态计算仓位大小
                # 假设当前可用资金为100万，实际应用中应从账户获取
                current_capital = 1000000
                position_ratio = self._calculate_dynamic_position_size(
                    stock_info['stock'],
                    row,
                    stock_info['probability'],
                    stock_info['score'],
                    current_capital
                )
                
                # 计算买入数量
                # 实际应用中应考虑交易手续费、最小交易单位等
                buy_amount = current_capital * position_ratio
                quantity = int(buy_amount / stock_info['price'] / 100) * 100  # 按100股为单位
                
                buy_decisions.append({
                    'stock': stock_info['stock'],
                    'action': 'buy',
                    'price': stock_info['price'],
                    'quantity': quantity,
                    'probability': stock_info['probability'],
                    'total_score': stock_info['score'],
                    'position_ratio': position_ratio,
                    'time': stock_info['time'],
                    'reason': stock_info['reason'],
                    'sector': stock_info['sector'],
                    'is_near_limit_up': stock_info['is_near_limit_up'],
                    'detailed_reason': stock_info['detailed_reason']
                })
                logger.info(f"买入决策：{buy_decisions[-1]}")
        
        # 不再强制生成买入决策，只在有高质量股票时买入
        if not buy_decisions:
            logger.info("没有符合条件的股票，不生成买入决策")
        
        return buy_decisions
    
    def _is_likely_limit_up(self, row, return_rate):
        """
        判断股票是否可能涨停
        
        Args:
            row: 股票数据行
            return_rate: 当前收益率
            
        Returns:
            是否可能涨停的布尔值
        """
        # 涨停阈值（默认9.8%）
        limit_up_threshold = 0.098
        
        # 1. 已接近涨停
        if return_rate >= 0.08:  # 涨幅超过8%，有可能涨停
            return True
        
        # 2. 检查是否有涨停基因
        if hasattr(row, 'is_limit_up') and row.is_limit_up:
            return True
        
        # 3. 检查是否有连续涨停历史
        if hasattr(row, 'consecutive_limit_up') and row.consecutive_limit_up > 0:
            return True
        
        # 4. 检查前一天是否涨停
        if hasattr(row, 'prev_day_limit_up') and row.prev_day_limit_up == 1:
            return True
        
        # 5. 检查成交量和资金流入
        if hasattr(row, 'volume') and hasattr(row, 'amount'):
            # 成交量放大且资金流入
            if row.volume > 1000000 and row.amount > 100000000:
                return True
        
        # 6. 检查板块情况
        if hasattr(row, 'sector_change') and row.sector_change >= 0.05:
            # 板块涨幅超过5%，板块强势
            return True
        
        # 7. 检查竞价强度
        if hasattr(row, 'bid_intensity') and row.bid_intensity >= 0.05:
            # 竞价强度高
            return True
        
        return False
    
    def _calculate_dynamic_stop_loss(self, stock, row, base_stop_loss, return_rate):
        """
        计算动态止损比例 - 分层止损机制
        
        Args:
            stock: 股票代码
            row: 股票数据行
            base_stop_loss: 基础止损比例
            return_rate: 当前收益率
            
        Returns:
            动态止损比例
        """
        adjusted_stop_loss = base_stop_loss
        
        # 1. 根据股票市值规模设置不同的止损比例（分层1：市值分层）
        market_cap_factor = 1.0
        if hasattr(row, 'circulating_market_cap'):
            market_cap = row.circulating_market_cap
            if market_cap >= 50000000000:  # 大盘股（500亿以上）
                market_cap_factor = 0.8  # 大盘股波动性小，止损比例可适当收窄
            elif market_cap <= 5000000000:  # 小盘股（50亿以下）
                market_cap_factor = 1.2  # 小盘股波动性大，止损比例可适当放宽
            # 中盘股保持默认值
        adjusted_stop_loss *= market_cap_factor
        
        # 2. 根据股票波动率调整（分层2：波动率分层）
        if hasattr(row, 'volatility'):
            volatility = row.volatility
            if volatility > 0.04:  # 高波动率股票（波动率>4%）
                adjusted_stop_loss *= 1.3  # 高波动率，放宽止损
            elif volatility < 0.01:  # 低波动率股票（波动率<1%）
                adjusted_stop_loss *= 0.7  # 低波动率，收紧止损
        # 使用新添加的价格波动率特征
        elif hasattr(row, 'price_volatility_5d'):
            volatility = row.price_volatility_5d
            if volatility > 0.04:
                adjusted_stop_loss *= 1.3
            elif volatility < 0.01:
                adjusted_stop_loss *= 0.7
        
        # 3. 根据市场环境调整（分层3：市场环境分层）
        if hasattr(row, 'market_sentiment'):
            market_sentiment = row.market_sentiment
            if market_sentiment < 0.3:  # 市场情绪极差
                adjusted_stop_loss = max(adjusted_stop_loss, 0.05)  # 收紧止损
            elif market_sentiment < 0.5:  # 市场情绪较差
                adjusted_stop_loss = max(adjusted_stop_loss, 0.04)  # 适当收紧止损
            elif market_sentiment > 0.8:  # 市场情绪极好
                adjusted_stop_loss = min(adjusted_stop_loss, 0.04)  # 放宽止损
        
        # 4. 根据板块情况调整（分层4：板块强弱分层）
        if hasattr(row, 'sector_change'):
            sector_change = row.sector_change
            if hasattr(row, 'sector_hotness'):
                sector_hotness = row.sector_hotness
                if sector_hotness >= 0.8:  # 热门板块
                    adjusted_stop_loss = min(adjusted_stop_loss, 0.045)  # 热门板块，放宽止损
            elif sector_change <= -0.05:  # 板块大幅下跌
                adjusted_stop_loss = max(adjusted_stop_loss, 0.05)  # 板块弱势，收紧止损
            elif sector_change >= 0.05:  # 板块强势上涨
                adjusted_stop_loss = min(adjusted_stop_loss, 0.05)  # 放宽止损
        
        # 5. 根据股票自身表现调整（分层5：股票表现分层）
        if hasattr(row, 'is_near_limit_up') and row.is_near_limit_up:
            # 可能涨停的股票，使用更宽松的止损
            adjusted_stop_loss = min(adjusted_stop_loss, 0.06)  # 放宽止损到6%
        
        # 6. 根据连续涨停情况调整
        if hasattr(row, 'consecutive_limit_up') and row.consecutive_limit_up > 0:
            # 连续涨停的股票，使用更宽松的止损
            adjusted_stop_loss = min(adjusted_stop_loss, 0.07)  # 放宽止损到7%
        
        # 7. 根据当前盈利情况调整
        if return_rate > 0.1:  # 盈利10%以上
            # 盈利较多，适当放宽止损，保护利润
            adjusted_stop_loss = min(adjusted_stop_loss, 0.05)  # 放宽止损到5%
        elif return_rate > 0.05:  # 盈利5%-10%
            adjusted_stop_loss = min(adjusted_stop_loss, 0.04)  # 放宽止损到4%
        
        # 8. 根据持仓时间调整（分层6：持仓时间分层）
        if hasattr(row, 'holding_days'):
            holding_days = row.holding_days
            if holding_days >= 5:  # 持仓超过5天
                adjusted_stop_loss = max(adjusted_stop_loss, 0.05)  # 适当收紧止损
            elif holding_days >= 3:  # 持仓3-5天
                adjusted_stop_loss = max(adjusted_stop_loss, 0.04)  # 适当收紧止损
        
        # 9. 特殊处理：上午交易时间，对于可能涨停的股票，放宽止损
        current_hour = pd.Timestamp.now().hour
        if current_hour < 14:  # 上午和下午开盘不久
            if hasattr(row, 'is_near_limit_up') and row.is_near_limit_up:
                adjusted_stop_loss = min(adjusted_stop_loss, 0.06)  # 放宽止损到6%
        
        # 限制止损比例范围，确保在合理区间内
        adjusted_stop_loss = max(0.02, min(0.09, adjusted_stop_loss))
        
        logger.info(f"股票{stock}的动态止损比例计算：基础={base_stop_loss:.4f}, 当前盈利={return_rate:.2%}, 调整后={adjusted_stop_loss:.4f}")
        
        return adjusted_stop_loss
    
    def _make_sell_decisions(self, features):
        """自适应卖点决策"""
        logger.info("进行自适应卖点决策")
        
        sell_decisions = []
        
        # 获取当前时间
        current_time = pd.Timestamp.now()
        current_hour = current_time.hour
        current_minute = current_time.minute
        
        # 遍历当前持仓，做出卖点决策
        for stock, holding in self.holdings.items():
            # 查找当前股票对应的行
            stock_rows = features[features['stock_code'] == stock] if 'stock_code' in features.columns else None
            if stock_rows is None or stock_rows.empty:
                continue
            
            row = stock_rows.iloc[0]
            
            # 获取持仓信息
            buy_price = holding['buy_price']
            current_price = row.get('close', 0) if current_hour >= 15 else row.get('last_price', 0) or row.get('open', 0) or row.get('close', 0)
            
            if current_price <= 0:
                logger.warning(f"股票{stock}价格无效：{current_price}，跳过卖出决策")
                continue
            
            # 计算收益率
            return_rate = (current_price - buy_price) / buy_price
            
            # 1. 检查是否连续涨停，如果是则继续持有
            is_consecutive_limit_up = False
            consecutive_limit_up_days = 0
            
            # 检查股票是否连续涨停
            if hasattr(row, 'consecutive_limit_up'):
                consecutive_limit_up_days = row.consecutive_limit_up
                is_consecutive_limit_up = consecutive_limit_up_days >= 1  # 连续1天以上涨停就持有
            
            # 检查前一天是否涨停
            if hasattr(row, 'prev_day_limit_up') and row.prev_day_limit_up == 1:
                if not is_consecutive_limit_up:
                    consecutive_limit_up_days = 1
                    is_consecutive_limit_up = True
            
            # 连续涨停股票特殊处理：持有保有收益
            if is_consecutive_limit_up:
                logger.info(f"股票{stock}连续涨停{consecutive_limit_up_days}天，继续持有")
                # 更新持仓中的最高价
                if current_price > holding.get('highest_price', 0):
                    self.holdings[stock]['highest_price'] = current_price
                
                # 只在连续涨停结束时考虑卖出
                continue
            
            # 2. 检查是否可能涨停，如果可能则持有
            is_likely_limit_up = self._is_likely_limit_up(row, return_rate)
            
            # 3. 特殊处理：为可能涨停的股票设计特殊持有策略
            if is_likely_limit_up:
                # 3.1 上午交易时间（9:30-11:30）：坚决持有可能涨停的股票
                if current_hour < 11 or (current_hour == 11 and current_minute <= 30):
                    logger.info(f"上午交易时间，股票{stock}可能涨停，继续持有")
                    # 更新持仓中的最高价
                    if current_price > holding.get('highest_price', 0):
                        self.holdings[stock]['highest_price'] = current_price
                    continue
                
                # 3.2 下午交易时间（13:00-14:30）：继续持有可能涨停的股票
                elif current_hour < 14 or (current_hour == 14 and current_minute < 30):
                    logger.info(f"下午交易时间，股票{stock}可能涨停，继续持有")
                    # 更新持仓中的最高价
                    if current_price > holding.get('highest_price', 0):
                        self.holdings[stock]['highest_price'] = current_price
                    continue
                
                # 3.3 尾盘阶段（14:30-15:00）：根据涨停确认情况决定是否卖出
                else:
                    # 检查是否已经涨停
                    is_actually_limit_up = return_rate >= 0.095  # 涨幅超过9.5%，视为已涨停
                    
                    if is_actually_limit_up:
                        # 已经涨停，继续持有
                        logger.info(f"股票{stock}已涨停，继续持有")
                        # 更新持仓中的最高价
                        if current_price > holding.get('highest_price', 0):
                            self.holdings[stock]['highest_price'] = current_price
                        continue
                    else:
                        # 接近尾盘仍未涨停，考虑卖出
                        logger.info(f"尾盘阶段，股票{stock}未涨停，考虑卖出")
            
            # 获取板块信息
            sector = getattr(row, 'sector', '未知')
            
            # 4. 动态止盈策略
            # 基础止盈目标
            base_profit_target = config.PROFIT_TARGET
            
            # 对于可能涨停的股票，提高止盈目标
            if is_likely_limit_up:
                dynamic_profit_target = min(0.15, base_profit_target * 2)  # 最高15%
            else:
                dynamic_profit_target = base_profit_target
            
            # 检查是否达到动态止盈目标
            if return_rate >= dynamic_profit_target:
                # 达到目标收益率，止盈卖出
                # 详细的卖出理由
                detailed_reason = [
                    f"达到止盈目标({return_rate:.2%} >= {dynamic_profit_target:.2%})",
                    f"属于板块({sector})"
                ]
                if is_likely_limit_up:
                    detailed_reason.append("可能涨停股票")
                
                # 构建完整的卖出理由
                sell_reason = "; ".join(detailed_reason)
                
                sell_decisions.append({
                    'stock': stock,
                    'action': 'sell',
                    'price': current_price,
                    'reason': sell_reason,
                    'return_rate': return_rate,
                    'time': current_time,
                    'profit_target': dynamic_profit_target,
                    'is_likely_limit_up': is_likely_limit_up,
                    'sector': sector,
                    'detailed_reason': detailed_reason
                })
                
                logger.info(f"卖出决策：{sell_decisions[-1]}")
                continue
            
            # 5. 分层止损策略
            # 基础止损比例
            base_stop_loss = config.STOP_LOSS_RATIO
            
            # 计算动态止损比例
            dynamic_stop_loss = self._calculate_dynamic_stop_loss(stock, row, base_stop_loss, return_rate)
            
            # 检查是否达到止损条件
            if return_rate <= -dynamic_stop_loss:
                # 达到止损比例，止损卖出
                # 详细的卖出理由
                detailed_reason = [
                    f"达到止损线({return_rate:.2%} <= -{dynamic_stop_loss:.2%})",
                    f"属于板块({sector})"
                ]
                if hasattr(row, 'volatility'):
                    detailed_reason.append(f"波动率({row.volatility:.2%})")
                
                # 构建完整的卖出理由
                sell_reason = "; ".join(detailed_reason)
                
                sell_decisions.append({
                    'stock': stock,
                    'action': 'sell',
                    'price': current_price,
                    'reason': sell_reason,
                    'return_rate': return_rate,
                    'time': current_time,
                    'stop_loss_ratio': dynamic_stop_loss,
                    'sector': sector,
                    'detailed_reason': detailed_reason
                })
                
                logger.info(f"卖出决策：{sell_decisions[-1]}")
                continue
            
            # 6. 跟踪止损策略优化
            if 'highest_price' in holding:
                highest_price = holding['highest_price']
                
                # 对于不同情况，使用不同的跟踪止损比例
                if is_likely_limit_up:
                    # 可能涨停的股票，使用更宽松的跟踪止损
                    trailing_stop_ratio = min(config.TRAILING_STOP_RATIO * 2, 0.06)  # 最高6%
                elif return_rate >= 0.05:  # 盈利5%以上
                    trailing_stop_ratio = min(config.TRAILING_STOP_RATIO * 1.5, 0.05)  # 最高5%
                else:
                    trailing_stop_ratio = config.TRAILING_STOP_RATIO
                
                # 检查是否达到跟踪止损条件
                if (highest_price - current_price) / highest_price >= trailing_stop_ratio:
                    # 达到跟踪止损比例，卖出
                    # 详细的卖出理由
                    detailed_reason = [
                        f"跟踪止损触发(从最高价回撤{(highest_price - current_price)/highest_price:.2%} >= {trailing_stop_ratio:.2%})",
                        f"最高价:{highest_price:.2f}, 当前价:{current_price:.2f}",
                        f"属于板块({sector})"
                    ]
                    
                    # 构建完整的卖出理由
                    sell_reason = "; ".join(detailed_reason)
                    
                    sell_decisions.append({
                        'stock': stock,
                        'action': 'sell',
                        'price': current_price,
                        'reason': sell_reason,
                        'return_rate': return_rate,
                        'time': current_time,
                        'trailing_stop_ratio': trailing_stop_ratio,
                        'highest_price': highest_price,
                        'sector': sector,
                        'detailed_reason': detailed_reason
                    })
                    
                    logger.info(f"卖出决策：{sell_decisions[-1]}")
                    continue
            
            # 7. 尾盘处理：14:55后，对于盈利的股票可以考虑卖出锁定利润
            if current_hour == 14 and current_minute >= 55 and return_rate > 0:
                # 对于可能涨停的股票，允许持有到收盘
                if is_likely_limit_up:
                    logger.info(f"可能涨停的股票{stock}，持有到收盘")
                    # 更新持仓中的最高价
                    if current_price > holding.get('highest_price', 0):
                        self.holdings[stock]['highest_price'] = current_price
                    continue
                else:
                    # 普通股票，尾盘卖出锁定利润
                    # 详细的卖出理由
                    detailed_reason = [
                        "尾盘锁定利润",
                        f"当前收益率({return_rate:.2%})",
                        f"属于板块({sector})"
                    ]
                    
                    # 构建完整的卖出理由
                    sell_reason = "; ".join(detailed_reason)
                    
                    sell_decisions.append({
                        'stock': stock,
                        'action': 'sell',
                        'price': current_price,
                        'reason': sell_reason,
                        'return_rate': return_rate,
                        'time': current_time,
                        'sector': sector,
                        'detailed_reason': detailed_reason
                    })
                    
                    logger.info(f"卖出决策：{sell_decisions[-1]}")
                    continue
            
            # 更新持仓中的最高价
            if current_price > holding.get('highest_price', 0):
                self.holdings[stock]['highest_price'] = current_price
        
        return sell_decisions
    
    def _calculate_dynamic_position_size(self, stock, row, probability, total_score, current_capital):
        """动态计算仓位大小
        
        Args:
            stock: 股票代码
            row: 股票数据行
            probability: 模型预测概率
            total_score: 综合评分
            current_capital: 当前可用资金
            
        Returns:
            仓位比例 (0-1之间)
        """
        # 基础仓位比例
        base_position = 0.1  # 基础仓位10%
        
        # 1. 根据模型预测概率调整仓位 (0.7-1.0)
        if probability >= 0.9:
            probability_factor = 3.0  # 高概率，大幅增加仓位
        elif probability >= 0.85:
            probability_factor = 2.5  # 较高概率，增加仓位
        elif probability >= 0.8:
            probability_factor = 2.0  # 中等高概率，适度增加仓位
        elif probability >= 0.75:
            probability_factor = 1.5  # 中等概率，小幅增加仓位
        else:
            probability_factor = 1.0  # 基础概率，基础仓位
        
        # 2. 根据综合评分调整仓位 (0-100)
        if total_score >= 90:
            score_factor = 2.0  # 极高评分，大幅增加仓位
        elif total_score >= 85:
            score_factor = 1.8  # 高分，增加仓位
        elif total_score >= 80:
            score_factor = 1.5  # 中等高分，适度增加仓位
        elif total_score >= 75:
            score_factor = 1.2  # 中等评分，小幅增加仓位
        else:
            score_factor = 1.0  # 基础评分，基础仓位
        
        # 3. 根据市场环境调整仓位
        market_factor = 1.0
        if hasattr(row, 'index_change'):
            index_change = row.index_change
            if index_change <= -0.03:  # 大盘下跌3%以上，极端风险
                market_factor = 0.5  # 大幅降低仓位
            elif index_change <= -0.02:  # 大盘下跌2%-3%
                market_factor = 0.7  # 降低仓位
            elif index_change <= -0.01:  # 大盘下跌1%-2%
                market_factor = 0.9  # 小幅降低仓位
            elif index_change >= 0.02:  # 大盘上涨2%以上，市场强势
                market_factor = 1.2  # 小幅增加仓位
        
        # 4. 根据涨跌家数比调整仓位
        if hasattr(row, 'up_down_ratio'):
            up_down_ratio = row.up_down_ratio
            if up_down_ratio <= 0.3:  # 涨跌家数比极低，市场弱势
                market_factor *= 0.8  # 降低仓位
            elif up_down_ratio <= 0.5:  # 涨跌家数比低，市场偏弱
                market_factor *= 0.9  # 小幅降低仓位
            elif up_down_ratio >= 1.5:  # 涨跌家数比高，市场强势
                market_factor *= 1.1  # 小幅增加仓位
            elif up_down_ratio >= 2.0:  # 涨跌家数比极高，市场极强
                market_factor *= 1.2  # 增加仓位
        
        # 5. 根据板块情况调整仓位
        sector_factor = 1.0
        if hasattr(row, 'sector_hotness'):
            sector_hotness = row.sector_hotness
            if sector_hotness >= 0.9:  # 板块热度极高
                sector_factor = 1.3  # 大幅增加仓位
            elif sector_hotness >= 0.8:
                sector_factor = 1.2  # 增加仓位
            elif sector_hotness >= 0.7:
                sector_factor = 1.1  # 小幅增加仓位
            elif sector_hotness <= 0.3:  # 板块热度极低
                sector_factor = 0.7  # 降低仓位
        
        # 6. 根据股票市值调整仓位
        market_cap_factor = 1.0
        if hasattr(row, 'circulating_market_cap'):
            market_cap = row.circulating_market_cap
            if market_cap >= 50000000000:  # 大盘股（500亿以上）
                market_cap_factor = 1.1  # 大盘股风险较低，可适当增加仓位
            elif market_cap <= 5000000000:  # 小盘股（50亿以下）
                market_cap_factor = 0.8  # 小盘股风险较高，降低仓位
        
        # 7. 计算连续亏损情况
        consecutive_losses = 0
        if len(self.trade_records) >= 5:
            recent_trades = [r for r in self.trade_records if r['action'] == 'sell'][-5:]
            for trade in reversed(recent_trades):
                if trade.get('return_rate', 0) < 0:
                    consecutive_losses += 1
                else:
                    break
        
        # 8. 根据连续亏损情况调整仓位
        loss_factor = 1.0
        if consecutive_losses >= 3:
            loss_factor = 0.5  # 连续3次亏损，大幅降低仓位
        elif consecutive_losses == 2:
            loss_factor = 0.7  # 连续2次亏损，降低仓位
        elif consecutive_losses == 1:
            loss_factor = 0.9  # 连续1次亏损，小幅降低仓位
        
        # 9. 计算总仓位控制
        current_total_position = sum(h['quantity'] * h['buy_price'] for h in self.holdings.values()) / current_capital if current_capital > 0 else 0
        total_position_factor = 1.0
        if current_total_position >= 0.8:  # 总仓位超过80%
            total_position_factor = 0.5  # 大幅降低单票仓位
        elif current_total_position >= 0.6:  # 总仓位超过60%
            total_position_factor = 0.7  # 降低单票仓位
        elif current_total_position >= 0.4:  # 总仓位超过40%
            total_position_factor = 0.9  # 小幅降低单票仓位
        
        # 计算最终仓位比例
        position_ratio = base_position * probability_factor * score_factor * market_factor * sector_factor * market_cap_factor * loss_factor * total_position_factor
        
        # 限制仓位比例范围
        position_ratio = max(0.05, min(0.4, position_ratio))  # 单票仓位在5%-40%之间
        
        logger.info(f"股票{stock}的动态仓位计算：基础={base_position:.2f}, 概率因子={probability_factor:.2f}, 评分因子={score_factor:.2f}, 市场因子={market_factor:.2f}, 板块因子={sector_factor:.2f}, 市值因子={market_cap_factor:.2f}, 亏损因子={loss_factor:.2f}, 总仓位因子={total_position_factor:.2f}, 最终={position_ratio:.2f}")
        
        return position_ratio
    
    def _check_extreme_risk(self):
        """检查极端风险情况
        
        Returns:
            bool: 是否存在极端风险
        """
        # 1. 检查连续亏损情况
        consecutive_losses = 0
        if len(self.trade_records) >= 5:
            recent_trades = [r for r in self.trade_records if r['action'] == 'sell'][-5:]
            for trade in reversed(recent_trades):
                if trade.get('return_rate', 0) < 0:
                    consecutive_losses += 1
                else:
                    break
        
        if consecutive_losses >= 4:  # 连续4次亏损，极端风险
            logger.warning(f"连续亏损{consecutive_losses}次，触发极端风险控制")
            return True
        
        # 2. 检查单日最大亏损
        if hasattr(self, 'daily_pnl'):
            daily_loss = min(self.daily_pnl.values()) if self.daily_pnl else 0
            if daily_loss <= -0.05:  # 单日亏损超过5%，极端风险
                logger.warning(f"单日亏损{daily_loss:.2%}，触发极端风险控制")
                return True
        
        # 3. 检查最大回撤
        if hasattr(self, 'max_drawdown') and self.max_drawdown <= -0.15:  # 最大回撤超过15%，极端风险
            logger.warning(f"最大回撤{self.max_drawdown:.2%}，触发极端风险控制")
            return True
        
        return False
    
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
                        'highest_price': max(self.holdings[stock]['highest_price'], price),
                        'update_time': time
                    })
                else:
                    # 如果没有持有该股票，则新建持仓记录
                    self.holdings[stock] = {
                        'quantity': quantity,
                        'buy_price': price,
                        'highest_price': price,
                        'buy_time': time,
                        'update_time': time
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
                        self.holdings[stock]['update_time'] = time
            
            # 增强交易记录，添加更多详细信息
            enhanced_record = {
                **result,
                'holdings_status': self.holdings.copy(),
                'total_holdings_count': len(self.holdings),
                'timestamp': pd.Timestamp.now()
            }
            
            # 记录交易
            self.trade_records.append(enhanced_record)
            
            # 记录到日志
            if action == 'buy':
                logger.info(f"买入交易记录：股票={stock}, 价格={price}, 数量={quantity}, 时间={time}, 理由={result.get('reason', '模型预测')}, 板块={result.get('sector', '未知')}")
            else:
                logger.info(f"卖出交易记录：股票={stock}, 价格={price}, 数量={quantity}, 时间={time}, 理由={result.get('reason', '止盈止损')}, 收益率={result.get('return_rate', 0):.4f}, 板块={result.get('sector', '未知')}")
    
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
