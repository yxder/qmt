#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
风险评估模块
构建风险评估子系统，对筛选出的个股进行风险评级
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from utils.logger import setup_logger
from data.data_source import DataSource

logger = setup_logger()


class RiskEvaluator:
    """风险评估器，用于对筛选出的个股进行风险评级"""
    
    def __init__(self):
        """初始化风险评估器
        
        风险评估指标体系：
        - 市场整体情绪风险：评估市场整体情绪对个股的影响
        - 个股历史波动率：评估个股自身的价格波动风险
        - 流动性风险：评估个股的流动性状况
        - 政策风险：评估行业政策对个股的影响
        - 相关性风险：评估个股与大盘或板块的相关性
        - 资金流风险：评估个股的资金流向风险
        - 估值风险：评估个股的估值水平
        - 板块联动性风险：新增，评估个股所在板块的联动性风险
        """
        logger.info("初始化风险评估器")
        self.data_source = DataSource()
        
        # 明确定义风险评估指标权重
        self.risk_weights = {
            'market_sentiment_risk': 0.20,  # 市场整体情绪风险权重
            'volatility_risk': 0.25,  # 个股历史波动率权重
            'liquidity_risk': 0.20,  # 流动性风险权重
            'policy_risk': 0.10,  # 政策风险权重
            'correlation_risk': 0.10,  # 相关性风险权重
            'fund_flow_risk': 0.08,  # 资金流风险权重
            'valuation_risk': 0.05,  # 估值风险权重
            'sector_correlation_risk': 0.02  # 新增：板块联动性风险权重
        }
        
        # 明确定义风险评级阈值
        self.risk_rating_thresholds = {
            '低风险': 0.3,  # 风险评分 <= 0.3 为低风险
            '中风险': 0.6,  # 0.3 < 风险评分 <= 0.6 为中风险
            '高风险': 1.0   # 风险评分 > 0.6 为高风险
        }
        
        # 明确定义风险预警阈值
        # 当单项风险评分或综合风险评分超过对应阈值时，触发风险预警
        self.risk_alert_thresholds = {
            'market_sentiment_risk': 0.7,  # 市场情绪风险预警阈值
            'volatility_risk': 0.7,  # 波动率风险预警阈值
            'liquidity_risk': 0.7,  # 流动性风险预警阈值
            'policy_risk': 0.8,  # 政策风险预警阈值
            'fund_flow_risk': 0.8,  # 资金流风险预警阈值
            'valuation_risk': 0.8,  # 估值风险预警阈值
            'sector_correlation_risk': 0.7,  # 板块联动性风险预警阈值
            'total_risk': 0.6  # 综合风险预警阈值
        }
        
        # 明确定义止损条件参数
        self.stop_loss_params = {
            'base_stop_loss_ratio': 0.06,  # 基础止损比例
            'base_trailing_stop_ratio': 0.08,  # 基础跟踪止损比例
            'max_stop_loss_ratio': 0.12,  # 最大止损比例
            'min_stop_loss_ratio': 0.03,  # 最小止损比例
            'risk_level_adjustment': {
                '低风险': 1.3,  # 低风险股票，止损比例可放宽
                '中风险': 1.0,  # 中风险股票，使用基础止损比例
                '高风险': 0.7   # 高风险股票，止损比例应收紧
            },
            'volatility_adjustment_factor': 0.15,  # 波动率调整因子
            'liquidity_adjustment_factor': 0.10,  # 流动性调整因子
            'market_environment_adjustment': {
                'bull': 1.2,  # 牛市环境，止损比例可放宽
                'bear': 0.8,  # 熊市环境，止损比例应收紧
                'neutral': 1.0  # 中性环境，使用基础止损比例
            }
        }
    
    def _get_dynamic_risk_weights(self, market_sentiment_risk: float) -> dict:
        """获取动态风险权重，根据市场环境调整各风险指标的权重
        
        Args:
            market_sentiment_risk: 市场整体情绪风险
            
        Returns:
            动态风险权重字典，包含所有风险指标的权重
        """
        logger.info("计算动态风险权重")
        
        # 初始化基础权重，使用配置中的完整权重体系
        weights = self.risk_weights.copy()
        
        # 根据市场情绪调整权重
        # 如果市场情绪风险高，增加市场情绪风险权重和相关性风险权重
        if market_sentiment_risk > 0.6:
            logger.info("市场情绪风险高，调整风险权重")
            # 增加市场情绪风险权重
            weights['market_sentiment_risk'] += 0.1
            # 增加相关性风险权重
            weights['correlation_risk'] += 0.05
            # 增加政策风险权重
            weights['policy_risk'] += 0.05
            # 减少波动率风险和流动性风险权重
            weights['volatility_risk'] -= 0.075
            weights['liquidity_risk'] -= 0.075
        elif market_sentiment_risk < 0.3:
            logger.info("市场情绪风险低，调整风险权重")
            # 市场情绪风险低，减少市场情绪风险权重，增加其他风险权重
            weights['market_sentiment_risk'] -= 0.1
            # 增加波动率风险和流动性风险权重
            weights['volatility_risk'] += 0.05
            weights['liquidity_risk'] += 0.05
            # 增加资金流风险和估值风险权重
            weights['fund_flow_risk'] += 0.05
            weights['valuation_risk'] += 0.05
        
        # 确保所有权重为正值
        for key in weights:
            weights[key] = max(0.01, weights[key])
        
        # 确保权重之和为1
        total_weight = sum(weights.values())
        for key in weights:
            weights[key] /= total_weight
        
        logger.info(f"动态风险权重：{weights}")
        return weights
    
    def evaluate_risk(self, candidate_stocks: pd.DataFrame, market_data: pd.DataFrame = None) -> pd.DataFrame:
        """对候选个股进行风险评级
        
        Args:
            candidate_stocks: 候选个股DataFrame
            market_data: 市场数据，如果为None则使用默认数据
            
        Returns:
            带有风险评级的候选个股DataFrame，包含：
            - 各项风险指标评分
            - 综合风险评分
            - 风险等级
            - 风险预警状态
            - 具体的止损条件
        """
        logger.info("开始对候选个股进行风险评级")
        
        if candidate_stocks.empty:
            logger.error("候选个股为空，无法进行风险评估")
            return pd.DataFrame()
        
        # 1. 评估市场整体情绪风险
        market_sentiment_risk = self._evaluate_market_sentiment_risk(market_data)
        logger.info(f"市场整体情绪风险：{market_sentiment_risk:.2f}")
        
        # 2. 获取动态风险权重
        dynamic_weights = self._get_dynamic_risk_weights(market_sentiment_risk)
        
        # 添加板块联动性风险权重到动态权重
        if 'sector_correlation_risk' not in dynamic_weights:
            dynamic_weights['sector_correlation_risk'] = self.risk_weights['sector_correlation_risk']
        
        logger.info(f"动态风险权重：{dynamic_weights}")
        
        # 3. 评估个股风险指标
        risk_evaluations = []
        
        for idx, stock in candidate_stocks.iterrows():
            stock_code = stock['stock_code']
            logger.info(f"评估股票{stock_code}的风险")
            
            # 获取股票所在板块
            sector = stock.get('sector', '未知')
            
            # 评估各项风险指标
            volatility_risk = self._evaluate_volatility_risk(stock_code)
            liquidity_risk = self._evaluate_liquidity_risk(stock_code)
            policy_risk = self._evaluate_policy_risk(stock_code)
            correlation_risk = self._evaluate_correlation_risk(stock_code)
            fund_flow_risk = self._evaluate_fund_flow_risk(stock_code)
            valuation_risk = self._evaluate_valuation_risk(stock_code)
            sector_correlation_risk = self._evaluate_sector_correlation_risk(sector, candidate_stocks)
            
            # 综合风险评分（使用动态权重）
            total_risk = (
                market_sentiment_risk * dynamic_weights['market_sentiment_risk'] +
                volatility_risk * dynamic_weights['volatility_risk'] +
                liquidity_risk * dynamic_weights['liquidity_risk'] +
                policy_risk * dynamic_weights['policy_risk'] +
                correlation_risk * dynamic_weights['correlation_risk'] +
                fund_flow_risk * dynamic_weights['fund_flow_risk'] +
                valuation_risk * dynamic_weights['valuation_risk'] +
                sector_correlation_risk * dynamic_weights['sector_correlation_risk']
            )
            
            # 确保风险评分在0-1范围内
            total_risk = max(0, min(1, total_risk))
            
            # 风险评级
            risk_level = self._get_risk_level(total_risk)
            
            # 风险预警 - 检查各项风险指标和综合风险
            risk_alert = self._check_risk_alert(total_risk, {
                'market_sentiment_risk': market_sentiment_risk,
                'volatility_risk': volatility_risk,
                'liquidity_risk': liquidity_risk,
                'policy_risk': policy_risk,
                'fund_flow_risk': fund_flow_risk,
                'valuation_risk': valuation_risk,
                'sector_correlation_risk': sector_correlation_risk
            })
            
            # 动态止损条件
            stop_loss_condition = self._calculate_stop_loss_condition(stock, total_risk, risk_level, volatility_risk, liquidity_risk)
            
            # 保存风险评估结果
            risk_evaluation = {
                'stock_code': stock_code,
                'market_sentiment_risk': market_sentiment_risk,
                'volatility_risk': volatility_risk,
                'liquidity_risk': liquidity_risk,
                'policy_risk': policy_risk,
                'correlation_risk': correlation_risk,
                'fund_flow_risk': fund_flow_risk,
                'valuation_risk': valuation_risk,
                'sector_correlation_risk': sector_correlation_risk,
                'total_risk': total_risk,
                'risk_level': risk_level,
                'risk_alert': risk_alert,
                'stop_loss_condition': stop_loss_condition
            }
            
            risk_evaluations.append(risk_evaluation)
        
        # 转换为DataFrame
        risk_df = pd.DataFrame(risk_evaluations)
        
        # 将风险评估结果合并到候选个股中
        result = pd.merge(candidate_stocks, risk_df, on='stock_code', how='left')
        
        logger.info("风险评估完成")
        return result
    
    def _evaluate_market_sentiment_risk(self, market_data: pd.DataFrame = None) -> float:
        """评估市场整体情绪风险
        
        Args:
            market_data: 市场数据，如果为None则使用默认数据
            
        Returns:
            市场整体情绪风险评分（0-1）
        """
        logger.info("评估市场整体情绪风险")
        
        # 如果没有提供市场数据，则生成默认数据
        if market_data is None:
            # 模拟市场情绪数据
            market_data = {
                'up_down_ratio': np.random.rand() * 2,  # 涨跌家数比
                'market_volatility': np.random.rand() * 0.05,  # 市场波动率
                'fund_flow': np.random.randint(-1000000000, 1000000000),  # 资金净流入
                'index_change': np.random.rand() * 0.05 - 0.025  # 指数涨跌幅
            }
        
        # 1. 涨跌家数比风险
        # 涨跌家数比越低，风险越高
        up_down_ratio = market_data.get('up_down_ratio', 1.0)
        up_down_risk = 1.0 - min(up_down_ratio / 3, 1.0)  # 归一化到0-1
        
        # 2. 市场波动率风险
        # 市场波动率越高，风险越高
        market_volatility = market_data.get('market_volatility', 0.02)
        volatility_risk = min(market_volatility / 0.05, 1.0)  # 归一化到0-1
        
        # 3. 资金流向风险
        # 资金净流出越多，风险越高
        fund_flow = market_data.get('fund_flow', 0)
        fund_flow_risk = min(max(-fund_flow / 5000000000, 0), 1.0)  # 归一化到0-1
        
        # 4. 指数涨跌幅风险
        # 指数跌幅越大，风险越高
        index_change = market_data.get('index_change', 0)
        index_risk = min(max(-index_change / 0.05, 0), 1.0)  # 归一化到0-1
        
        # 综合市场情绪风险
        market_sentiment_risk = (
            up_down_risk * 0.3 +
            volatility_risk * 0.3 +
            fund_flow_risk * 0.2 +
            index_risk * 0.2
        )
        
        return market_sentiment_risk
    
    def _evaluate_volatility_risk(self, stock_code: str) -> float:
        """评估个股历史波动率风险
        
        Args:
            stock_code: 股票代码
            
        Returns:
            个股历史波动率风险评分（0-1）
        """
        logger.info(f"评估股票{stock_code}的历史波动率风险")
        
        # 获取股票历史数据
        hist_data = self.data_source.get_historical_data(stock_code, '2025-08-01', '2025-08-31')
        
        if hist_data.empty:
            # 如果没有历史数据，返回默认风险值
            return 0.5
        
        # 计算日收益率
        hist_data['daily_return'] = hist_data['close'].pct_change()
        
        # 计算波动率（标准差）
        volatility = hist_data['daily_return'].std()
        
        # 年化波动率
        annual_volatility = volatility * np.sqrt(252)
        
        # 归一化到0-1范围
        # 设定波动率阈值：0.2为中等风险，0.4为高风险
        volatility_risk = min(annual_volatility / 0.4, 1.0)
        
        logger.info(f"股票{stock_code}的年化波动率：{annual_volatility:.2f}，风险评分：{volatility_risk:.2f}")
        return volatility_risk
    
    def _evaluate_liquidity_risk(self, stock_code: str) -> float:
        """评估流动性风险
        
        Args:
            stock_code: 股票代码
            
        Returns:
            流动性风险评分（0-1）
        """
        logger.info(f"评估股票{stock_code}的流动性风险")
        
        # 获取股票历史数据
        hist_data = self.data_source.get_historical_data(stock_code, '2025-08-01', '2025-08-31')
        
        if hist_data.empty:
            # 如果没有历史数据，返回默认风险值
            return 0.5
        
        # 1. 日均成交量
        avg_volume = hist_data['volume'].mean()
        
        # 2. 日均成交额
        avg_amount = hist_data['amount'].mean()
        
        # 3. 换手率
        avg_turnover_rate = hist_data['turnover_rate'].mean()
        
        # 计算流动性风险
        # 成交量越低、成交额越低、换手率越低，流动性风险越高
        volume_risk = 1.0 - min(avg_volume / 10000000, 1.0)  # 1000万股为基准
        amount_risk = 1.0 - min(avg_amount / 100000000, 1.0)  # 1亿为基准
        turnover_risk = 1.0 - min(avg_turnover_rate / 0.05, 1.0)  # 5%为基准
        
        # 综合流动性风险
        liquidity_risk = (
            volume_risk * 0.4 +
            amount_risk * 0.4 +
            turnover_risk * 0.2
        )
        
        logger.info(f"股票{stock_code}的流动性风险评分：{liquidity_risk:.2f}")
        return liquidity_risk
    
    def _evaluate_policy_risk(self, stock_code: str) -> float:
        """评估政策风险
        
        Args:
            stock_code: 股票代码
            
        Returns:
            政策风险评分（0-1）
        """
        logger.info(f"评估股票{stock_code}的政策风险")
        
        # 获取股票基本信息
        basic_info = self.data_source.get_stock_basic_info(stock_code)
        industry = basic_info.get('industry', '未知')
        
        # 定义高政策风险行业
        high_policy_risk_industries = ['金融', '地产', '医药', '教育', '互联网']
        
        # 定义中政策风险行业
        medium_policy_risk_industries = ['新能源', '科技', '通信', '传媒']
        
        # 根据行业评估政策风险
        if industry in high_policy_risk_industries:
            policy_risk = 0.7
        elif industry in medium_policy_risk_industries:
            policy_risk = 0.4
        else:
            policy_risk = 0.2
        
        logger.info(f"股票{stock_code}所属行业{industry}，政策风险评分：{policy_risk:.2f}")
        return policy_risk
    
    def _evaluate_correlation_risk(self, stock_code: str) -> float:
        """评估相关性风险
        
        Args:
            stock_code: 股票代码
            
        Returns:
            相关性风险评分（0-1）
        """
        logger.info(f"评估股票{stock_code}的相关性风险")
        
        # 模拟相关性风险，实际实现中应计算与大盘或板块的相关性
        # 相关性越高，风险越高
        correlation_risk = np.random.rand() * 0.5 + 0.2  # 生成0.2-0.7之间的随机值
        
        logger.info(f"股票{stock_code}的相关性风险评分：{correlation_risk:.2f}")
        return correlation_risk
    
    def _evaluate_fund_flow_risk(self, stock_code: str) -> float:
        """评估资金流风险
        
        Args:
            stock_code: 股票代码
            
        Returns:
            资金流风险评分（0-1）
        """
        logger.info(f"评估股票{stock_code}的资金流风险")
        
        # 获取股票历史数据
        hist_data = self.data_source.get_historical_data(stock_code, '2025-08-01', '2025-08-31')
        
        if hist_data.empty:
            # 如果没有历史数据，返回默认风险值
            return 0.5
        
        # 计算资金流向指标
        # 1. 近5日资金净流入情况
        if 'fund_flow_net_flow' in hist_data.columns:
            # 计算近5日资金净流入
            recent_fund_flow = hist_data['fund_flow_net_flow'].tail(5)
            
            # 计算资金净流入均值
            avg_net_flow = recent_fund_flow.mean()
            
            # 计算资金净流入标准差
            std_net_flow = recent_fund_flow.std()
            
            # 计算资金净流入为负的天数比例
            negative_days_ratio = (recent_fund_flow < 0).mean()
            
            # 计算大单资金占比
            if 'fund_flow_large_order_flow' in hist_data.columns:
                large_order_ratio = recent_fund_flow.apply(lambda x: 
                    hist_data['fund_flow_large_order_flow'].iloc[x.name] / (x + 1e-10) if x > 0 else 0
                ).mean()
            else:
                large_order_ratio = 0.5
            
            # 计算资金流风险
            # 资金净流出越多、波动越大、负天数比例越高，风险越高
            # 大单资金占比越高，风险越低
            fund_flow_risk = (
                max(0, -avg_net_flow / 10000000) * 0.3 +  # 资金净流出风险（每1000万净流出增加0.3风险）
                std_net_flow / (abs(avg_net_flow) + 1e-10) * 0.2 +  # 波动风险
                negative_days_ratio * 0.3 +  # 负天数比例风险
                (1 - large_order_ratio) * 0.2  # 大单占比风险
            )
            
            # 归一化到0-1范围
            fund_flow_risk = min(fund_flow_risk, 1.0)
            logger.info(f"股票{stock_code}的资金流风险评分：{fund_flow_risk:.2f}")
            return fund_flow_risk
        
        # 如果没有资金流数据，返回默认风险值
        return 0.5
    
    def _evaluate_valuation_risk(self, stock_code: str) -> float:
        """评估估值风险
        
        Args:
            stock_code: 股票代码
            
        Returns:
            估值风险评分（0-1）
        """
        logger.info(f"评估股票{stock_code}的估值风险")
        
        # 获取股票基本信息
        basic_info = self.data_source.get_stock_basic_info(stock_code)
        
        if not basic_info:
            # 如果没有基本信息，返回默认风险值
            return 0.5
        
        # 提取估值指标
        pe = basic_info.get('pe', 30)  # 市盈率
        pb = basic_info.get('pb', 3)  # 市净率
        ps = basic_info.get('ps', 5)  # 市销率
        pe_ttm = basic_info.get('pe_ttm', 25)  # 滚动市盈率
        
        # 获取行业平均估值
        industry = basic_info.get('industry', '未知')
        industry_pe = basic_info.get('industry_pe', 20)  # 行业平均市盈率
        industry_pb = basic_info.get('industry_pb', 2)  # 行业平均市净率
        
        # 计算估值风险
        # 估值越高、偏离行业平均越多，风险越高
        
        # 市盈率风险（相对行业平均）
        pe_risk = min(max((pe - industry_pe) / industry_pe, -0.5), 1.5)  # 归一化到-0.5到1.5
        pe_risk = (pe_risk + 0.5) / 2  # 转换为0到1
        
        # 市净率风险（相对行业平均）
        pb_risk = min(max((pb - industry_pb) / industry_pb, -0.5), 1.5)  # 归一化到-0.5到1.5
        pb_risk = (pb_risk + 0.5) / 2  # 转换为0到1
        
        # 绝对值估值风险
        # 设定合理估值范围：PE 10-30，PB 1-5
        abs_pe_risk = min(max((pe - 20) / 10, -1), 1)  # PE偏离20的程度
        abs_pe_risk = (abs_pe_risk + 1) / 2  # 转换为0到1
        
        abs_pb_risk = min(max((pb - 3) / 2, -1), 1)  # PB偏离3的程度
        abs_pb_risk = (abs_pb_risk + 1) / 2  # 转换为0到1
        
        # 综合估值风险
        valuation_risk = (
            pe_risk * 0.3 +
            pb_risk * 0.3 +
            abs_pe_risk * 0.2 +
            abs_pb_risk * 0.2
        )
        
        # 确保估值风险在0-1范围内
        valuation_risk = min(max(valuation_risk, 0), 1)
        
        logger.info(f"股票{stock_code}的估值风险评分：{valuation_risk:.2f} (PE: {pe}, PB: {pb}, 行业PE: {industry_pe}, 行业PB: {industry_pb})")
        return valuation_risk
    
    def _calculate_total_risk(self, market_sentiment_risk: float, volatility_risk: float, 
                            liquidity_risk: float, policy_risk: float, correlation_risk: float) -> float:
        """计算综合风险评分
        
        Args:
            market_sentiment_risk: 市场整体情绪风险
            volatility_risk: 个股历史波动率风险
            liquidity_risk: 流动性风险
            policy_risk: 政策风险
            correlation_risk: 相关性风险
            
        Returns:
            综合风险评分（0-1）
        """
        # 综合风险评分 = 各项风险指标 * 权重之和
        total_risk = (
            market_sentiment_risk * self.risk_weights['market_sentiment_risk'] +
            volatility_risk * self.risk_weights['volatility_risk'] +
            liquidity_risk * self.risk_weights['liquidity_risk'] +
            policy_risk * self.risk_weights['policy_risk'] +
            correlation_risk * self.risk_weights['correlation_risk']
        )
        
        # 确保风险评分在0-1范围内
        total_risk = max(0, min(1, total_risk))
        
        return total_risk
    
    def _get_risk_level(self, total_risk: float) -> str:
        """根据综合风险评分获取风险等级
        
        Args:
            total_risk: 综合风险评分
            
        Returns:
            风险等级：低风险、中风险、高风险
        """
        if total_risk <= self.risk_rating_thresholds['低风险']:
            return '低风险'
        elif total_risk <= self.risk_rating_thresholds['中风险']:
            return '中风险'
        else:
            return '高风险'
    
    def _check_risk_alert(self, total_risk: float, risk_metrics: Dict[str, float] = None) -> bool:
        """检查是否触发风险预警
        
        Args:
            total_risk: 综合风险评分
            risk_metrics: 各项风险指标评分字典
            
        Returns:
            是否触发风险预警
        """
        # 检查综合风险评分是否超过预警阈值
        if total_risk >= self.risk_alert_thresholds['total_risk']:
            logger.info(f"综合风险评分{total_risk:.2f}超过预警阈值{self.risk_alert_thresholds['total_risk']:.2f}，触发风险预警")
            return True
        
        # 检查各项风险指标是否超过预警阈值
        if risk_metrics:
            for risk_type, risk_score in risk_metrics.items():
                if risk_type in self.risk_alert_thresholds:
                    if risk_score >= self.risk_alert_thresholds[risk_type]:
                        logger.info(f"{risk_type}评分{risk_score:.2f}超过预警阈值{self.risk_alert_thresholds[risk_type]:.2f}，触发风险预警")
                        return True
        
        return False
    
    def _evaluate_sector_correlation_risk(self, sector: str, candidate_stocks: pd.DataFrame) -> float:
        """评估板块联动性风险
        
        Args:
            sector: 股票所在板块
            candidate_stocks: 候选个股DataFrame
            
        Returns:
            板块联动性风险评分（0-1）
        """
        logger.info(f"评估板块{sector}的联动性风险")
        
        # 筛选该板块的股票
        sector_stocks = candidate_stocks[candidate_stocks['sector'] == sector]
        
        if len(sector_stocks) < 3:
            # 板块股票数量不足，返回默认风险值
            return 0.2
        
        try:
            # 计算板块内股票涨跌幅的相关性
            price_changes = sector_stocks['price_bid_change'].values
            correlation_matrix = np.corrcoef(price_changes)
            
            # 计算平均相关系数
            upper_triangle = correlation_matrix[np.triu_indices(len(correlation_matrix), k=1)]
            avg_correlation = np.mean(upper_triangle)
            
            # 板块联动性越高，风险越大（因为板块内股票同涨同跌，分散投资效果差）
            sector_correlation_risk = min(avg_correlation * 1.5, 1.0)  # 放大相关性影响，归一化到0-1
            
            logger.info(f"板块{sector}的联动性风险评分：{sector_correlation_risk:.2f}")
            return sector_correlation_risk
        except Exception as e:
            logger.error(f"计算板块{sector}联动性风险失败：{e}")
            return 0.5
    
    def _calculate_stop_loss_condition(self, stock: pd.Series, total_risk: float, risk_level: str, 
                                      volatility_risk: float = 0.5, liquidity_risk: float = 0.5) -> Dict[str, float]:
        """计算止损条件
        
        Args:
            stock: 股票数据
            total_risk: 综合风险评分
            risk_level: 风险等级
            volatility_risk: 波动率风险评分
            liquidity_risk: 流动性风险评分
            
        Returns:
            止损条件字典，包含：
            - 固定止损比例
            - 跟踪止损比例
            - 动态调整因子
            - 止损触发条件
        """
        logger.info(f"计算股票{stock['stock_code']}的止损条件，风险等级：{risk_level}")
        
        # 获取基础止损参数
        base_stop_loss = self.stop_loss_params['base_stop_loss_ratio']
        base_trailing_stop = self.stop_loss_params['base_trailing_stop_ratio']
        min_stop_loss = self.stop_loss_params['min_stop_loss_ratio']
        max_stop_loss = self.stop_loss_params['max_stop_loss_ratio']
        
        # 根据风险等级调整止损比例
        risk_level_adjustment = self.stop_loss_params['risk_level_adjustment'].get(risk_level, 1.0)
        logger.info(f"风险等级调整因子：{risk_level_adjustment}")
        
        # 根据波动率调整止损比例
        volatility_adjustment = self._get_volatility_adjustment(stock['stock_code'])
        logger.info(f"波动率调整因子：{volatility_adjustment}")
        
        # 根据流动性风险调整止损比例
        # 流动性风险越高，止损比例应越小（避免无法及时止损）
        liquidity_adjustment = 1.0 - (liquidity_risk - 0.5) * self.stop_loss_params['liquidity_adjustment_factor']
        logger.info(f"流动性调整因子：{liquidity_adjustment}")
        
        # 根据综合风险评分调整止损比例
        total_risk_adjustment = 1.0 + (total_risk - 0.5) * 0.2  # 0.2为风险调整系数
        logger.info(f"综合风险调整因子：{total_risk_adjustment}")
        
        # 根据股票涨幅调整止损比例
        # 涨幅越大，止损比例可适当放宽
        price_bid_change = stock.get('price_bid_change', 0)
        price_change_adjustment = 1.0 + min(price_bid_change * 2, 0.5)  # 涨幅每增加1%，止损比例放宽2%，最大放宽50%
        logger.info(f"价格涨幅调整因子：{price_change_adjustment}")
        
        # 根据资金流入调整止损比例
        # 资金流入越强，止损比例可适当放宽
        fund_flow_net_flow = stock.get('fund_flow_net_flow', 0)
        fund_flow_adjustment = 1.0 + min(fund_flow_net_flow / 100000000, 0.3)  # 每1亿资金流入，止损比例放宽1%，最大放宽30%
        logger.info(f"资金流入调整因子：{fund_flow_adjustment}")
        
        # 根据成交量比调整止损比例
        # 成交量比越大，止损比例可适当放宽
        volume_bid_volume_ratio = stock.get('volume_bid_volume_ratio', 1.0)
        volume_ratio_adjustment = 1.0 + min((volume_bid_volume_ratio - 1) * 0.1, 0.5)  # 成交量比每增加1倍，止损比例放宽10%，最大放宽50%
        logger.info(f"成交量比调整因子：{volume_ratio_adjustment}")
        
        # 计算最终止损比例
        stop_loss_ratio = base_stop_loss * risk_level_adjustment * volatility_adjustment * liquidity_adjustment * \
                         total_risk_adjustment * price_change_adjustment * fund_flow_adjustment * volume_ratio_adjustment
        trailing_stop_ratio = base_trailing_stop * risk_level_adjustment * volatility_adjustment * liquidity_adjustment * \
                            total_risk_adjustment * price_change_adjustment * fund_flow_adjustment * volume_ratio_adjustment
        
        # 确保止损比例在设定范围内
        stop_loss_ratio = max(min_stop_loss, min(stop_loss_ratio, max_stop_loss))
        trailing_stop_ratio = max(min_stop_loss / 2, min(trailing_stop_ratio, max_stop_loss / 2))
        
        # 计算止损触发条件
        # 基于当前价格的止损价格
        current_price = stock.get('price_bid_price', stock.get('price_open', 0))
        stop_loss_price = current_price * (1 - stop_loss_ratio)
        trailing_stop_price = current_price * (1 - trailing_stop_ratio)
        
        logger.info(f"最终止损条件：固定止损={stop_loss_ratio:.2%} ({stop_loss_price:.2f}元)，跟踪止损={trailing_stop_ratio:.2%} ({trailing_stop_price:.2f}元)")
        
        return {
            'stop_loss_ratio': stop_loss_ratio,  # 固定止损比例
            'stop_loss_price': stop_loss_price,  # 固定止损价格
            'trailing_stop_ratio': trailing_stop_ratio,  # 跟踪止损比例
            'trailing_stop_price': trailing_stop_price,  # 跟踪止损价格
            'risk_level_adjustment': risk_level_adjustment,  # 风险等级调整因子
            'volatility_adjustment': volatility_adjustment,  # 波动率调整因子
            'liquidity_adjustment': liquidity_adjustment,  # 流动性调整因子
            'total_risk_adjustment': total_risk_adjustment,  # 综合风险调整因子
            'price_change_adjustment': price_change_adjustment,  # 价格涨幅调整因子
            'fund_flow_adjustment': fund_flow_adjustment,  # 资金流入调整因子
            'volume_ratio_adjustment': volume_ratio_adjustment,  # 成交量比调整因子
            'base_stop_loss': base_stop_loss,  # 基础止损比例
            'base_trailing_stop': base_trailing_stop  # 基础跟踪止损比例
        }
    
    def _get_volatility_adjustment(self, stock_code: str) -> float:
        """获取波动率调整系数
        
        Args:
            stock_code: 股票代码
            
        Returns:
            波动率调整系数
        """
        # 获取股票历史数据
        hist_data = self.data_source.get_historical_data(stock_code, '2025-08-01', '2025-08-31')
        
        if hist_data.empty:
            return 1.0
        
        # 计算日收益率
        hist_data['daily_return'] = hist_data['close'].pct_change()
        
        # 计算波动率
        volatility = hist_data['daily_return'].std()
        
        # 年化波动率
        annual_volatility = volatility * np.sqrt(252)
        
        # 波动率调整系数：波动率越高，调整系数越大
        volatility_adjustment = min(annual_volatility / 0.2 + 0.5, 2.0)
        
        return volatility_adjustment
    
    def get_risk_report(self, risk_evaluated_stocks: pd.DataFrame) -> Dict[str, any]:
        """生成风险评估报告
        
        Args:
            risk_evaluated_stocks: 带有风险评级的候选个股DataFrame
            
        Returns:
            风险评估报告字典，包含详细的风险分析
        """
        logger.info("生成风险评估报告")
        
        if risk_evaluated_stocks.empty:
            return {}
        
        # 计算各项风险指标的平均值
        risk_metrics = [
            'market_sentiment_risk', 'volatility_risk', 'liquidity_risk',
            'policy_risk', 'correlation_risk', 'fund_flow_risk',
            'valuation_risk', 'sector_correlation_risk'
        ]
        
        average_risks = {}
        for metric in risk_metrics:
            if metric in risk_evaluated_stocks.columns:
                average_risks[metric] = risk_evaluated_stocks[metric].mean()
        
        # 计算各风险等级的详细信息
        risk_level_details = {}
        for level in ['低风险', '中风险', '高风险']:
            level_stocks = risk_evaluated_stocks[risk_evaluated_stocks['risk_level'] == level]
            if not level_stocks.empty:
                risk_level_details[level] = {
                    'count': len(level_stocks),
                    'average_total_risk': level_stocks['total_risk'].mean(),
                    'stocks': level_stocks['stock_code'].tolist()
                }
        
        # 计算风险预警详细信息
        alert_stocks = risk_evaluated_stocks[risk_evaluated_stocks['risk_alert'] == True]
        alert_details = {
            'count': len(alert_stocks),
            'stocks': alert_stocks['stock_code'].tolist(),
            'average_total_risk': alert_stocks['total_risk'].mean() if not alert_stocks.empty else 0
        }
        
        # 计算各风险指标的分布
        risk_metric_distributions = {}
        for metric in risk_metrics:
            if metric in risk_evaluated_stocks.columns:
                metric_values = risk_evaluated_stocks[metric]
                risk_metric_distributions[metric] = {
                    'min': metric_values.min(),
                    'max': metric_values.max(),
                    'mean': metric_values.mean(),
                    'median': metric_values.median(),
                    'std': metric_values.std()
                }
        
        # 计算板块风险分析
        sector_risk_analysis = self._get_sector_risk_analysis(risk_evaluated_stocks)
        
        report = {
            'summary': {
                'total_stocks': len(risk_evaluated_stocks),
                'risk_level_distribution': risk_evaluated_stocks['risk_level'].value_counts().to_dict(),
                'average_total_risk': risk_evaluated_stocks['total_risk'].mean(),
                'risk_alert_count': risk_evaluated_stocks['risk_alert'].sum()
            },
            'average_risks': average_risks,
            'risk_level_details': risk_level_details,
            'risk_alert_details': alert_details,
            'risk_metric_distributions': risk_metric_distributions,
            'sector_risk_analysis': sector_risk_analysis,
            'top_10_low_risk_stocks': risk_evaluated_stocks.sort_values(by='total_risk').head(10)['stock_code'].tolist(),
            'top_10_high_risk_stocks': risk_evaluated_stocks.sort_values(by='total_risk', ascending=False).head(10)['stock_code'].tolist()
        }
        
        logger.info(f"风险评估报告生成完成")
        return report
    
    def _get_risk_by_industry(self, risk_evaluated_stocks: pd.DataFrame) -> Dict[str, float]:
        """获取各行业的平均风险
        
        Args:
            risk_evaluated_stocks: 带有风险评级的候选个股DataFrame
            
        Returns:
            各行业的平均风险字典
        """
        # 如果没有行业信息，返回空字典
        if 'industry' not in risk_evaluated_stocks.columns:
            return {}
        
        # 按行业分组计算平均风险
        industry_risk = risk_evaluated_stocks.groupby('industry')['total_risk'].mean().to_dict()
        
        return industry_risk
    
    def _get_sector_risk_analysis(self, risk_evaluated_stocks: pd.DataFrame) -> Dict[str, any]:
        """获取板块风险分析
        
        Args:
            risk_evaluated_stocks: 带有风险评级的候选个股DataFrame
            
        Returns:
            板块风险分析字典
        """
        if 'sector' not in risk_evaluated_stocks.columns:
            return {}
        
        sector_analysis = {}
        
        for sector, sector_stocks in risk_evaluated_stocks.groupby('sector'):
            sector_analysis[sector] = {
                'stock_count': len(sector_stocks),
                'average_total_risk': sector_stocks['total_risk'].mean(),
                'risk_level_distribution': sector_stocks['risk_level'].value_counts().to_dict(),
                'average_sector_correlation_risk': sector_stocks['sector_correlation_risk'].mean() if 'sector_correlation_risk' in sector_stocks.columns else 0,
                'stocks': sector_stocks['stock_code'].tolist()
            }
        
        return sector_analysis
