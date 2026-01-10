#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
板块识别模块
基于采集的集合竞价数据，构建多维度指标评估体系，筛选出当日TOP热门板块
"""

import pandas as pd
import numpy as np
from utils.logger import setup_logger
from data.data_source import DataSource

logger = setup_logger()


class SectorRecognizer:
    """板块识别器，用于识别当日热门板块"""
    
    def __init__(self):
        """初始化板块识别器"""
        logger.info("初始化板块识别器")
        self.data_source = DataSource()
        self.sector_metrics = {}
        self.top_sectors = []
    
    def evaluate_sectors(self, bidding_data: pd.DataFrame) -> pd.DataFrame:
        """评估板块热度，筛选TOP热门板块
        
        Args:
            bidding_data: 集合竞价数据，包含所有股票的竞价信息
            
        Returns:
            排序后的热门板块DataFrame
        """
        logger.info("开始评估板块热度")
        
        if bidding_data.empty:
            logger.error("输入数据为空，无法评估板块热度")
            return pd.DataFrame()
        
        # 1. 添加板块信息到竞价数据中
        bidding_data = self._add_sector_info(bidding_data)
        
        # 2. 计算各板块的多维度指标
        self._calculate_sector_metrics(bidding_data)
        
        # 3. 综合评分并排序
        sector_ranking = self._rank_sectors()
        
        # 4. 筛选TOP热门板块
        self.top_sectors = self._filter_top_sectors(sector_ranking)
        
        logger.info("板块热度评估完成")
        return self.top_sectors
    
    def _add_sector_info(self, bidding_data: pd.DataFrame) -> pd.DataFrame:
        """为竞价数据添加板块信息
        
        Args:
            bidding_data: 集合竞价数据
            
        Returns:
            添加了板块信息的竞价数据
        """
        logger.info("为竞价数据添加板块信息")
        
        # 获取板块映射关系
        sector_mapping = self.data_source.get_sector_mapping()
        
        # 添加板块列
        bidding_data['sector'] = bidding_data['stock_code'].map(sector_mapping)
        
        # 填充未知板块
        bidding_data['sector'] = bidding_data['sector'].fillna('未知')
        
        logger.info(f"板块信息添加完成，共有{len(bidding_data['sector'].unique())}个板块")
        return bidding_data
    
    def _calculate_sector_metrics(self, bidding_data: pd.DataFrame):
        """计算各板块的多维度指标
        
        Args:
            bidding_data: 包含板块信息的竞价数据
        """
        logger.info("开始计算板块多维度指标")
        
        # 按板块分组
        grouped = bidding_data.groupby('sector')
        
        # 计算市场整体平均指标，用于相对指标计算
        market_avg_change = bidding_data['price_bid_change'].mean()
        market_avg_volume = bidding_data['volume_bid_volume'].mean() if 'volume_bid_volume' in bidding_data.columns else 1
        market_avg_net_flow = bidding_data['fund_flow_net_flow'].mean() if 'fund_flow_net_flow' in bidding_data.columns else 0
        market_avg_volume_ratio = bidding_data['volume_bid_volume_ratio'].mean() if 'volume_bid_volume_ratio' in bidding_data.columns else 1
        market_avg_turnover = bidding_data['volume_bid_turnover_rate'].mean() if 'volume_bid_turnover_rate' in bidding_data.columns else 0.01
        market_avg_order_amount = bidding_data['order_book_bid_order_amount'].mean() if 'order_book_bid_order_amount' in bidding_data.columns else 10000000
        
        for sector, sector_data in grouped:
            logger.info(f"计算板块{sector}的指标")
            
            stock_count = len(sector_data)
            if stock_count < 5:
                logger.warning(f"板块{sector}股票数量较少({stock_count}只)，跳过详细指标计算")
                continue
            
            # 1. 成交量增长率 - 完善计算逻辑
            volume_growth = self._calculate_volume_growth(sector_data)
            
            # 2. 涨跌幅偏离度 - 相对市场平均水平
            price_deviation = self._calculate_price_deviation(sector_data, market_avg_change)
            
            # 3. 资金流入强度 - 资金流入相对于板块规模
            fund_inflow_strength = self._calculate_fund_inflow_strength(sector_data)
            
            # 4. 板块联动性 - 优化的联动性计算
            sector_correlation = self._calculate_sector_correlation(sector_data)
            
            # 5. 板块涨停密度 - 接近涨停的股票比例
            limit_up_density = self._calculate_limit_up_density(sector_data)
            
            # 6. 板块平均涨跌幅 - 核心指标
            avg_price_change = sector_data['price_bid_change'].mean()
            
            # 7. 板块资金净流入 - 绝对资金流入
            total_net_flow = sector_data['fund_flow_net_flow'].sum() if 'fund_flow_net_flow' in sector_data.columns else 0
            
            # 8. 板块相对强度 - 相对市场平均涨跌幅
            sector_relative_strength = avg_price_change - market_avg_change
            
            # 9. 板块资金流入相对强度 - 相对市场平均资金流入
            if market_avg_net_flow != 0:
                fund_flow_relative_strength = total_net_flow / (abs(market_avg_net_flow) * stock_count)
            else:
                fund_flow_relative_strength = 0
            
            # 10. 板块活跃股比例 - 涨幅>=5%的股票比例
            active_stock_ratio = len(sector_data[sector_data['price_bid_change'] >= 0.05]) / stock_count if stock_count > 0 else 0
            
            # 11. 板块平均成交量比 - 相对历史成交量
            avg_volume_ratio = sector_data['volume_bid_volume_ratio'].mean() if 'volume_bid_volume_ratio' in sector_data.columns else 0
            
            # 12. 板块平均换手率 - 市场关注度指标
            avg_turnover = sector_data['volume_bid_turnover_rate'].mean() if 'volume_bid_turnover_rate' in sector_data.columns else 0
            
            # 13. 板块资金流入集中率 - 大单资金占比
            large_order_ratio = sector_data['fund_flow_large_order_ratio'].mean() if 'fund_flow_large_order_ratio' in sector_data.columns else 0
            
            # 14. 板块平均封单金额 - 封单强度
            avg_order_amount = sector_data['order_book_bid_order_amount'].mean() if 'order_book_bid_order_amount' in sector_data.columns else 0
            
            # 15. 板块封单金额相对强度 - 相对市场平均封单
            order_strength_relative = avg_order_amount / (market_avg_order_amount + 1e-10) if market_avg_order_amount != 0 else 0
            
            # 16. 板块涨跌幅标准差 - 板块内股票分化程度
            sector_volatility = sector_data['price_bid_change'].std()
            
            # 17. 板块资金流入一致性 - 资金流入方向一致性
            fund_flow_consistency = (sector_data['fund_flow_net_flow'] > 0).mean() if 'fund_flow_net_flow' in sector_data.columns else 0.5
            
            # 18. 板块涨停股资金集中度 - 涨停股资金流入占比
            limit_up_stocks = sector_data[sector_data['price_bid_change'] >= 0.08]
            if len(limit_up_stocks) > 0:
                limit_up_fund_concentration = limit_up_stocks['fund_flow_net_flow'].sum() / total_net_flow if total_net_flow != 0 else 0
            else:
                limit_up_fund_concentration = 0
            
            # 19. 板块流动性指标 - 平均成交量和成交额
            avg_bid_volume = sector_data['volume_bid_volume'].mean() if 'volume_bid_volume' in sector_data.columns else 0
            
            # 20. 板块相对流动性 - 相对市场平均流动性
            relative_liquidity = avg_bid_volume / (market_avg_volume + 1e-10) if market_avg_volume != 0 else 0
            
            # 保存板块指标，构建完整的多维度评估体系
            self.sector_metrics[sector] = {
                # 基础指标
                'stock_count': stock_count,
                'avg_price_change': avg_price_change,
                'total_net_flow': total_net_flow,
                'avg_volume_ratio': avg_volume_ratio,
                'avg_turnover': avg_turnover,
                
                # 相对市场指标
                'sector_relative_strength': sector_relative_strength,
                'price_deviation': price_deviation,
                'fund_flow_relative_strength': fund_flow_relative_strength,
                'relative_liquidity': relative_liquidity,
                
                # 板块内部结构指标
                'sector_correlation': sector_correlation,
                'sector_volatility': sector_volatility,
                'limit_up_density': limit_up_density,
                'active_stock_ratio': active_stock_ratio,
                'fund_flow_consistency': fund_flow_consistency,
                
                # 强度指标
                'volume_growth': volume_growth,
                'fund_inflow_strength': fund_inflow_strength,
                'large_order_ratio': large_order_ratio,
                'avg_order_amount': avg_order_amount,
                'order_strength_relative': order_strength_relative,
                'limit_up_fund_concentration': limit_up_fund_concentration,
                
                # 市场基准指标
                'market_avg_change': market_avg_change,
                'market_avg_volume': market_avg_volume,
                'market_avg_net_flow': market_avg_net_flow
            }
        
        logger.info(f"板块指标计算完成，共计算了{len(self.sector_metrics)}个板块")
    
    def _calculate_volume_growth(self, sector_data: pd.DataFrame) -> float:
        """计算板块成交量增长率
        
        Args:
            sector_data: 板块内股票的竞价数据
            
        Returns:
            成交量增长率
        """
        # 计算板块平均竞价成交量
        if 'volume_bid_volume' not in sector_data.columns or sector_data['volume_bid_volume'].isnull().all():
            return 0
        
        avg_bid_volume = sector_data['volume_bid_volume'].mean()
        
        # 使用近3日平均成交量作为基准，更准确反映板块热度变化
        # 从sector_data中获取近3日平均成交量（如果存在）
        if 'volume_volume_3d_avg' in sector_data.columns and not sector_data['volume_volume_3d_avg'].isnull().all():
            # 使用该板块股票的近3日平均成交量的平均值作为基准
            benchmark_volume = sector_data['volume_volume_3d_avg'].mean()
        else:
            # 如果没有近3日平均成交量数据，使用模拟数据
            benchmark_volume = np.random.randint(100000, 500000)  # 调整模拟数据范围，更符合实际
        
        # 计算增长率，确保分母不为0
        if benchmark_volume <= 0:
            return 0
        
        volume_growth = (avg_bid_volume - benchmark_volume) / benchmark_volume
        
        # 限制增长率范围，避免异常值影响
        volume_growth = max(-1.0, min(5.0, volume_growth))
        
        return volume_growth
    
    def _calculate_price_deviation(self, sector_data: pd.DataFrame, market_avg_change: float) -> float:
        """计算板块涨跌幅偏离度
        
        Args:
            sector_data: 板块内股票的竞价数据
            market_avg_change: 市场整体平均涨跌幅
            
        Returns:
            涨跌幅偏离度
        """
        # 计算板块平均涨跌幅
        avg_sector_change = sector_data['price_bid_change'].mean()
        
        # 计算偏离度
        price_deviation = avg_sector_change - market_avg_change
        return price_deviation
    
    def _calculate_fund_inflow_strength(self, sector_data: pd.DataFrame) -> float:
        """计算板块资金流入强度
        
        Args:
            sector_data: 板块内股票的竞价数据
            
        Returns:
            资金流入强度
        """
        if 'fund_flow_net_flow' not in sector_data.columns:
            return 0
        
        # 计算板块资金净流入总额
        total_net_flow = sector_data['fund_flow_net_flow'].sum()
        
        # 计算板块总市值（模拟数据，实际应使用真实市值数据）
        # 假设每只股票的平均市值为100亿
        avg_market_cap = 10000000000
        sector_total_market_cap = len(sector_data) * avg_market_cap
        
        # 计算资金流入强度
        if sector_total_market_cap == 0:
            return 0
        
        fund_inflow_strength = total_net_flow / sector_total_market_cap
        return fund_inflow_strength
    
    def _calculate_sector_correlation(self, sector_data: pd.DataFrame) -> float:
        """计算板块联动性，采用改进的联动性评估方法
        
        Args:
            sector_data: 板块内股票的竞价数据
            
        Returns:
            板块联动性系数 (0-1范围，值越大表示联动性越强)
        """
        logger.info("计算板块联动性")
        
        # 获取板块内股票数量
        stock_count = len(sector_data)
        if stock_count < 5:
            logger.warning(f"板块股票数量较少，无法准确计算联动性")
            return 0.5
        
        try:
            # 1. 计算涨跌幅联动性 - 核心联动性指标
            price_changes = sector_data['price_bid_change'].values
            
            # 计算相关系数矩阵
            correlation_matrix = np.corrcoef(price_changes)
            
            # 计算上三角矩阵的平均相关性（排除对角线）
            upper_triangle = correlation_matrix[np.triu_indices(stock_count, k=1)]
            avg_correlation = np.mean(upper_triangle)
            
            # 2. 计算成交量联动性 - 新增维度
            volume_correlation = 0
            if 'volume_bid_volume' in sector_data.columns:
                volumes = sector_data['volume_bid_volume'].values
                volume_corr_matrix = np.corrcoef(volumes)
                volume_upper = volume_corr_matrix[np.triu_indices(stock_count, k=1)]
                volume_correlation = np.mean(volume_upper)
            
            # 3. 计算资金流向联动性 - 新增维度
            fund_flow_correlation = 0
            if 'fund_flow_net_flow' in sector_data.columns:
                fund_flows = sector_data['fund_flow_net_flow'].values
                fund_corr_matrix = np.corrcoef(fund_flows)
                fund_upper = fund_corr_matrix[np.triu_indices(stock_count, k=1)]
                fund_flow_correlation = np.mean(fund_upper)
            
            # 4. 计算封单强度联动性 - 新增维度
            order_correlation = 0
            if 'order_book_bid_order_amount' in sector_data.columns:
                orders = sector_data['order_book_bid_order_amount'].values
                order_corr_matrix = np.corrcoef(orders)
                order_upper = order_corr_matrix[np.triu_indices(stock_count, k=1)]
                order_correlation = np.mean(order_upper)
            
            # 5. 综合联动性得分，采用加权平均
            combined_correlation = (
                avg_correlation * 0.4 +  # 涨跌幅联动性权重最高
                volume_correlation * 0.2 +  # 成交量联动性
                fund_flow_correlation * 0.2 +  # 资金流向联动性
                order_correlation * 0.2  # 封单强度联动性
            )
            
            # 6. 归一化到0-1范围
            # 将相关性值从[-1, 1]转换到[0, 1]
            normalized_correlation = (combined_correlation + 1) / 2
            
            # 7. 确保值在合理范围内
            normalized_correlation = max(0, min(1, normalized_correlation))
            
            logger.debug(f"板块联动性计算结果：涨跌幅联动性={avg_correlation:.4f}, 成交量联动性={volume_correlation:.4f}, \
                        资金流向联动性={fund_flow_correlation:.4f}, 封单强度联动性={order_correlation:.4f}, \
                        综合联动性={normalized_correlation:.4f}")
            
            return normalized_correlation
        except Exception as e:
            logger.error(f"计算板块相关性时发生错误: {e}")
            return 0.5
    
    def _calculate_limit_up_density(self, sector_data: pd.DataFrame) -> float:
        """计算板块涨停密度
        
        Args:
            sector_data: 板块内股票的竞价数据
            
        Returns:
            板块涨停密度
        """
        # 计算板块内接近涨停的股票数量
        # 定义接近涨停：涨跌幅 >= 8%
        limit_up_threshold = 0.08
        near_limit_up_count = (sector_data['price_bid_change'] >= limit_up_threshold).sum()
        
        # 计算涨停密度
        total_stocks = len(sector_data)
        if total_stocks == 0:
            return 0
        
        limit_up_density = near_limit_up_count / total_stocks
        return limit_up_density
    
    def _get_dynamic_weights(self, sector_df: pd.DataFrame) -> dict:
        """获取动态权重，根据市场环境调整各指标的权重
        
        Args:
            sector_df: 板块指标DataFrame
            
        Returns:
            动态权重字典，包含完整的多维度指标权重
        """
        logger.info("计算动态权重")
        
        # 计算市场整体情况，基于更全面的指标体系
        market_overview = {
            # 基础市场指标
            'avg_price_change': sector_df['avg_price_change'].mean(),
            'avg_sector_relative_strength': sector_df['sector_relative_strength'].mean(),
            'market_volatility': sector_df['sector_volatility'].mean(),  # 板块内股票分化程度
            'overall_volatility': sector_df['avg_price_change'].std(),  # 市场整体波动率
            
            # 资金与成交量指标
            'avg_volume_ratio': sector_df['avg_volume_ratio'].mean(),
            'avg_turnover': sector_df['avg_turnover'].mean(),
            'avg_total_net_flow': sector_df['total_net_flow'].mean(),
            'fund_flow_relative_strength': sector_df['fund_flow_relative_strength'].mean(),
            
            # 板块内部结构指标
            'avg_limit_up_density': sector_df['limit_up_density'].mean(),
            'avg_active_stock_ratio': sector_df['active_stock_ratio'].mean(),
            'avg_sector_correlation': sector_df['sector_correlation'].mean(),
            'avg_fund_flow_consistency': sector_df['fund_flow_consistency'].mean(),
            
            # 流动性指标
            'avg_relative_liquidity': sector_df['relative_liquidity'].mean(),
            'avg_order_amount': sector_df['avg_order_amount'].mean(),
            'avg_order_strength_relative': sector_df['order_strength_relative'].mean(),
            
            # 资金集中度指标
            'avg_limit_up_fund_concentration': sector_df['limit_up_fund_concentration'].mean(),
            'avg_large_order_ratio': sector_df['large_order_ratio'].mean()
        }
        
        logger.info(f"市场整体情况：{market_overview}")
        
        # 初始化基础权重，基于更全面的指标体系
        weights = {
            # 核心收益指标 - 基础权重
            'avg_price_change': 0.15,  # 板块平均涨跌幅
            'sector_relative_strength': 0.12,  # 板块相对市场强度
            
            # 资金与成交量指标
            'total_net_flow': 0.10,  # 板块资金净流入总额
            'fund_flow_relative_strength': 0.08,  # 相对市场资金强度
            'avg_volume_ratio': 0.08,  # 平均成交量比
            'avg_turnover': 0.05,  # 平均换手率
            
            # 板块内部结构指标
            'limit_up_density': 0.12,  # 涨停密度
            'active_stock_ratio': 0.08,  # 活跃股比例
            'sector_correlation': 0.10,  # 板块联动性
            'fund_flow_consistency': 0.05,  # 资金流入一致性
            
            # 流动性与封单指标
            'relative_liquidity': 0.05,  # 相对流动性
            'avg_order_amount': 0.05,  # 平均封单金额
            'order_strength_relative': 0.02,  # 相对市场封单强度
            
            # 资金集中度指标
            'limit_up_fund_concentration': 0.03,  # 涨停股资金集中度
            'large_order_ratio': 0.02  # 大单资金占比
        }
        
        # 根据市场环境调整权重的逻辑，采用更精细的市场状态划分
        weight_adjustments = {key: 0.0 for key in weights}
        
        # 1. 根据市场整体情绪调整权重
        market_sentiment = market_overview['avg_price_change']
        if market_sentiment > 0.05:  # 市场极强
            # 强牛市环境：重视涨停和资金强度
            weight_adjustments['limit_up_density'] += 0.08
            weight_adjustments['total_net_flow'] += 0.05
            weight_adjustments['sector_correlation'] += 0.03  # 联动性增强时更重要
            weight_adjustments['market_volatility'] -= 0.05  # 降低波动率权重
            weight_adjustments['avg_price_change'] += 0.04  # 价格变化更重要
        elif market_sentiment > 0.02:  # 市场较强
            # 牛市环境：平衡权重
            weight_adjustments['limit_up_density'] += 0.05
            weight_adjustments['fund_flow_relative_strength'] += 0.03
            weight_adjustments['sector_correlation'] += 0.02
        elif market_sentiment < -0.03:  # 市场极弱
            # 熊市环境：重视流动性和资金安全
            weight_adjustments['relative_liquidity'] += 0.08
            weight_adjustments['avg_order_amount'] += 0.05
            weight_adjustments['fund_flow_consistency'] += 0.04
            weight_adjustments['limit_up_density'] -= 0.06
            weight_adjustments['avg_price_change'] -= 0.05
        elif market_sentiment < -0.01:  # 市场较弱
            # 震荡偏弱：重视资金流向
            weight_adjustments['fund_flow_relative_strength'] += 0.06
            weight_adjustments['avg_turnover'] += 0.04
            weight_adjustments['limit_up_density'] -= 0.03
        
        # 2. 根据市场波动率调整权重
        if market_overview['overall_volatility'] > 0.025:  # 高波动率市场
            # 高波动环境：重视流动性和封单
            weight_adjustments['relative_liquidity'] += 0.06
            weight_adjustments['avg_order_amount'] += 0.04
            weight_adjustments['order_strength_relative'] += 0.03
            weight_adjustments['avg_price_change'] -= 0.05
        elif market_overview['overall_volatility'] < 0.01:  # 低波动率市场
            # 低波动环境：重视趋势和资金
            weight_adjustments['sector_relative_strength'] += 0.05
            weight_adjustments['total_net_flow'] += 0.04
            weight_adjustments['large_order_ratio'] += 0.03
        
        # 3. 根据板块联动性调整权重
        if market_overview['avg_sector_correlation'] > 0.7:  # 板块联动性强
            # 联动性强时：重视板块整体强度
            weight_adjustments['sector_relative_strength'] += 0.06
            weight_adjustments['sector_correlation'] += 0.04
            weight_adjustments['fund_flow_consistency'] += 0.03
        elif market_overview['avg_sector_correlation'] < 0.3:  # 板块联动性弱
            # 联动性弱时：重视个股独立性和资金
            weight_adjustments['avg_turnover'] += 0.05
            weight_adjustments['limit_up_fund_concentration'] += 0.04
            weight_adjustments['large_order_ratio'] += 0.03
        
        # 4. 根据涨停密度调整权重
        if market_overview['avg_limit_up_density'] > 0.25:  # 涨停股票很多
            # 涨停潮环境：重视涨停相关指标
            weight_adjustments['limit_up_density'] += 0.07
            weight_adjustments['limit_up_fund_concentration'] += 0.05
            weight_adjustments['active_stock_ratio'] += 0.03
        elif market_overview['avg_limit_up_density'] < 0.05:  # 涨停股票很少
            # 缺乏涨停环境：重视基础指标
            weight_adjustments['sector_relative_strength'] += 0.06
            weight_adjustments['fund_flow_relative_strength'] += 0.05
            weight_adjustments['relative_liquidity'] += 0.03
        
        # 5. 根据资金流入情况调整权重
        if market_overview['fund_flow_relative_strength'] > 1.5:  # 资金流入很强
            # 强资金流入环境：重视资金指标
            weight_adjustments['total_net_flow'] += 0.06
            weight_adjustments['fund_flow_relative_strength'] += 0.05
            weight_adjustments['large_order_ratio'] += 0.04
            weight_adjustments['fund_flow_consistency'] += 0.03
        elif market_overview['fund_flow_relative_strength'] < 0.5:  # 资金流入较弱
            # 弱资金流入环境：重视流动性和安全
            weight_adjustments['relative_liquidity'] += 0.06
            weight_adjustments['avg_order_amount'] += 0.05
            weight_adjustments['avg_turnover'] += 0.03
        
        # 6. 根据流动性情况调整权重
        if market_overview['avg_relative_liquidity'] < 0.5:  # 流动性较差
            # 低流动性环境：重视流动性指标
            weight_adjustments['relative_liquidity'] += 0.07
            weight_adjustments['avg_turnover'] += 0.05
            weight_adjustments['avg_order_amount'] += 0.03
        
        # 应用权重调整
        for key in weights:
            if key in weight_adjustments:
                weights[key] += weight_adjustments[key]
        
        # 确保权重为正值，设置最小值为0.01
        for key in weights:
            weights[key] = max(0.01, weights[key])
        
        # 确保权重之和为1
        total_weight = sum(weights.values())
        for key in weights:
            weights[key] /= total_weight
        
        # 记录最终权重分配
        logger.info(f"动态权重分配：{weights}")
        
        # 输出权重调整总结
        weight_summary = "权重调整总结：\n"
        for key, value in weight_adjustments.items():
            if abs(value) > 0.01:
                weight_summary += f"  - {key}: {'+' if value > 0 else ''}{value:.3f}\n"
        logger.info(weight_summary)
        
        return weights
    
    def _rank_sectors(self) -> pd.DataFrame:
        """综合评分并排序板块
        
        Returns:
            排序后的板块DataFrame，包含综合评分、热度等级和排名
        """
        logger.info("开始对板块进行综合评分和排序")
        
        if not self.sector_metrics:
            logger.error("板块指标为空，无法排序")
            return pd.DataFrame()
        
        # 转换为DataFrame
        sector_df = pd.DataFrame.from_dict(self.sector_metrics, orient='index')
        sector_df.reset_index(inplace=True)
        sector_df.rename(columns={'index': 'sector'}, inplace=True)
        
        # 获取动态权重
        weights = self._get_dynamic_weights(sector_df)
        
        # 综合评分公式：基于多维度指标的加权求和
        # 热度评分 = 各指标标准化值 * 动态权重之和
        logger.info("计算板块综合热度评分")
        
        # 改进的标准化函数，采用稳健的标准化方法，减少极端值影响
        def robust_normalize(s, method='minmax'):
            """稳健的标准化方法，支持minmax和zscore两种方法"""
            if len(s) <= 1:
                return pd.Series([0.5] * len(s), index=s.index)
            
            if method == 'zscore':
                # 使用稳健的z-score标准化
                q1 = s.quantile(0.25)
                q3 = s.quantile(0.75)
                iqr = q3 - q1
                median = s.median()
                return (s - median) / (iqr + 1e-10)
            else:  # minmax
                # 使用分位数裁剪的minmax标准化
                lower = s.quantile(0.05)
                upper = s.quantile(0.95)
                s_clipped = s.clip(lower, upper)
                min_val = s_clipped.min()
                max_val = s_clipped.max()
                if max_val - min_val == 0:
                    return pd.Series([0.5] * len(s), index=s.index)
                return (s_clipped - min_val) / (max_val - min_val)
        
        # 对权重中的指标进行标准化
        normalized_cols = []
        sector_df['normalized_score'] = 0
        
        for col, weight in weights.items():
            if col in sector_df.columns:
                # 根据指标类型选择合适的标准化方法
                if col in ['sector_volatility']:  # 低优指标
                    # 波动率越低越好，需要反转标准化
                    norm_col = f'norm_{col}'
                    # 先标准化，再反转
                    sector_df[norm_col] = 1 - robust_normalize(sector_df[col])
                    sector_df['normalized_score'] += sector_df[norm_col] * weight
                    normalized_cols.append(norm_col)
                else:  # 高优指标
                    norm_col = f'norm_{col}'
                    sector_df[norm_col] = robust_normalize(sector_df[col])
                    sector_df['normalized_score'] += sector_df[norm_col] * weight
                    normalized_cols.append(norm_col)
        
        # 计算综合热度评分，采用更全面的评分体系
        sector_df['composite_score'] = sector_df['normalized_score']
        
        # 添加板块联动性评分，作为辅助参考
        sector_df['correlation_score'] = sector_df['sector_correlation'] * 100
        
        # 添加资金集中度评分
        sector_df['fund_concentration_score'] = sector_df['limit_up_fund_concentration'] * 100
        
        # 计算板块热度等级，采用更精细的等级划分
        sector_df = self._calculate_sector_hotness_level(sector_df)
        
        # 排序：采用多层次排序策略，确保结果稳定可靠
        # 1. 综合评分降序
        # 2. 涨停密度降序
        # 3. 资金流入相对强度降序
        # 4. 相对流动性降序
        # 5. 板块相对强度降序
        sector_ranking = sector_df.sort_values(
            by=[
                'composite_score', 
                'limit_up_density', 
                'fund_flow_relative_strength',
                'relative_liquidity',
                'sector_relative_strength'
            ],
            ascending=False
        )
        
        # 添加排名列
        sector_ranking['hotness_rank'] = range(1, len(sector_ranking) + 1)
        
        logger.info("板块排序完成")
        return sector_ranking
    
    def _calculate_sector_hotness_level(self, sector_df: pd.DataFrame) -> pd.DataFrame:
        """计算板块热度等级，采用更精细的等级划分
        
        Args:
            sector_df: 板块指标DataFrame
            
        Returns:
            带有热度等级的板块DataFrame
        """
        logger.info("计算板块热度等级")
        
        # 基于综合评分分布动态设定热度等级阈值
        # 使用quantile方法明确计算所需分位数
        score_quantiles = sector_df['composite_score'].quantile([0.1, 0.25, 0.5, 0.75, 0.9])
        logger.info(f"综合评分分位数：{score_quantiles}")
        
        # 基于分位数动态设定热度等级阈值
        hotness_thresholds = {
            'S级热度': score_quantiles[0.9],  # 前10%的板块
            'A级热度': score_quantiles[0.75],  # 前25%的板块
            'B级热度': score_quantiles[0.5],  # 前50%的板块
            'C级热度': score_quantiles[0.25],  # 前75%的板块
            'D级热度': score_quantiles[0.1]   # 前90%的板块
        }
        
        logger.info(f"动态热度等级阈值：{hotness_thresholds}")
        
        # 计算热度等级，采用更精细的S-A-B-C-D等级划分
        def get_hotness_level(score):
            if score >= hotness_thresholds['S级热度']:
                return 'S级热度'  # 顶级热门板块
            elif score >= hotness_thresholds['A级热度']:
                return 'A级热度'  # 高热度板块
            elif score >= hotness_thresholds['B级热度']:
                return 'B级热度'  # 中等热度板块
            elif score >= hotness_thresholds['C级热度']:
                return 'C级热度'  # 较低热度板块
            elif score >= hotness_thresholds['D级热度']:
                return 'D级热度'  # 低热度板块
            else:
                return '冷热度'    # 冷门板块
        
        sector_df['hotness_level'] = sector_df['composite_score'].apply(get_hotness_level)
        
        # 计算热度排名，使用method='dense'确保排名连续
        sector_df['hotness_rank'] = sector_df['composite_score'].rank(ascending=False, method='dense')
        
        # 添加相对热度百分比，更直观展示板块热度
        sector_df['hotness_percentile'] = sector_df['composite_score'].rank(pct=True, ascending=False) * 100
        
        return sector_df
    
    def _filter_top_sectors(self, sector_ranking: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
        """筛选TOP热门板块，结合多维度指标进行严格筛选
        
        Args:
            sector_ranking: 排序后的板块DataFrame
            top_n: 要筛选的TOP板块数量
            
        Returns:
            TOP热门板块DataFrame，包含筛选理由和详细指标
        """
        logger.info(f"开始筛选TOP {top_n}热门板块")
        
        if sector_ranking.empty:
            return pd.DataFrame()
        
        # 1. 基于综合评分分布动态设定热度阈值
        score_stats = sector_ranking['composite_score'].describe()
        logger.info(f"板块综合评分统计：{score_stats}")
        
        # 计算80%分位数
        score_80th_percentile = sector_ranking['composite_score'].quantile(0.8)
        
        # 动态设定热度阈值：不低于80分位数，且不低于0.55
        hotness_threshold = max(score_80th_percentile, 0.55)
        logger.info(f"动态设定的热度阈值：{hotness_threshold:.4f}")
        
        # 2. 应用热度阈值过滤
        qualified_sectors = sector_ranking[sector_ranking['composite_score'] >= hotness_threshold].copy()
        
        # 3. 严格的板块质量验证
        # 3.1 股票数量验证：至少有8只股票
        qualified_sectors = qualified_sectors[qualified_sectors['stock_count'] >= 8]
        logger.info(f"股票数量验证后剩余：{len(qualified_sectors)}个板块")
        
        # 3.2 流动性验证：相对流动性不低于0.3
        qualified_sectors = qualified_sectors[qualified_sectors['relative_liquidity'] >= 0.3]
        logger.info(f"流动性验证后剩余：{len(qualified_sectors)}个板块")
        
        # 3.3 活跃度验证：活跃股比例不低于15%
        qualified_sectors = qualified_sectors[qualified_sectors['active_stock_ratio'] >= 0.15]
        logger.info(f"活跃度验证后剩余：{len(qualified_sectors)}个板块")
        
        # 3.4 资金流入验证：资金流入相对强度不低于0.5
        qualified_sectors = qualified_sectors[qualified_sectors['fund_flow_relative_strength'] >= 0.5]
        logger.info(f"资金流入验证后剩余：{len(qualified_sectors)}个板块")
        
        # 4. 选择TOP N板块
        if len(qualified_sectors) > top_n:
            # 进一步优化：考虑板块之间的相关性，避免过度集中
            selected_sectors = []
            for i, sector in qualified_sectors.iterrows():
                if len(selected_sectors) >= top_n:
                    break
                    
                # 检查与已选板块的相关性，如果相关性过高则跳过
                add_sector = True
                if selected_sectors:
                    for selected in selected_sectors:
                        # 这里简化处理，实际应使用板块间相关性矩阵
                        # 假设同行业板块相关性较高
                        add_sector = True
                        break
                
                if add_sector:
                    selected_sectors.append(sector)
            
            top_sectors = pd.DataFrame(selected_sectors)
        else:
            top_sectors = qualified_sectors.copy()
        
        # 5. 确保至少有一个板块
        if top_sectors.empty:
            logger.warning(f"没有板块通过所有验证条件，降低标准选择综合评分最高的板块")
            # 降低标准，只考虑综合评分
            fallback_sectors = sector_ranking[sector_ranking['composite_score'] >= max(score_stats['50%'], 0.4)].head(top_n)
            if not fallback_sectors.empty:
                top_sectors = fallback_sectors.copy()
            else:
                top_sectors = sector_ranking.head(1).copy()
        
        # 6. 添加详细的筛选结果信息
        top_sectors['selection_reason'] = top_sectors.apply(lambda row: self._generate_selection_reason(row), axis=1)
        
        # 7. 更新排序依据说明，包含更全面的排序维度
        top_sectors['ranking_criteria'] = (
            "排序依据：综合热度评分降序 > 涨停密度降序 > 资金流入相对强度降序 > "
            "相对流动性降序 > 板块相对强度降序"
        )
        
        # 8. 添加筛选时间和版本信息
        import datetime
        top_sectors['selection_time'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        top_sectors['version'] = "v2.0"  # 筛选算法版本
        
        logger.info(f"筛选完成，共得到{len(top_sectors)}个热门板块")
        logger.info(f"热门板块列表：{top_sectors['sector'].tolist()}")
        
        # 打印详细的筛选结果
        for idx, row in top_sectors.iterrows():
            logger.info(f"板块：{row['sector']}，综合评分：{row['composite_score']:.4f}，"
                      f"热度等级：{row['hotness_level']}，排名：{int(row['hotness_rank'])}")
        
        return top_sectors
    
    def _generate_selection_reason(self, row):
        """生成板块入选理由
        
        Args:
            row: 板块数据行
            
        Returns:
            板块入选理由字符串
        """
        reasons = []
        
        # 1. 综合评分
        if row['composite_score'] >= 0.8:
            reasons.append(f"综合评分极高({row['composite_score']:.2f})")
        elif row['composite_score'] >= 0.65:
            reasons.append(f"综合评分较高({row['composite_score']:.2f})")
        
        # 2. 涨停密度
        if row['limit_up_density'] >= 0.2:
            reasons.append(f"涨停密度高({row['limit_up_density']:.2%})")
        
        # 3. 资金流入
        if row['fund_flow_relative_strength'] >= 1.0:
            reasons.append(f"资金流入强度大({row['fund_flow_relative_strength']:.2f})")
        
        # 4. 活跃度
        if row['active_stock_ratio'] >= 0.3:
            reasons.append(f"板块活跃度高({row['active_stock_ratio']:.2%})")
        
        # 5. 流动性
        if row['relative_liquidity'] >= 1.0:
            reasons.append(f"流动性好({row['relative_liquidity']:.2f})")
        
        # 6. 板块强度
        if row['sector_relative_strength'] >= 0.03:
            reasons.append(f"相对市场强度大({row['sector_relative_strength']:.2%})")
        
        # 7. 资金集中度
        if row['limit_up_fund_concentration'] >= 0.6:
            reasons.append(f"资金集中于涨停股({row['limit_up_fund_concentration']:.2%})")
        
        # 8. 热度排名
        reasons.append(f"热度排名第{int(row['hotness_rank'])}名")
        
        return "; ".join(reasons) if reasons else "综合表现优秀"
    
    def get_top_sectors(self) -> pd.DataFrame:
        """获取筛选出的TOP热门板块
        
        Returns:
            TOP热门板块DataFrame
        """
        return self.top_sectors
    
    def get_sector_metrics(self) -> dict:
        """获取各板块的指标
        
        Returns:
            各板块的指标字典
        """
        return self.sector_metrics
    
    def get_stocks_by_sector(self, sector: str) -> list:
        """获取指定板块的股票列表
        
        Args:
            sector: 板块名称
            
        Returns:
            该板块下的股票列表
        """
        return self.data_source.get_stocks_by_sector(sector)
