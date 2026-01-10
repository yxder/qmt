#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
个股筛选模块
在已识别的热门板块内，建立个股精选模型，设定严格的筛选条件，形成个股候选池
"""

import pandas as pd
import numpy as np
from utils.logger import setup_logger
from data.data_source import DataSource

logger = setup_logger()


class StockFilter:
    """个股筛选器，用于在热门板块内筛选优质个股"""
    
    def __init__(self):
        """初始化个股筛选器"""
        logger.info("初始化个股筛选器")
        self.data_source = DataSource()
        self.candidate_stocks = []
    
    def filter_stocks(self, bidding_data: pd.DataFrame, top_sectors: pd.DataFrame, historical_data: dict = None) -> pd.DataFrame:
        """在热门板块内筛选个股
        
        Args:
            bidding_data: 集合竞价数据
            top_sectors: TOP热门板块DataFrame
            historical_data: 历史数据字典，键为股票代码，值为历史数据DataFrame
            
        Returns:
            个股候选池DataFrame
        """
        logger.info("开始在热门板块内筛选个股")
        
        if bidding_data.empty or top_sectors.empty:
            logger.error("输入数据为空，无法筛选个股")
            return pd.DataFrame()
        
        # 1. 提取热门板块列表
        hot_sectors = top_sectors['sector'].tolist()
        logger.info(f"热门板块列表：{hot_sectors}")
        
        # 2. 在热门板块内筛选个股
        candidate_stocks = []
        
        for sector in hot_sectors:
            logger.info(f"在板块{sector}内筛选个股")
            
            # 获取该板块的股票
            sector_stocks = bidding_data[bidding_data['sector'] == sector]
            
            if sector_stocks.empty:
                logger.warning(f"板块{sector}内没有股票，跳过")
                continue
            
            # 应用筛选条件
            filtered_stocks = self._apply_filter_conditions(sector_stocks, historical_data)
            
            if not filtered_stocks.empty:
                candidate_stocks.append(filtered_stocks)
        
        # 合并所有候选股票
        if candidate_stocks:
            self.candidate_stocks = pd.concat(candidate_stocks, ignore_index=True)
        else:
            self.candidate_stocks = pd.DataFrame()
        
        logger.info(f"个股筛选完成，共筛选出{len(self.candidate_stocks)}只候选股票")
        return self.candidate_stocks
    
    def _get_dynamic_filter_thresholds(self, sector_stocks: pd.DataFrame) -> dict:
        """获取动态筛选阈值，明确各筛选条件的严格数值标准
        
        Args:
            sector_stocks: 板块内的股票数据
            
        Returns:
            动态筛选阈值字典，包含明确的数值标准和调整逻辑
        """
        logger.info("计算动态筛选阈值")
        
        # 计算更全面的市场环境指标
        market_environment = self._calculate_market_environment(sector_stocks)
        
        # 初始化基础筛选阈值，明确各条件的具体数值标准
        thresholds = {
            # 1. 价格异动幅度标准 - 明确异动幅度定义
            'price_bid_change_lower': 0.06,  # 早盘价格异动下限：6%
            'price_bid_change_upper': 0.098,  # 早盘价格异动上限：9.8%（避免已涨停股票）
            
            # 2. 成交量放大标准 - 明确放大比例阈值
            'volume_bid_volume_ratio': 2.0,  # 竞价成交量比：至少2倍
            'volume_3d_growth_ratio': 0.4,  # 近3日成交量每日增长率：至少40%
            'volume_growth_consistency': 3,  # 近3日中满足增长条件的天数：全部3天
            'volume_3d_total_growth': 2.0,  # 近3日总成交量放大倍数：至少2倍
            
            # 3. 流通盘规模适中 - 设定具体数值范围
            'circulating_market_cap_lower': 3000000000,  # 流通盘下限：30亿
            'circulating_market_cap_upper': 15000000000,  # 流通盘上限：150亿
            
            # 4. 市场关注度指标 - 明确换手率、舆情热度等标准
            'volume_bid_turnover_rate_lower': 0.008,  # 竞价换手率下限：0.8%
            'volume_bid_turnover_rate_upper': 0.12,  # 竞价换手率上限：12%（避免过高换手率）
            'prev_day_turnover_rate_lower': 0.02,  # 前一日换手率下限：2%
            'bid_turnover_rate_growth': 0.5,  # 竞价换手率较前一日增长：至少50%
            
            # 5. 委托单强度指标 - 明确封单强度标准
            'order_book_bid_order_amount': 15000000,  # 封单金额：1500万以上
            'order_book_bid_order_ratio': 0.2,  # 封单比例：20%以上
            'order_imbalance_threshold': 0.15,  # 委托单失衡阈值：15%
            'avg_bid_order_size': 50000,  # 平均买单大小：5万股以上
            
            # 6. 资金流入指标 - 明确资金强度标准
            'fund_flow_net_flow_lower': 8000000,  # 资金净流入：800万以上
            'large_order_ratio_lower': 0.65,  # 大单资金占比：65%以上
            'fund_flow_consistency': 0.6,  # 资金流入强度一致性：60%
            'net_flow_per_volume': 10,  # 单位成交量资金净流入：10元以上
            'large_order_flow_lower': 5000000,  # 大单资金流入：500万以上
            
            # 7. 技术面指标 - 新增技术筛选条件
            'price_volatility_upper': 0.05,  # 5日价格波动率上限：5%
            'trend_5d_lower': 0.05,  # 5日趋势下限：5%
            'bullish_ma_arrangement': 1,  # 均线多头排列
            
            # 8. 综合指标 - 明确评分标准
            'stock_score_threshold': 0.7,  # 个股综合评分阈值：70分
            'prediction_confidence_threshold': 0.75  # 预测置信度阈值：75%
        }
        
        # 根据市场环境动态调整阈值，采用更精细的调整策略
        logger.info(f"市场环境指标：{market_environment}")
        
        # 1. 根据板块涨势调整价格异动阈值
        if market_environment['sector_avg_change'] > 0.09:  # 板块整体极强
            thresholds['price_bid_change_lower'] = 0.05  # 放宽价格异动下限到5%
            thresholds['price_bid_change_upper'] = 0.099  # 放宽价格异动上限到9.9%
        elif market_environment['sector_avg_change'] > 0.06:  # 板块整体强势
            thresholds['price_bid_change_lower'] = 0.055  # 放宽价格异动下限到5.5%
            thresholds['price_bid_change_upper'] = 0.098  # 维持价格异动上限
        elif market_environment['sector_avg_change'] < 0.03:  # 板块整体弱势
            thresholds['price_bid_change_lower'] = 0.07  # 提高价格异动下限到7%
            thresholds['price_bid_change_upper'] = 0.095  # 收紧价格异动上限到9.5%
        
        # 2. 根据板块成交量比调整成交量阈值
        if market_environment['sector_avg_volume_ratio'] > 4.0:  # 板块成交量非常活跃
            thresholds['volume_bid_volume_ratio'] = 1.8  # 适度放宽成交量比到1.8倍
            thresholds['volume_3d_growth_ratio'] = 0.35  # 适度降低增长率阈值到35%
        elif market_environment['sector_avg_volume_ratio'] > 2.5:  # 板块成交量活跃
            thresholds['volume_bid_volume_ratio'] = 1.9  # 适度放宽成交量比到1.9倍
            thresholds['volume_3d_growth_ratio'] = 0.38  # 适度降低增长率阈值到38%
        elif market_environment['sector_avg_volume_ratio'] < 1.2:  # 板块成交量低迷
            thresholds['volume_bid_volume_ratio'] = 2.2  # 提高成交量比到2.2倍
            thresholds['volume_3d_growth_ratio'] = 0.45  # 提高增长率阈值到45%
        
        # 3. 根据板块换手率调整换手率阈值
        if market_environment['sector_avg_turnover'] > 0.04:  # 板块换手率极高
            thresholds['volume_bid_turnover_rate_lower'] = 0.01  # 提高换手率下限到1%
            thresholds['volume_bid_turnover_rate_upper'] = 0.15  # 放宽换手率上限到15%
        elif market_environment['sector_avg_turnover'] > 0.02:  # 板块换手率较高
            thresholds['volume_bid_turnover_rate_lower'] = 0.009  # 提高换手率下限到0.9%
            thresholds['volume_bid_turnover_rate_upper'] = 0.13  # 放宽换手率上限到13%
        elif market_environment['sector_avg_turnover'] < 0.008:  # 板块换手率极低
            thresholds['volume_bid_turnover_rate_lower'] = 0.006  # 降低换手率下限到0.6%
            thresholds['volume_bid_turnover_rate_upper'] = 0.10  # 收紧换手率上限到10%
        
        # 4. 根据板块资金流入调整资金流入阈值
        if market_environment['sector_avg_net_flow'] > 80000000:  # 板块资金流入极强
            thresholds['fund_flow_net_flow_lower'] = 12000000  # 提高资金净流入门槛到1200万
            thresholds['large_order_ratio_lower'] = 0.7  # 提高大单资金占比到70%
            thresholds['large_order_flow_lower'] = 8000000  # 提高大单资金流入到800万
        elif market_environment['sector_avg_net_flow'] > 40000000:  # 板块资金流入较强
            thresholds['fund_flow_net_flow_lower'] = 10000000  # 提高资金净流入门槛到1000万
            thresholds['large_order_ratio_lower'] = 0.68  # 提高大单资金占比到68%
            thresholds['large_order_flow_lower'] = 6000000  # 提高大单资金流入到600万
        elif market_environment['sector_avg_net_flow'] < 10000000:  # 板块资金流入较弱
            thresholds['fund_flow_net_flow_lower'] = 6000000  # 降低资金净流入门槛到600万
            thresholds['large_order_ratio_lower'] = 0.62  # 降低大单资金占比到62%
            thresholds['large_order_flow_lower'] = 3000000  # 降低大单资金流入到300万
        
        # 5. 根据板块封单情况调整封单阈值
        if market_environment['sector_avg_order_amount'] > 30000000:  # 板块封单极强
            thresholds['order_book_bid_order_amount'] = 20000000  # 提高封单金额门槛到2000万
            thresholds['order_book_bid_order_ratio'] = 0.25  # 提高封单比例到25%
        elif market_environment['sector_avg_order_amount'] > 15000000:  # 板块封单较强
            thresholds['order_book_bid_order_amount'] = 18000000  # 提高封单金额门槛到1800万
            thresholds['order_book_bid_order_ratio'] = 0.22  # 提高封单比例到22%
        elif market_environment['sector_avg_order_amount'] < 8000000:  # 板块封单较弱
            thresholds['order_book_bid_order_amount'] = 12000000  # 降低封单金额门槛到1200万
            thresholds['order_book_bid_order_ratio'] = 0.18  # 降低封单比例到18%
        
        # 6. 根据市场波动率调整技术面阈值
        if market_environment['price_volatility'] > 0.03:  # 市场波动率较高
            thresholds['price_volatility_upper'] = 0.06  # 放宽价格波动率上限到6%
        elif market_environment['price_volatility'] < 0.01:  # 市场波动率较低
            thresholds['price_volatility_upper'] = 0.04  # 收紧价格波动率上限到4%
        
        logger.info(f"动态筛选阈值：{thresholds}")
        return thresholds
    
    def _calculate_market_environment(self, sector_stocks: pd.DataFrame) -> dict:
        """计算更全面的市场环境指标，用于动态调整筛选阈值
        
        Args:
            sector_stocks: 板块内的股票数据
            
        Returns:
            市场环境指标字典
        """
        # 1. 板块平均涨跌幅
        sector_avg_change = sector_stocks['price_bid_change'].mean()
        
        # 2. 板块平均成交量比
        sector_avg_volume_ratio = sector_stocks['volume_bid_volume_ratio'].mean() if 'volume_bid_volume_ratio' in sector_stocks.columns else 1.0
        
        # 3. 板块平均换手率
        sector_avg_turnover = sector_stocks['volume_bid_turnover_rate'].mean() if 'volume_bid_turnover_rate' in sector_stocks.columns else 0.01
        
        # 4. 板块平均资金流入
        sector_avg_net_flow = sector_stocks['fund_flow_net_flow'].mean() if 'fund_flow_net_flow' in sector_stocks.columns else 0
        
        # 5. 板块平均封单金额
        sector_avg_order_amount = sector_stocks['order_book_bid_order_amount'].mean() if 'order_book_bid_order_amount' in sector_stocks.columns else 10000000
        
        # 6. 板块平均大单资金占比
        sector_avg_large_order_ratio = sector_stocks['fund_flow_large_order_ratio'].mean() if 'fund_flow_large_order_ratio' in sector_stocks.columns else 0.5
        
        # 7. 板块价格波动率
        price_volatility = sector_stocks['price_bid_change'].std()
        
        # 8. 板块资金流入一致性
        fund_flow_consistency = (sector_stocks['fund_flow_net_flow'] > 0).mean() if 'fund_flow_net_flow' in sector_stocks.columns else 0.5
        
        # 9. 板块活跃度
        active_stock_ratio = (sector_stocks['price_bid_change'] >= 0.05).mean()
        
        return {
            'sector_avg_change': sector_avg_change,
            'sector_avg_volume_ratio': sector_avg_volume_ratio,
            'sector_avg_turnover': sector_avg_turnover,
            'sector_avg_net_flow': sector_avg_net_flow,
            'sector_avg_order_amount': sector_avg_order_amount,
            'sector_avg_large_order_ratio': sector_avg_large_order_ratio,
            'price_volatility': price_volatility,
            'fund_flow_consistency': fund_flow_consistency,
            'active_stock_ratio': active_stock_ratio
        }
    
    def _apply_filter_conditions(self, sector_stocks: pd.DataFrame, historical_data: dict = None) -> pd.DataFrame:
        """应用筛选条件
        
        Args:
            sector_stocks: 板块内的股票数据
            historical_data: 历史数据字典
            
        Returns:
            筛选后的股票DataFrame
        """
        logger.info("应用个股筛选条件")
        
        # 获取动态筛选阈值
        thresholds = self._get_dynamic_filter_thresholds(sector_stocks)
        
        initial_count = len(sector_stocks)
        logger.info(f"初始股票数量：{initial_count}")
        
        # 1. 近3日成交量持续放大
        # 明确放大比例阈值：每日成交量增长率不低于30%，且近3日中至少2天满足增长条件
        if historical_data:
            logger.info(f"应用近3日成交量持续放大条件，增长率阈值：{thresholds['volume_3d_growth_ratio']:.0%}，一致性要求：{thresholds['volume_growth_consistency']}天")
            sector_stocks = sector_stocks[sector_stocks['stock_code'].apply(
                lambda x: self._check_volume_growth(x, historical_data, thresholds['volume_3d_growth_ratio'], thresholds['volume_growth_consistency'])
            )]
            logger.info(f"近3日成交量持续放大筛选后剩余：{len(sector_stocks)}只股票")
        
        if sector_stocks.empty:
            logger.warning("成交量增长筛选后无股票剩余")
            return sector_stocks
        
        # 2. 早盘价格出现显著异动
        # 明确异动幅度标准：5%-9.5%
        logger.info(f"应用早盘价格异动条件，幅度范围：{thresholds['price_bid_change_lower']:.0%} - {thresholds['price_bid_change_upper']:.1%}")
        sector_stocks = sector_stocks[
            (sector_stocks['price_bid_change'] >= thresholds['price_bid_change_lower']) & 
            (sector_stocks['price_bid_change'] < thresholds['price_bid_change_upper'])
        ]
        
        logger.info(f"价格异动筛选后剩余：{len(sector_stocks)}只股票")
        if sector_stocks.empty:
            logger.warning("价格异动筛选后无股票剩余")
            return sector_stocks
        
        # 3. 流通盘规模适中
        # 明确数值范围：50亿 <= 流通市值 <= 200亿
        logger.info(f"应用流通盘规模条件，范围：{thresholds['circulating_market_cap_lower']/1e8:.0f}亿 - {thresholds['circulating_market_cap_upper']/1e8:.0f}亿")
        sector_stocks['circulating_market_cap'] = sector_stocks['stock_code'].apply(
            lambda x: self._get_circulating_market_cap(x)
        )
        sector_stocks = sector_stocks[
            (sector_stocks['circulating_market_cap'] >= thresholds['circulating_market_cap_lower']) & 
            (sector_stocks['circulating_market_cap'] <= thresholds['circulating_market_cap_upper'])
        ]
        
        logger.info(f"流通盘筛选后剩余：{len(sector_stocks)}只股票")
        if sector_stocks.empty:
            logger.warning("流通盘筛选后无股票剩余")
            return sector_stocks
        
        # 4. 市场关注度指标达标
        # 明确标准：竞价换手率0.5%-8%
        logger.info(f"应用市场关注度条件，换手率范围：{thresholds['volume_bid_turnover_rate_lower']:.1%} - {thresholds['volume_bid_turnover_rate_upper']:.0%}")
        sector_stocks = sector_stocks[
            (sector_stocks['volume_bid_turnover_rate'] >= thresholds['volume_bid_turnover_rate_lower']) & 
            (sector_stocks['volume_bid_turnover_rate'] < thresholds['volume_bid_turnover_rate_upper'])
        ]
        
        logger.info(f"换手率筛选后剩余：{len(sector_stocks)}只股票")
        if sector_stocks.empty:
            logger.warning("换手率筛选后无股票剩余")
            return sector_stocks
        
        # 5. 竞价强度指标
        # 明确标准：竞价成交量比1.5倍以上
        logger.info(f"应用竞价强度条件，成交量比阈值：{thresholds['volume_bid_volume_ratio']:.1f}倍")
        sector_stocks = sector_stocks[sector_stocks['volume_bid_volume_ratio'] >= thresholds['volume_bid_volume_ratio']]
        
        logger.info(f"成交量比筛选后剩余：{len(sector_stocks)}只股票")
        if sector_stocks.empty:
            logger.warning("成交量比筛选后无股票剩余")
            return sector_stocks
        
        # 6. 委托单失衡指标
        # 明确标准：委托单失衡度10%以上（买单量显著大于卖单量）
        logger.info(f"应用委托单失衡条件，失衡度阈值：{thresholds['order_imbalance_threshold']:.0%}")
        if 'order_book_order_imbalance' in sector_stocks.columns:
            sector_stocks = sector_stocks[sector_stocks['order_book_order_imbalance'] >= thresholds['order_imbalance_threshold']]
        else:
            # 计算委托单失衡度
            sector_stocks['order_imbalance'] = (sector_stocks['order_book_bid_order_amount'] - sector_stocks['order_book_ask_order_amount']) / \
                                             (sector_stocks['order_book_bid_order_amount'] + sector_stocks['order_book_ask_order_amount'] + 1e-10)
            sector_stocks = sector_stocks[sector_stocks['order_imbalance'] >= thresholds['order_imbalance_threshold']]
        
        logger.info(f"委托单失衡筛选后剩余：{len(sector_stocks)}只股票")
        if sector_stocks.empty:
            logger.warning("委托单失衡筛选后无股票剩余")
            return sector_stocks
        
        # 7. 封单量指标
        # 明确标准：封单金额1000万以上
        logger.info(f"应用封单量条件，封单金额阈值：{thresholds['order_book_bid_order_amount']/1e4:.0f}万")
        sector_stocks = sector_stocks[sector_stocks['order_book_bid_order_amount'] >= thresholds['order_book_bid_order_amount']]
        
        logger.info(f"封单金额筛选后剩余：{len(sector_stocks)}只股票")
        if sector_stocks.empty:
            logger.warning("封单金额筛选后无股票剩余")
            return sector_stocks
        
        # 8. 封单比例指标
        # 明确标准：封单比例15%以上
        logger.info(f"应用封单比例条件，封单比例阈值：{thresholds['order_book_bid_order_ratio']:.0%}")
        sector_stocks = sector_stocks[sector_stocks['order_book_bid_order_ratio'] >= thresholds['order_book_bid_order_ratio']]
        
        logger.info(f"封单比例筛选后剩余：{len(sector_stocks)}只股票")
        if sector_stocks.empty:
            logger.warning("封单比例筛选后无股票剩余")
            return sector_stocks
        
        # 9. 资金流入指标
        # 明确标准：资金净流入500万以上
        if 'fund_flow_net_flow' in sector_stocks.columns:
            logger.info(f"应用资金流入条件，资金净流入阈值：{thresholds['fund_flow_net_flow_lower']/1e4:.0f}万")
            sector_stocks = sector_stocks[sector_stocks['fund_flow_net_flow'] >= thresholds['fund_flow_net_flow_lower']]
            logger.info(f"资金流入筛选后剩余：{len(sector_stocks)}只股票")
        
        if sector_stocks.empty:
            logger.warning("资金流入筛选后无股票剩余")
            return sector_stocks
        
        # 10. 大单资金占比指标
        # 明确标准：大单资金占比60%以上
        if 'fund_flow_large_order_flow' in sector_stocks.columns and 'fund_flow_net_flow' in sector_stocks.columns:
            logger.info(f"应用大单资金占比条件，大单占比阈值：{thresholds['large_order_ratio_lower']:.0%}")
            # 计算大单资金占比
            sector_stocks['large_order_ratio'] = sector_stocks['fund_flow_large_order_flow'] / (sector_stocks['fund_flow_net_flow'] + 1e-10)
            sector_stocks = sector_stocks[sector_stocks['large_order_ratio'] >= thresholds['large_order_ratio_lower']]
            logger.info(f"大单资金占比筛选后剩余：{len(sector_stocks)}只股票")
        
        # 11. 资金流入强度一致性
        # 明确标准：资金流入强度一致性50%以上
        if 'fund_flow_net_flow_growth_rate' in sector_stocks.columns:
            logger.info(f"应用资金流入强度一致性条件，一致性阈值：{thresholds['fund_flow_consistency']:.0%}")
            sector_stocks = sector_stocks[sector_stocks['fund_flow_net_flow_growth_rate'] >= thresholds['fund_flow_consistency']]
            logger.info(f"资金流入强度一致性筛选后剩余：{len(sector_stocks)}只股票")
        
        logger.info(f"最终筛选完成，剩余{len(sector_stocks)}只股票")
        return sector_stocks
    
    def _check_volume_growth(self, stock_code: str, historical_data: dict, growth_threshold: float = 0.3, growth_consistency: int = 2) -> bool:
        """检查近3日成交量是否持续放大
        
        Args:
            stock_code: 股票代码
            historical_data: 历史数据字典
            growth_threshold: 成交量增长率阈值，默认为30%
            growth_consistency: 近3日中至少满足增长条件的天数，默认为2天
            
        Returns:
            近3日成交量是否持续放大
        """
        # 获取该股票的历史数据
        if stock_code not in historical_data:
            logger.warning(f"股票{stock_code}没有历史数据，跳过成交量增长检查")
            return False  # 没有历史数据时，不通过该条件
        
        hist_data = historical_data[stock_code]
        
        # 检查数据长度是否足够
        if len(hist_data) < 3:
            logger.warning(f"股票{stock_code}历史数据不足3天，跳过成交量增长检查")
            return False  # 历史数据不足时，不通过该条件
        
        # 确保有volume列
        if 'volume' not in hist_data.columns:
            logger.warning(f"股票{stock_code}历史数据缺少volume列，跳过成交量增长检查")
            return False
        
        # 取近3日的成交量，确保数据有效
        recent_volumes = hist_data['volume'].tail(3).values
        
        # 检查是否有异常值或缺失值
        if np.isnan(recent_volumes).any() or (recent_volumes == 0).any():
            logger.warning(f"股票{stock_code}成交量数据存在异常，跳过成交量增长检查")
            return False
        
        # 检查是否持续放大
        # 定义持续放大：每日成交量增长率 >= growth_threshold
        growth_days = 0
        total_growth_rate = 0
        
        for i in range(1, len(recent_volumes)):
            growth_rate = (recent_volumes[i] - recent_volumes[i-1]) / recent_volumes[i-1]
            if growth_rate >= growth_threshold:
                growth_days += 1
            total_growth_rate += growth_rate
        
        # 1. 检查满足增长条件的天数是否达到要求
        days_condition = growth_days >= growth_consistency
        
        # 2. 检查整体趋势是否向上（总增长率为正）
        trend_condition = total_growth_rate > 0
        
        # 3. 检查最后一天是否增长（确保最新趋势向上）
        last_day_growth = (recent_volumes[-1] - recent_volumes[-2]) / recent_volumes[-2] >= growth_threshold
        
        # 综合判断：满足天数条件 且 整体趋势向上 且 最后一天增长
        result = days_condition and trend_condition and last_day_growth
        
        logger.debug(f"股票{stock_code}近3日成交量增长检查：{result} (增长天数：{growth_days}, 总增长率：{total_growth_rate:.2f}, 最后一天增长：{last_day_growth})")
        return result
    
    def _get_circulating_market_cap(self, stock_code: str) -> float:
        """获取股票的流通市值
        
        Args:
            stock_code: 股票代码
            
        Returns:
            流通市值
        """
        # 模拟流通市值数据，实际实现中应从真实数据源获取
        # 生成50亿到200亿之间的随机流通市值
        # 使用np.random.uniform生成浮点数，避免int32范围问题
        return np.random.uniform(5000000000, 20000000000)
    
    def get_candidate_stocks(self) -> pd.DataFrame:
        """获取个股候选池
        
        Returns:
            个股候选池DataFrame
        """
        return self.candidate_stocks
    
    def rank_candidate_stocks(self, candidate_stocks: pd.DataFrame = None) -> pd.DataFrame:
        """对候选个股进行排序
        
        Args:
            candidate_stocks: 候选个股DataFrame，如果为None则使用内部候选池
            
        Returns:
            排序后的候选个股DataFrame
        """
        logger.info("开始对候选个股进行排序")
        
        if candidate_stocks is None:
            candidate_stocks = self.candidate_stocks
        
        if candidate_stocks.empty:
            logger.error("候选个股为空，无法排序")
            return pd.DataFrame()
        
        # 计算个股综合评分
        candidate_stocks['stock_score'] = self._calculate_stock_score(candidate_stocks)
        
        # 按综合评分降序排序
        ranked_stocks = candidate_stocks.sort_values(by='stock_score', ascending=False)
        
        logger.info("候选个股排序完成")
        return ranked_stocks
    
    def _calculate_stock_score(self, candidate_stocks: pd.DataFrame) -> pd.Series:
        """计算个股综合评分
        
        Args:
            candidate_stocks: 候选个股DataFrame
            
        Returns:
            个股综合评分Series
        """
        # 动态权重分配，根据市场环境自动调整各指标权重
        # 1. 计算市场环境指标
        market_environment = {
            'avg_bid_change': candidate_stocks['price_bid_change'].mean(),
            'avg_volume_ratio': candidate_stocks['volume_bid_volume_ratio'].mean(),
            'avg_turnover_rate': candidate_stocks['volume_bid_turnover_rate'].mean(),
            'avg_order_amount': candidate_stocks['order_book_bid_order_amount'].mean(),
            'avg_order_ratio': candidate_stocks['order_book_bid_order_ratio'].mean(),
            'avg_net_flow': candidate_stocks['fund_flow_net_flow'].mean() if 'fund_flow_net_flow' in candidate_stocks.columns else 0
        }
        
        logger.info(f"个股筛选市场环境：{market_environment}")
        
        # 2. 基于市场环境的动态权重调整
        base_weights = {
            'price_bid_change': 0.30,  # 竞价涨跌幅
            'volume_bid_volume_ratio': 0.20,  # 竞价成交量比
            'volume_bid_turnover_rate': 0.15,  # 竞价换手率
            'order_book_bid_order_amount': 0.15,  # 封单金额
            'order_book_bid_order_ratio': 0.10,  # 封单比例
            'fund_flow_net_flow': 0.10  # 资金净流入
        }
        
        # 根据市场环境调整权重
        weights = base_weights.copy()
        
        # 市场强势时，增加价格相关指标权重
        if market_environment['avg_bid_change'] > 0.07:
            weights['price_bid_change'] += 0.05
            weights['volume_bid_volume_ratio'] -= 0.02
            weights['volume_bid_turnover_rate'] -= 0.01
            weights['order_book_bid_order_amount'] -= 0.01
            weights['order_book_bid_order_ratio'] -= 0.01
        # 市场弱势时，增加资金和成交量指标权重
        elif market_environment['avg_bid_change'] < 0.04:
            weights['fund_flow_net_flow'] += 0.05
            weights['volume_bid_volume_ratio'] += 0.03
            weights['price_bid_change'] -= 0.04
            weights['volume_bid_turnover_rate'] -= 0.02
            weights['order_book_bid_order_amount'] -= 0.02
        
        # 成交量放大时，增加成交量相关指标权重
        if market_environment['avg_volume_ratio'] > 3.0:
            weights['volume_bid_volume_ratio'] += 0.04
            weights['volume_bid_turnover_rate'] += 0.02
            weights['price_bid_change'] -= 0.03
            weights['order_book_bid_order_amount'] -= 0.03
        
        # 封单强度高时，增加封单相关指标权重
        if market_environment['avg_order_amount'] > 1e8 or market_environment['avg_order_ratio'] > 0.2:
            weights['order_book_bid_order_amount'] += 0.03
            weights['order_book_bid_order_ratio'] += 0.02
            weights['fund_flow_net_flow'] -= 0.02
            weights['volume_bid_turnover_rate'] -= 0.03
        
        # 确保权重为正值
        for key in weights:
            weights[key] = max(0.05, weights[key])
        
        # 归一化权重
        total_weight = sum(weights.values())
        for key in weights:
            weights[key] /= total_weight
        
        logger.info(f"个股综合评分动态权重：{weights}")
        
        # 3. 归一化处理各指标
        def normalize_series(s):
            """归一化序列，处理极值"""
            if len(s) == 0:
                return pd.Series([0] * len(s), index=s.index)
            
            # 使用分位数归一化，减少极值影响
            q1 = s.quantile(0.1)
            q9 = s.quantile(0.9)
            
            # 处理极端值
            s_clipped = s.clip(q1, q9)
            
            min_val = s_clipped.min()
            max_val = s_clipped.max()
            
            if max_val - min_val == 0:
                return pd.Series([0.5] * len(s), index=s.index)  # 所有值相同时，返回0.5
            
            return (s_clipped - min_val) / (max_val - min_val)
        
        # 4. 归一化各指标
        normalized_metrics = {}
        for metric, weight in weights.items():
            if metric in candidate_stocks.columns:
                normalized_metrics[metric] = normalize_series(candidate_stocks[metric])
            else:
                normalized_metrics[metric] = pd.Series([0] * len(candidate_stocks), index=candidate_stocks.index)
        
        # 5. 计算综合评分
        stock_score = pd.Series(0, index=candidate_stocks.index)
        for metric, weight in weights.items():
            stock_score += normalized_metrics[metric] * weight
        
        # 6. 添加额外的加分项
        # 6.1 热门板块加分
        if 'sector' in candidate_stocks.columns:
            # 假设前2个板块为热门板块
            top_sectors = candidate_stocks['sector'].value_counts().head(2).index.tolist()
            sector_bonus = candidate_stocks['sector'].apply(lambda x: 0.1 if x in top_sectors else 0)
            stock_score += sector_bonus
        
        # 6.2 连续涨停加分
        if 'consecutive_limit_up' in candidate_stocks.columns:
            limit_up_bonus = candidate_stocks['consecutive_limit_up'] * 0.05  # 每个连续涨停加5分
            stock_score += limit_up_bonus
        
        # 6.3 均线多头排列加分
        if 'bullish_ma_arrangement' in candidate_stocks.columns:
            ma_bonus = candidate_stocks['bullish_ma_arrangement'] * 0.05  # 均线多头加5分
            stock_score += ma_bonus
        
        # 6.4 MACD金叉加分
        if 'macd_golden_cross' in candidate_stocks.columns:
            macd_bonus = candidate_stocks['macd_golden_cross'] * 0.05  # MACD金叉加5分
            stock_score += macd_bonus
        
        # 归一化最终得分到0-100范围
        stock_score = (stock_score - stock_score.min()) / (stock_score.max() - stock_score.min()) * 100
        
        return stock_score
    
    def filter_top_stocks(self, ranked_stocks: pd.DataFrame = None, top_n: int = 10) -> pd.DataFrame:
        """筛选TOP个股
        
        Args:
            ranked_stocks: 排序后的候选个股DataFrame，如果为None则内部排序
            top_n: 要筛选的TOP个股数量
            
        Returns:
            TOP个股DataFrame
        """
        logger.info(f"筛选TOP {top_n}个股")
        
        if ranked_stocks is None:
            ranked_stocks = self.rank_candidate_stocks()
        
        if ranked_stocks.empty:
            return pd.DataFrame()
        
        # 选择TOP N个股
        top_stocks = ranked_stocks.head(top_n).copy()
        
        logger.info(f"筛选完成，共得到{len(top_stocks)}只TOP个股")
        return top_stocks
