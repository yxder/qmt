#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
行业分析模块
负责热门板块识别和行业轮动策略
"""

import pandas as pd
import numpy as np
from utils.logger import setup_logger

logger = setup_logger()

class SectorAnalysis:
    """行业分析类，用于识别热门板块"""
    
    def __init__(self):
        """初始化行业分析器"""
        logger.info("初始化行业分析器")
        
        # 行业分类映射（示例数据，实际应用中可从外部获取）
        self.sector_mapping = {
            '000001.SZ': '银行',
            '000002.SZ': '房地产',
            '600000.SH': '银行',
            '600001.SH': '钢铁',
            '600002.SH': '钢铁',
            '000004.SZ': '综合',
            '000005.SZ': '房地产',
            '600003.SH': '建筑装饰',
            '600004.SH': '公用事业',
            '600005.SH': '采掘',
            '000006.SZ': '房地产',
            '000007.SZ': '房地产',
            '000008.SZ': '综合',
            '600006.SH': '汽车',
            '600007.SH': '商业贸易',
            '600008.SH': '公用事业',
            '600009.SH': '交通运输',
            '000009.SZ': '综合',
            '000010.SZ': '有色金属',
            '600010.SH': '钢铁',
            '600011.SH': '公用事业',
            '600012.SH': '交通运输',
            '000011.SZ': '公用事业',
            '000012.SZ': '交通运输',
            '000014.SZ': '房地产',
            '600015.SH': '银行',
            '600016.SH': '银行',
            '600017.SH': '交通运输',
            '000016.SZ': '房地产',
            '000017.SZ': '公用事业',
            '600018.SH': '交通运输',
            '600019.SH': '钢铁',
            '600020.SH': '有色金属',
            '000018.SZ': '综合',
            '000019.SZ': '交通运输',
            '000020.SZ': '纺织服装',
            '600021.SH': '银行',
            '600022.SH': '钢铁',
            '600023.SH': '公用事业',
            '000021.SZ': '房地产',
            '000022.SZ': '食品饮料',
            '000023.SZ': '房地产',
            '600025.SH': '有色金属',
            '600026.SH': '交通运输',
            '600027.SH': '公用事业',
            '000024.SZ': '汽车',
            '000025.SZ': '房地产',
            '000026.SZ': '家用电器',
            '600028.SH': '石油化工',
            '600029.SH': '交通运输',
            '600030.SH': '非银金融',
            '000027.SZ': '房地产',
            '000028.SZ': '食品饮料',
            '000029.SZ': '交通运输',
            '600031.SH': '有色金属',
            '600033.SH': '交通运输',
            '600035.SH': '银行',
            '000030.SZ': '医药生物',
            '000031.SZ': '房地产',
            '000032.SZ': '房地产',
            '600036.SH': '银行',
            '600037.SH': '房地产',
            '600038.SH': '电子',
            '000033.SZ': '家用电器',
            '000034.SZ': '房地产',
            '000035.SZ': '家用电器',
            '600039.SH': '建筑材料',
            '600048.SH': '建筑装饰',
            '600050.SH': '通信',
        }
    
    def get_stock_sector(self, stock_code):
        """
        获取股票所属行业
        
        Args:
            stock_code: 股票代码
            
        Returns:
            行业名称
        """
        return self.sector_mapping.get(stock_code, '其他')
    
    def calculate_sector_hotness(self, bid_data, market_data):
        """
        计算板块热度评分
        
        Args:
            bid_data: 竞价数据，包含股票代码和竞价相关指标
            market_data: 市场数据，包含股票的其他信息
            
        Returns:
            板块热度评分字典
        """
        logger.info("计算板块热度评分")
        
        # 合并竞价数据和市场数据
        if isinstance(bid_data, dict):
            # 如果是字典格式，转换为DataFrame
            df_list = []
            for stock_code, stock_data in bid_data.items():
                if isinstance(stock_data, pd.DataFrame) and not stock_data.empty:
                    df = stock_data.copy()
                    df['stock_code'] = stock_code
                    df_list.append(df)
            
            if not df_list:
                logger.warning("竞价数据为空，无法计算板块热度")
                return {}
            
            bid_df = pd.concat(df_list, ignore_index=True)
        else:
            bid_df = bid_data.copy()
        
        # 添加行业信息
        bid_df['sector'] = bid_df['stock_code'].apply(self.get_stock_sector)
        
        # 1. 计算板块平均涨幅
        sector_avg_change = bid_df.groupby('sector')['close'].mean()
        
        # 2. 计算板块涨停家数（假设涨幅≥9.8%为涨停）
        bid_df['is_limit_up'] = bid_df['close'] >= 0.098
        sector_limit_up = bid_df.groupby('sector')['is_limit_up'].sum()
        
        # 3. 计算板块资金流入（假设amount为资金流入）
        sector_money_flow = bid_df.groupby('sector')['amount'].sum()
        
        # 4. 计算板块平均竞价强度
        if 'bid_intensity' in bid_df.columns:
            sector_bid_intensity = bid_df.groupby('sector')['bid_intensity'].mean()
        else:
            # 如果没有竞价强度，使用涨幅代替
            sector_bid_intensity = sector_avg_change
        
        # 5. 计算板块平均成交量
        sector_avg_volume = bid_df.groupby('sector')['volume'].mean()
        
        # 合并所有指标
        sector_metrics = pd.DataFrame({
            'avg_change': sector_avg_change,
            'limit_up_count': sector_limit_up,
            'money_flow': sector_money_flow,
            'bid_intensity': sector_bid_intensity,
            'avg_volume': sector_avg_volume
        })
        
        # 标准化各指标
        sector_metrics['norm_change'] = (sector_metrics['avg_change'] - sector_metrics['avg_change'].min()) / \
                                       (sector_metrics['avg_change'].max() - sector_metrics['avg_change'].min())
        
        sector_metrics['norm_limit_up'] = (sector_metrics['limit_up_count'] - sector_metrics['limit_up_count'].min()) / \
                                          (sector_metrics['limit_up_count'].max() - sector_metrics['limit_up_count'].min())
        
        sector_metrics['norm_money_flow'] = (sector_metrics['money_flow'] - sector_metrics['money_flow'].min()) / \
                                            (sector_metrics['money_flow'].max() - sector_metrics['money_flow'].min())
        
        sector_metrics['norm_bid_intensity'] = (sector_metrics['bid_intensity'] - sector_metrics['bid_intensity'].min()) / \
                                               (sector_metrics['bid_intensity'].max() - sector_metrics['bid_intensity'].min())
        
        sector_metrics['norm_volume'] = (sector_metrics['avg_volume'] - sector_metrics['avg_volume'].min()) / \
                                        (sector_metrics['avg_volume'].max() - sector_metrics['avg_volume'].min())
        
        # 计算综合热度评分（权重可调整）
        sector_metrics['hotness_score'] = (
            sector_metrics['norm_change'] * 0.25 +
            sector_metrics['norm_limit_up'] * 0.3 +
            sector_metrics['norm_money_flow'] * 0.2 +
            sector_metrics['norm_bid_intensity'] * 0.15 +
            sector_metrics['norm_volume'] * 0.1
        )
        
        # 按热度评分降序排序
        sector_metrics = sector_metrics.sort_values('hotness_score', ascending=False)
        
        logger.info(f"板块热度计算完成，共{len(sector_metrics)}个板块")
        logger.info(f"热度排名前5的板块：{sector_metrics.head(5).index.tolist()}")
        
        return sector_metrics.to_dict('index')
    
    def get_top3_sectors(self, bid_data, market_data):
        """
        获取Top3热门板块
        
        Args:
            bid_data: 竞价数据
            market_data: 市场数据
            
        Returns:
            Top3热门板块列表
        """
        logger.info("获取Top3热门板块")
        
        sector_hotness = self.calculate_sector_hotness(bid_data, market_data)
        
        if not sector_hotness:
            logger.warning("无法计算板块热度，返回默认板块")
            return ['银行', '房地产', '医药生物']
        
        # 获取热度最高的3个板块
        sorted_sectors = sorted(sector_hotness.items(), key=lambda x: x[1]['hotness_score'], reverse=True)
        top3_sectors = [sector[0] for sector in sorted_sectors[:3]]
        
        logger.info(f"Top3热门板块：{top3_sectors}")
        
        return top3_sectors
    
    def filter_stocks_by_sectors(self, stocks, sectors):
        """
        根据板块过滤股票
        
        Args:
            stocks: 股票列表或DataFrame
            sectors: 目标板块列表
            
        Returns:
            过滤后的股票列表或DataFrame
        """
        logger.info(f"根据板块{sectors}过滤股票")
        
        if isinstance(stocks, list):
            # 列表格式
            filtered_stocks = [stock for stock in stocks if self.get_stock_sector(stock) in sectors]
        elif isinstance(stocks, pd.DataFrame):
            # DataFrame格式
            stocks['sector'] = stocks['stock_code'].apply(self.get_stock_sector)
            filtered_stocks = stocks[stocks['sector'].isin(sectors)]
        else:
            logger.warning(f"股票数据格式不支持：{type(stocks)}")
            filtered_stocks = stocks
        
        logger.info(f"过滤前股票数量：{len(stocks)}, 过滤后：{len(filtered_stocks)}")
        
        return filtered_stocks
    
    def update_sector_mapping(self, new_mapping):
        """
        更新行业分类映射
        
        Args:
            new_mapping: 新的行业分类映射字典
        """
        logger.info("更新行业分类映射")
        self.sector_mapping.update(new_mapping)
        logger.info(f"行业映射更新完成，共{len(self.sector_mapping)}条记录")
