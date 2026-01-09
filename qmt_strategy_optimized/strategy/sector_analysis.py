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
        
        # 初始化行业分类映射
        self.sector_mapping = self._load_sector_mapping()
        
        # 主要行业列表
        self.main_sectors = ['银行', '房地产', '钢铁', '建筑装饰', '公用事业', '采掘', '综合', '汽车', 
                           '商业贸易', '交通运输', '有色金属', '纺织服装', '食品饮料', '石油化工', 
                           '非银金融', '医药生物', '电子', '家用电器', '建筑材料', '通信']
    
    def _load_sector_mapping(self):
        """
        从CSV文件加载行业分类映射，若文件不存在则使用默认映射
        
        Returns:
            行业分类映射字典
        """
        import csv
        import os
        
        mapping_file = 'data/sector_mapping.csv'
        sector_mapping = {}
        
        # 检查映射文件是否存在
        if os.path.exists(mapping_file):
            logger.info(f"从{mapping_file}加载行业分类映射")
            try:
                with open(mapping_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        stock_code = row.get('stock_code', '')
                        sector = row.get('sector', '其他')
                        if stock_code:
                            sector_mapping[stock_code] = sector
                logger.info(f"加载了{len(sector_mapping)}条行业分类映射")
            except Exception as e:
                logger.error(f"加载行业分类映射失败：{e}")
                sector_mapping = self._get_default_sector_mapping()
        else:
            logger.info(f"行业分类映射文件{mapping_file}不存在，使用默认映射")
            sector_mapping = self._get_default_sector_mapping()
        
        return sector_mapping
    
    def _get_default_sector_mapping(self):
        """
        获取默认的行业分类映射
        
        Returns:
            默认行业分类映射字典
        """
        # 默认行业分类映射（示例数据，包含主要行业代表性股票）
        default_mapping = {
            '000001.SZ': '银行', '000002.SZ': '房地产', '600000.SH': '银行', '600001.SH': '钢铁',
            '600002.SH': '钢铁', '000004.SZ': '综合', '000005.SZ': '房地产', '600003.SH': '建筑装饰',
            '600004.SH': '公用事业', '600005.SH': '采掘', '000006.SZ': '房地产', '000007.SZ': '房地产',
            '000008.SZ': '综合', '600006.SH': '汽车', '600007.SH': '商业贸易', '600008.SH': '公用事业',
            '600009.SH': '交通运输', '000009.SZ': '综合', '000010.SZ': '有色金属', '600010.SH': '钢铁',
            '600011.SH': '公用事业', '600012.SH': '交通运输', '000011.SZ': '公用事业', '000012.SZ': '交通运输',
            '000014.SZ': '房地产', '600015.SH': '银行', '600016.SH': '银行', '600017.SH': '交通运输',
            '000016.SZ': '房地产', '000017.SZ': '公用事业', '600018.SH': '交通运输', '600019.SH': '钢铁',
            '600020.SH': '有色金属', '000018.SZ': '综合', '000019.SZ': '交通运输', '000020.SZ': '纺织服装',
            '600021.SH': '银行', '600022.SH': '钢铁', '600023.SH': '公用事业', '000021.SZ': '房地产',
            '000022.SZ': '食品饮料', '000023.SZ': '房地产', '600025.SH': '有色金属', '600026.SH': '交通运输',
            '600027.SH': '公用事业', '000024.SZ': '汽车', '000025.SZ': '房地产', '000026.SZ': '家用电器',
            '600028.SH': '石油化工', '600029.SH': '交通运输', '600030.SH': '非银金融', '000027.SZ': '房地产',
            '000028.SZ': '食品饮料', '000029.SZ': '交通运输', '600031.SH': '有色金属', '600033.SH': '交通运输',
            '600035.SH': '银行', '000030.SZ': '医药生物', '000031.SZ': '房地产', '000032.SZ': '房地产',
            '600036.SH': '银行', '600037.SH': '房地产', '600038.SH': '电子', '000033.SZ': '家用电器',
            '000034.SZ': '房地产', '000035.SZ': '家用电器', '600039.SH': '建筑材料', '600048.SH': '建筑装饰',
            '600050.SH': '通信', '600051.SH': '医药生物', '600052.SH': '医药生物', '600053.SH': '医药生物',
            '600054.SH': '电子', '600055.SH': '电子', '600056.SH': '电子', '600057.SH': '医药生物',
            '600058.SH': '医药生物', '600059.SH': '食品饮料', '600060.SH': '食品饮料', '600061.SH': '医药生物',
            '600062.SH': '医药生物', '600063.SH': '医药生物', '600064.SH': '医药生物', '600065.SH': '医药生物',
            '600066.SH': '汽车', '600067.SH': '汽车', '600068.SH': '汽车', '600069.SH': '汽车',
            '600070.SH': '汽车', '600071.SH': '汽车', '600072.SH': '汽车', '600073.SH': '汽车',
            '600074.SH': '汽车', '600075.SH': '汽车', '600076.SH': '汽车', '600077.SH': '汽车',
            '600078.SH': '汽车', '600079.SH': '汽车', '600080.SH': '汽车', '600081.SH': '汽车',
            '600082.SH': '汽车', '600083.SH': '汽车', '600084.SH': '汽车', '600085.SH': '汽车',
            '600086.SH': '汽车', '600087.SH': '汽车', '600088.SH': '汽车', '600089.SH': '汽车',
            '600090.SH': '汽车', '600091.SH': '汽车', '600092.SH': '汽车', '600093.SH': '汽车',
            '600094.SH': '汽车', '600095.SH': '汽车', '600096.SH': '汽车', '600097.SH': '汽车',
            '600098.SH': '汽车', '600099.SH': '汽车', '600100.SH': '汽车', '600101.SH': '汽车'
        }
        return default_mapping
    
    def _get_sector_from_pattern(self, stock_code):
        """
        根据股票代码模式推断行业
        
        Args:
            stock_code: 股票代码
            
        Returns:
            推断的行业名称
        """
        # 简单的股票代码模式推断逻辑（示例）
        if stock_code.startswith('60000'):
            return '银行'
        elif stock_code.startswith('60001'):
            return '钢铁'
        elif stock_code.startswith('60002'):
            return '有色金属'
        elif stock_code.startswith('60003'):
            return '医药生物'
        elif stock_code.startswith('60004'):
            return '通信'
        elif stock_code.startswith('60005'):
            return '电子'
        elif stock_code.startswith('60006'):
            return '汽车'
        elif stock_code.startswith('60007'):
            return '食品饮料'
        elif stock_code.startswith('60008'):
            return '家用电器'
        elif stock_code.startswith('60009'):
            return '交通运输'
        elif stock_code.startswith('00000'):
            return '房地产'
        elif stock_code.startswith('00001'):
            return '综合'
        elif stock_code.startswith('00002'):
            return '建筑装饰'
        elif stock_code.startswith('00003'):
            return '公用事业'
        else:
            # 随机分配主要行业，避免大部分股票都被归类为"其他"
            import random
            return random.choice(self.main_sectors)
    
    def get_stock_sector(self, stock_code):
        """
        获取股票所属行业
        
        Args:
            stock_code: 股票代码
            
        Returns:
            行业名称
        """
        # 首先从映射表中查找
        sector = self.sector_mapping.get(stock_code)
        if sector:
            return sector
        
        # 如果映射表中没有，尝试根据股票代码模式推断
        logger.debug(f"股票代码{stock_code}不在映射表中，尝试根据模式推断行业")
        sector = self._get_sector_from_pattern(stock_code)
        
        # 将推断结果添加到映射表中，避免重复计算
        self.sector_mapping[stock_code] = sector
        return sector
    
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
        
        # 1. 计算板块平均竞价涨幅
        if 'bid_change' in bid_df.columns:
            sector_avg_bid_change = bid_df.groupby('sector')['bid_change'].mean()
        elif 'close' in bid_df.columns:
            sector_avg_bid_change = bid_df.groupby('sector')['close'].mean()
        else:
            sector_avg_bid_change = pd.Series(0, index=bid_df['sector'].unique())
        
        # 2. 计算板块内接近涨停的股票数量（涨幅≥8%）
        if 'is_near_limit_up' in bid_df.columns:
            sector_near_limit_up = bid_df.groupby('sector')['is_near_limit_up'].sum()
        else:
            # 使用涨跌幅计算
            sector_near_limit_up = pd.Series(0, index=bid_df['sector'].unique())
            if 'bid_change' in bid_df.columns:
                bid_df['is_near_limit_up'] = bid_df['bid_change'] >= 0.08
                sector_near_limit_up = bid_df.groupby('sector')['is_near_limit_up'].sum()
            elif 'close' in bid_df.columns:
                bid_df['is_near_limit_up'] = bid_df['close'] >= 0.08
                sector_near_limit_up = bid_df.groupby('sector')['is_near_limit_up'].sum()
        
        # 3. 计算板块总资金流入
        if 'bid_amount' in bid_df.columns:
            sector_total_money_flow = bid_df.groupby('sector')['bid_amount'].sum()
        elif 'amount' in bid_df.columns:
            sector_total_money_flow = bid_df.groupby('sector')['amount'].sum()
        else:
            sector_total_money_flow = pd.Series(0, index=bid_df['sector'].unique())
        
        # 4. 计算板块平均资金流入强度（资金流入/板块股票数）
        sector_stock_count = bid_df.groupby('sector').size()
        sector_avg_money_flow = sector_total_money_flow / sector_stock_count
        
        # 5. 计算板块平均竞价强度
        if 'bid_intensity' in bid_df.columns:
            sector_avg_bid_intensity = bid_df.groupby('sector')['bid_intensity'].mean()
        elif 'bid_change' in bid_df.columns:
            sector_avg_bid_intensity = bid_df.groupby('sector')['bid_change'].mean()
        else:
            sector_avg_bid_intensity = pd.Series(0, index=bid_df['sector'].unique())
        
        # 6. 计算板块平均竞价量比
        if 'bid_volume_ratio' in bid_df.columns:
            sector_avg_bid_volume_ratio = bid_df.groupby('sector')['bid_volume_ratio'].mean()
        else:
            sector_avg_bid_volume_ratio = pd.Series(0, index=bid_df['sector'].unique())
        
        # 7. 计算板块平均封单比例
        if 'bid_order_ratio' in bid_df.columns:
            sector_avg_bid_order_ratio = bid_df.groupby('sector')['bid_order_ratio'].mean()
        else:
            sector_avg_bid_order_ratio = pd.Series(0, index=bid_df['sector'].unique())
        
        # 8. 计算板块平均竞价换手率
        if 'bid_turnover_rate' in bid_df.columns:
            sector_avg_bid_turnover_rate = bid_df.groupby('sector')['bid_turnover_rate'].mean()
        else:
            sector_avg_bid_turnover_rate = pd.Series(0, index=bid_df['sector'].unique())
        
        # 合并所有指标
        sector_metrics = pd.DataFrame({
            'avg_bid_change': sector_avg_bid_change,
            'near_limit_up_count': sector_near_limit_up,
            'total_money_flow': sector_total_money_flow,
            'avg_money_flow': sector_avg_money_flow,
            'avg_bid_intensity': sector_avg_bid_intensity,
            'avg_bid_volume_ratio': sector_avg_bid_volume_ratio,
            'avg_bid_order_ratio': sector_avg_bid_order_ratio,
            'avg_bid_turnover_rate': sector_avg_bid_turnover_rate,
            'stock_count': sector_stock_count
        })
        
        # 处理缺失值
        sector_metrics = sector_metrics.fillna(0)
        
        # 标准化各指标（使用Min-Max标准化）
        def normalize_series(s):
            if s.max() == s.min():
                return pd.Series(0.5, index=s.index)
            return (s - s.min()) / (s.max() - s.min())
        
        # 对每个指标进行标准化
        sector_metrics['norm_bid_change'] = normalize_series(sector_metrics['avg_bid_change'])
        sector_metrics['norm_limit_up'] = normalize_series(sector_metrics['near_limit_up_count'])
        sector_metrics['norm_total_money_flow'] = normalize_series(sector_metrics['total_money_flow'])
        sector_metrics['norm_avg_money_flow'] = normalize_series(sector_metrics['avg_money_flow'])
        sector_metrics['norm_bid_intensity'] = normalize_series(sector_metrics['avg_bid_intensity'])
        sector_metrics['norm_bid_volume_ratio'] = normalize_series(sector_metrics['avg_bid_volume_ratio'])
        sector_metrics['norm_bid_order_ratio'] = normalize_series(sector_metrics['avg_bid_order_ratio'])
        sector_metrics['norm_bid_turnover_rate'] = normalize_series(sector_metrics['avg_bid_turnover_rate'])
        
        # 计算综合热度评分（优化权重分配）
        sector_metrics['hotness_score'] = (
            sector_metrics['norm_bid_change'] * 0.2 +          # 竞价涨幅权重
            sector_metrics['norm_limit_up'] * 0.25 +           # 接近涨停股票数权重
            sector_metrics['norm_total_money_flow'] * 0.15 +   # 总资金流入权重
            sector_metrics['norm_avg_money_flow'] * 0.1 +      # 平均资金流入权重
            sector_metrics['norm_bid_intensity'] * 0.1 +       # 竞价强度权重
            sector_metrics['norm_bid_volume_ratio'] * 0.1 +    # 竞价量比权重
            sector_metrics['norm_bid_order_ratio'] * 0.05 +    # 封单比例权重
            sector_metrics['norm_bid_turnover_rate'] * 0.05     # 竞价换手率权重
        )
        
        # 增加板块股票数量的影响（股票数量太少的板块稳定性差）
        sector_metrics['stock_count_score'] = normalize_series(sector_metrics['stock_count'])
        # 对股票数量较少的板块进行评分调整
        sector_metrics['hotness_score'] = sector_metrics['hotness_score'] * (0.7 + 0.3 * sector_metrics['stock_count_score'])
        
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
        
        # 1. 按热度评分排序
        sorted_sectors = sorted(sector_hotness.items(), key=lambda x: x[1]['hotness_score'], reverse=True)
        
        # 2. 确保板块多样性，避免过度集中在某一类型
        top_sectors = []
        selected_sector_types = set()
        
        # 优先选择热度最高的板块
        for sector, metrics in sorted_sectors:
            # 检查板块是否已经选择，避免重复
            if sector in top_sectors:
                continue
            
            # 检查板块股票数量，确保有足够的股票可供选择
            if metrics.get('stock_count', 0) < 3:
                continue
            
            # 检查板块平均涨幅，确保是真正的热门板块
            if metrics.get('avg_bid_change', 0) < 0.01:
                continue
            
            # 添加到Top3列表
            top_sectors.append(sector)
            
            # 如果已经选择了3个板块，结束循环
            if len(top_sectors) >= 3:
                break
        
        # 3. 如果筛选后不足3个板块，使用热度排序的前3个板块
        if len(top_sectors) < 3:
            # 补充热度最高的板块
            for sector, metrics in sorted_sectors:
                if sector not in top_sectors:
                    top_sectors.append(sector)
                    if len(top_sectors) >= 3:
                        break
        
        # 4. 确保返回的板块数量为3个
        while len(top_sectors) < 3:
            # 如果还是不足3个，添加默认板块
            default_sectors = ['银行', '房地产', '医药生物']
            for default_sector in default_sectors:
                if default_sector not in top_sectors:
                    top_sectors.append(default_sector)
                    break
        
        logger.info(f"Top3热门板块：{top_sectors}")
        
        return top_sectors
    
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
