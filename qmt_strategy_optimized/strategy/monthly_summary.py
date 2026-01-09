#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
月度汇总模块
负责生成月度交易汇总和股票收益率相关指标
"""

import pandas as pd
import numpy as np
from utils.logger import setup_logger
import os

logger = setup_logger()

class MonthlySummary:
    """月度汇总类"""
    
    def __init__(self):
        """初始化月度汇总"""
        logger.info("初始化月度汇总模块")
        self.trade_records = []
        self.monthly_reports = {}
    
    def add_trade_record(self, trade_record):
        """
        添加交易记录
        
        Args:
            trade_record: 交易记录
        """
        self.trade_records.append(trade_record)
    
    def add_trade_records(self, trade_records):
        """
        添加多条交易记录
        
        Args:
            trade_records: 交易记录列表
        """
        self.trade_records.extend(trade_records)
    
    def generate_monthly_summary(self, year=None, month=None):
        """
        生成月度汇总报告
        
        Args:
            year: 年份，默认为当前年份
            month: 月份，默认为当前月份
            
        Returns:
            月度汇总报告
        """
        logger.info(f"生成月度汇总报告，年份：{year}，月份：{month}")
        
        if not self.trade_records:
            logger.warning("没有交易记录，无法生成月度汇总")
            return None
        
        # 转换为DataFrame
        trade_df = pd.DataFrame(self.trade_records)
        
        # 转换日期格式
        if 'time' in trade_df.columns:
            trade_df['time'] = pd.to_datetime(trade_df['time'])
        elif 'buy_date' in trade_df.columns:
            trade_df['buy_date'] = pd.to_datetime(trade_df['buy_date'])
            if 'sell_date' in trade_df.columns:
                trade_df['sell_date'] = pd.to_datetime(trade_df['sell_date'])
        else:
            logger.warning("交易记录中没有日期信息，无法生成月度汇总")
            return None
        
        # 筛选指定年份和月份的数据
        if year and month:
            if 'time' in trade_df.columns:
                mask = (trade_df['time'].dt.year == year) & (trade_df['time'].dt.month == month)
            else:
                mask = (((trade_df['buy_date'].dt.year == year) & (trade_df['buy_date'].dt.month == month)) |
                       ((trade_df['sell_date'].dt.year == year) & (trade_df['sell_date'].dt.month == month)))
            monthly_trades = trade_df[mask]
            month_key = f"{year}-{month:02d}"
        else:
            # 生成所有月份的汇总
            return self.generate_all_monthly_summaries()
        
        if monthly_trades.empty:
            logger.warning(f"{year}年{month}月没有交易记录")
            return None
        
        # 计算月度指标
        summary = {
            'month': month_key,
            'total_trades': len(monthly_trades),
            'stock_count': monthly_trades['stock'].nunique() if 'stock' in monthly_trades.columns else monthly_trades['stock_code'].nunique(),
            'report_generated_time': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # 计算盈亏情况
        if 'pnl' in monthly_trades.columns:
            # 已平仓交易的盈亏
            closed_trades = monthly_trades[monthly_trades['pnl'].notnull()]
            if not closed_trades.empty:
                summary['total_pnl'] = closed_trades['pnl'].sum()
                summary['avg_pnl_per_trade'] = closed_trades['pnl'].mean()
                summary['profit_trades'] = len(closed_trades[closed_trades['pnl'] > 0])
                summary['loss_trades'] = len(closed_trades[closed_trades['pnl'] < 0])
                summary['break_even_trades'] = len(closed_trades[closed_trades['pnl'] == 0])
                summary['win_rate'] = summary['profit_trades'] / len(closed_trades) if len(closed_trades) > 0 else 0
                
                # 计算最大盈利和最大亏损
                summary['max_profit'] = closed_trades['pnl'].max() if not closed_trades[closed_trades['pnl'] > 0].empty else 0
                summary['max_loss'] = closed_trades['pnl'].min() if not closed_trades[closed_trades['pnl'] < 0].empty else 0
                
                # 计算盈亏比
                avg_profit = closed_trades[closed_trades['pnl'] > 0]['pnl'].mean() if summary['profit_trades'] > 0 else 0
                avg_loss = abs(closed_trades[closed_trades['pnl'] < 0]['pnl'].mean()) if summary['loss_trades'] > 0 else 1
                summary['profit_loss_ratio'] = avg_profit / avg_loss
                
                # 计算连续盈利和连续亏损次数
                closed_trades_sorted = closed_trades.sort_values('time' if 'time' in closed_trades.columns else 'sell_date')
                consecutive_wins = 0
                max_consecutive_wins = 0
                consecutive_losses = 0
                max_consecutive_losses = 0
                
                for _, trade in closed_trades_sorted.iterrows():
                    if trade['pnl'] > 0:
                        consecutive_wins += 1
                        max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)
                        consecutive_losses = 0
                    else:
                        consecutive_losses += 1
                        max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
                        consecutive_wins = 0
                
                summary['max_consecutive_wins'] = max_consecutive_wins
                summary['max_consecutive_losses'] = max_consecutive_losses
        
        # 计算收益率相关指标
        if 'return_rate' in monthly_trades.columns:
            # 已平仓交易的收益率
            closed_trades = monthly_trades[monthly_trades['return_rate'].notnull()]
            if not closed_trades.empty:
                summary['avg_return_rate'] = closed_trades['return_rate'].mean()
                summary['median_return_rate'] = closed_trades['return_rate'].median()
                summary['max_return_rate'] = closed_trades['return_rate'].max()
                summary['min_return_rate'] = closed_trades['return_rate'].min()
                summary['positive_return_rate'] = len(closed_trades[closed_trades['return_rate'] > 0]) / len(closed_trades)
                
                # 计算收益率分布
                returns = closed_trades['return_rate']
                summary['return_distribution'] = {
                    'count': len(returns),
                    'mean': float(returns.mean()),
                    'std': float(returns.std()),
                    'skew': float(returns.skew()),
                    'kurtosis': float(returns.kurtosis()),
                    'percentile_25': float(np.percentile(returns, 25)),
                    'percentile_50': float(np.percentile(returns, 50)),
                    'percentile_75': float(np.percentile(returns, 75)),
                    'percentile_95': float(np.percentile(returns, 95)),
                    'percentile_5': float(np.percentile(returns, 5))
                }
        
        # 按股票统计
        stock_col = 'stock' if 'stock' in monthly_trades.columns else 'stock_code'
        stock_stats = monthly_trades.groupby(stock_col).agg({
            'pnl': ['sum', 'mean', 'count'],
            'return_rate': ['mean', 'max', 'min'],
            'time': ['min', 'max']  # 首次交易时间和最后交易时间
        }).reset_index()
        
        stock_stats.columns = [stock_col, 'total_pnl', 'avg_pnl', 'trade_count', 'avg_return_rate', 'max_return_rate', 'min_return_rate', 'first_trade_time', 'last_trade_time']
        stock_stats = stock_stats.sort_values('total_pnl', ascending=False)
        
        # 添加股票胜率
        for idx, row in stock_stats.iterrows():
            # 获取该股票的所有交易
            stock_trades = monthly_trades[monthly_trades[stock_col] == row[stock_col]]
            # 计算胜率
            closed_trades = stock_trades[stock_trades['pnl'].notnull()]
            if not closed_trades.empty:
                win_rate = len(closed_trades[closed_trades['pnl'] > 0]) / len(closed_trades)
                stock_stats.at[idx, 'win_rate'] = win_rate
            else:
                stock_stats.at[idx, 'win_rate'] = 0
        
        # 股票收益率排行榜（按总盈亏）
        stock_profit_rank = stock_stats.sort_values('total_pnl', ascending=False).head(10)
        summary['stock_profit_rank'] = stock_profit_rank.to_dict('records')
        
        # 股票收益率排行榜（按平均收益率）
        stock_avg_return_rank = stock_stats[stock_stats['trade_count'] >= 2].sort_values('avg_return_rate', ascending=False).head(10)
        summary['stock_avg_return_rank'] = stock_avg_return_rank.to_dict('records')
        
        # 按板块统计（如果有行业信息）
        if 'sector' in monthly_trades.columns:
            sector_stats = monthly_trades.groupby('sector').agg({
                'pnl': ['sum', 'mean', 'count'],
                'return_rate': ['mean', 'max', 'min'],
                'stock': ['nunique']  # 每个板块涉及的股票数量
            }).reset_index()
            
            sector_stats.columns = ['sector', 'total_pnl', 'avg_pnl', 'trade_count', 'avg_return_rate', 'max_return_rate', 'min_return_rate', 'stock_count']
            sector_stats = sector_stats.sort_values('total_pnl', ascending=False)
            
            # 计算板块胜率
            for idx, row in sector_stats.iterrows():
                sector_trades = monthly_trades[monthly_trades['sector'] == row['sector']]
                closed_trades = sector_trades[sector_trades['pnl'].notnull()]
                if not closed_trades.empty:
                    win_rate = len(closed_trades[closed_trades['pnl'] > 0]) / len(closed_trades)
                    sector_stats.at[idx, 'win_rate'] = win_rate
                else:
                    sector_stats.at[idx, 'win_rate'] = 0
            
            # 板块收益排行榜
            sector_profit_rank = sector_stats.sort_values('total_pnl', ascending=False).head(10)
            summary['sector_profit_rank'] = sector_profit_rank.to_dict('records')
            
            # 板块平均收益率排行榜
            sector_avg_return_rank = sector_stats[sector_stats['trade_count'] >= 3].sort_values('avg_return_rate', ascending=False).head(10)
            summary['sector_avg_return_rank'] = sector_avg_return_rank.to_dict('records')
        
        # 按交易类型统计
        if 'action' in monthly_trades.columns:
            action_stats = monthly_trades['action'].value_counts().to_dict()
            summary['action_stats'] = action_stats
            
            # 按交易类型统计盈亏
            if 'pnl' in monthly_trades.columns:
                action_pnl = monthly_trades.groupby('action')['pnl'].sum().to_dict()
                action_avg_pnl = monthly_trades.groupby('action')['pnl'].mean().to_dict()
                summary['action_pnl'] = action_pnl
                summary['action_avg_pnl'] = action_avg_pnl
        
        # 按天统计交易量
        if 'time' in monthly_trades.columns:
            daily_trades = monthly_trades.groupby(monthly_trades['time'].dt.date).size().reset_index(name='trade_count')
            daily_trades.columns = ['date', 'trade_count']
            summary['daily_trade_distribution'] = daily_trades.to_dict('records')
        
        # 交易理由分析
        if 'reason' in monthly_trades.columns:
            # 统计主要交易理由
            buy_reasons = []
            sell_reasons = []
            for _, trade in monthly_trades.iterrows():
                if trade['action'] == 'buy':
                    buy_reasons.append(trade['reason'])
                else:
                    sell_reasons.append(trade['reason'])
            
            # 计算前5个主要理由
            from collections import Counter
            if buy_reasons:
                buy_reason_counter = Counter(buy_reasons)
                summary['top_buy_reasons'] = buy_reason_counter.most_common(5)
            
            if sell_reasons:
                sell_reason_counter = Counter(sell_reasons)
                summary['top_sell_reasons'] = sell_reason_counter.most_common(5)
        
        # 保存到月度报告字典
        self.monthly_reports[month_key] = summary
        
        logger.info(f"月度汇总报告生成完成：{summary}")
        return summary
    
    def generate_all_monthly_summaries(self):
        """
        生成所有月份的汇总报告
        
        Returns:
            所有月份的汇总报告字典
        """
        logger.info("生成所有月份的汇总报告")
        
        if not self.trade_records:
            logger.warning("没有交易记录，无法生成月度汇总")
            return {}
        
        # 转换为DataFrame
        trade_df = pd.DataFrame(self.trade_records)
        
        # 转换日期格式
        if 'time' in trade_df.columns:
            trade_df['time'] = pd.to_datetime(trade_df['time'])
            trade_df['year_month'] = trade_df['time'].dt.strftime('%Y-%m')
        elif 'buy_date' in trade_df.columns:
            trade_df['buy_date'] = pd.to_datetime(trade_df['buy_date'])
            trade_df['year_month'] = trade_df['buy_date'].dt.strftime('%Y-%m')
        else:
            logger.warning("交易记录中没有日期信息，无法生成月度汇总")
            return {}
        
        # 按月份分组
        grouped = trade_df.groupby('year_month')
        
        all_summaries = {}
        
        for month, group in grouped:
            logger.info(f"生成{month}月份的汇总报告")
            
            year = int(month.split('-')[0])
            month_num = int(month.split('-')[1])
            
            # 生成月度汇总
            summary = self.generate_monthly_summary(year, month_num)
            if summary:
                all_summaries[month] = summary
        
        logger.info(f"所有月份的汇总报告生成完成，共{len(all_summaries)}个月份")
        return all_summaries
    
    def save_monthly_summary(self, month_key, output_dir="reports/monthly"):
        """
        保存月度汇总报告到文件
        
        Args:
            month_key: 月份键，格式为YYYY-MM
            output_dir: 输出目录
        """
        logger.info(f"保存月度汇总报告到文件，月份：{month_key}，输出目录：{output_dir}")
        
        if month_key not in self.monthly_reports:
            logger.warning(f"没有找到{month_key}月份的汇总报告")
            return False
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存为JSON文件
        import json
        json_file = os.path.join(output_dir, f"monthly_summary_{month_key}.json")
        
        try:
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(self.monthly_reports[month_key], f, indent=4, ensure_ascii=False, default=str)
            logger.info(f"月度汇总报告保存成功：{json_file}")
        except Exception as e:
            logger.error(f"保存月度汇总报告失败：{e}")
            return False
        
        # 保存为CSV文件（股票统计部分）
        stock_stats = self.monthly_reports[month_key].get('stock_stats', [])
        if stock_stats:
            stock_df = pd.DataFrame(stock_stats)
            csv_file = os.path.join(output_dir, f"monthly_stock_stats_{month_key}.csv")
            try:
                stock_df.to_csv(csv_file, index=False, encoding='utf-8-sig')
                logger.info(f"月度股票统计保存成功：{csv_file}")
            except Exception as e:
                logger.error(f"保存月度股票统计失败：{e}")
        
        return True
    
    def save_all_monthly_summaries(self, output_dir="reports/monthly"):
        """
        保存所有月度汇总报告到文件
        
        Args:
            output_dir: 输出目录
        """
        logger.info(f"保存所有月度汇总报告到文件，输出目录：{output_dir}")
        
        success_count = 0
        total_count = len(self.monthly_reports)
        
        for month_key in self.monthly_reports:
            if self.save_monthly_summary(month_key, output_dir):
                success_count += 1
        
        logger.info(f"所有月度汇总报告保存完成，成功：{success_count}/{total_count}")
        return success_count == total_count
    
    def get_monthly_summary(self, month_key):
        """
        获取指定月份的汇总报告
        
        Args:
            month_key: 月份键，格式为YYYY-MM
            
        Returns:
            月度汇总报告
        """
        return self.monthly_reports.get(month_key)
    
    def get_all_monthly_summaries(self):
        """
        获取所有月份的汇总报告
        
        Returns:
            所有月份的汇总报告字典
        """
        return self.monthly_reports
    
    def print_monthly_summary(self, month_key):
        """
        打印月度汇总报告
        
        Args:
            month_key: 月份键，格式为YYYY-MM
        """
        summary = self.get_monthly_summary(month_key)
        if not summary:
            logger.warning(f"没有找到{month_key}月份的汇总报告")
            return
        
        print(f"\n{'='*60}")
        print(f"{month_key} 月度交易汇总报告")
        print(f"{'='*60}")
        print(f"总交易次数：{summary['total_trades']}")
        print(f"涉及股票数量：{summary['stock_count']}")
        
        if 'total_pnl' in summary:
            print(f"总盈亏：{summary['total_pnl']:.2f} 元")
            print(f"平均每笔盈亏：{summary['avg_pnl_per_trade']:.2f} 元")
            print(f"盈利交易次数：{summary['profit_trades']}")
            print(f"亏损交易次数：{summary['loss_trades']}")
            print(f"平手交易次数：{summary['break_even_trades']}")
            print(f"胜率：{summary['win_rate']:.2%}")
            print(f"最大盈利：{summary['max_profit']:.2f} 元")
            print(f"最大亏损：{summary['max_loss']:.2f} 元")
            print(f"盈亏比：{summary['profit_loss_ratio']:.2f}")
        
        if 'avg_return_rate' in summary:
            print(f"平均收益率：{summary['avg_return_rate']:.2%}")
            print(f"中位数收益率：{summary['median_return_rate']:.2%}")
            print(f"最大收益率：{summary['max_return_rate']:.2%}")
            print(f"最小收益率：{summary['min_return_rate']:.2%}")
            print(f"正收益率比例：{summary['positive_return_rate']:.2%}")
        
        if 'action_stats' in summary:
            print(f"\n交易类型统计：")
            for action, count in summary['action_stats'].items():
                print(f"  {action}：{count} 次")
        
        if 'stock_stats' in summary and summary['stock_stats']:
            print(f"\n股票收益排名（前5名）：")
            for i, stock in enumerate(summary['stock_stats'][:5]):
                stock_code = stock['stock'] if 'stock' in stock else stock['stock_code']
                print(f"  {i+1}. {stock_code}：总盈亏 {stock['total_pnl']:.2f} 元，平均收益率 {stock['avg_return_rate']:.2%}")
        
        if 'sector_stats' in summary and summary['sector_stats']:
            print(f"\n行业收益排名（前3名）：")
            for i, sector in enumerate(summary['sector_stats'][:3]):
                print(f"  {i+1}. {sector['sector']}：总盈亏 {sector['total_pnl']:.2f} 元，平均收益率 {sector['avg_return_rate']:.2%}")
        
        print(f"{'='*60}\n")
