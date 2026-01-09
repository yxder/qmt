#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
回测结果可视化脚本
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def load_backtest_results():
    """加载回测结果"""
    # 查找最新的回测结果文件
    reports_dir = 'reports'
    if not os.path.exists(reports_dir):
        print(f"报告目录 {reports_dir} 不存在")
        return None
    
    # 获取所有回测结果文件
    result_files = [f for f in os.listdir(reports_dir) if f.startswith('backtest_results_') and f.endswith('.json')]
    if not result_files:
        print("没有找到回测结果文件")
        return None
    
    # 按时间排序，取最新的
    result_files.sort(reverse=True)
    latest_file = os.path.join(reports_dir, result_files[0])
    
    print(f"加载最新回测结果: {latest_file}")
    with open(latest_file, 'r', encoding='utf-8') as f:
        backtest_results = json.load(f)
    
    # 加载绩效报告
    perf_file = latest_file.replace('backtest_results', 'performance_report')
    if os.path.exists(perf_file):
        with open(perf_file, 'r', encoding='utf-8') as f:
            perf_report = json.load(f)
    else:
        perf_report = None
    
    # 加载资金曲线
    equity_file = latest_file.replace('backtest_results', 'equity_curve').replace('.json', '.csv')
    if os.path.exists(equity_file):
        equity_curve = pd.read_csv(equity_file, header=None, names=['equity'])
    else:
        equity_curve = None
    
    return backtest_results, perf_report, equity_curve

def plot_equity_curve(equity_curve):
    """绘制资金曲线"""
    if equity_curve is None:
        print("没有资金曲线数据")
        return
    
    plt.figure(figsize=(12, 6))
    plt.plot(equity_curve['equity'], label='资金曲线')
    plt.title('回测资金曲线', fontsize=14)
    plt.xlabel('交易天数', fontsize=12)
    plt.ylabel('资金金额（元）', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('reports/equity_curve_plot.png', dpi=300)
    plt.close()
    print("资金曲线图表已保存: reports/equity_curve_plot.png")

def plot_performance_metrics(backtest_results):
    """绘制绩效指标雷达图"""
    # 提取关键指标
    metrics = {
        '年化收益率': backtest_results.get('annual_return', 0),
        '夏普比率': backtest_results.get('sharpe_ratio', 0) if backtest_results.get('sharpe_ratio') != -np.inf else 0,
        '最大回撤': backtest_results.get('max_drawdown', 0),
        '胜率': backtest_results.get('win_rate', 0),
        '交易次数': backtest_results.get('trade_count', 0)
    }
    
    # 将指标转换为适合雷达图的格式
    categories = list(metrics.keys())
    values = list(metrics.values())
    
    # 调整指标范围，使其适合雷达图展示
    # 对于百分比指标，转换为小数
    adjusted_values = []
    for cat, val in zip(categories, values):
        if cat in ['年化收益率', '最大回撤', '胜率']:
            adjusted_values.append(val * 100)  # 转换为百分比
        else:
            adjusted_values.append(val)
    
    # 雷达图绘制
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    values = adjusted_values + adjusted_values[:1]  # 闭合
    angles = angles + angles[:1]
    
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)
    ax.plot(angles, values, 'o-', linewidth=2, label='绩效指标')
    ax.fill(angles, values, alpha=0.25)
    ax.set_thetagrids(np.degrees(angles[:-1]), categories, fontsize=12)
    ax.set_title('回测绩效雷达图', fontsize=14, pad=20)
    ax.grid(True)
    ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.tight_layout()
    plt.savefig('reports/performance_radar.png', dpi=300)
    plt.close()
    print("绩效雷达图已保存: reports/performance_radar.png")

def plot_model_performance():
    """绘制模型表现"""
    # 查找最新的模型性能指标文件
    models_dir = 'models/trained_models'
    if not os.path.exists(models_dir):
        print(f"模型目录 {models_dir} 不存在")
        return
    
    # 获取所有模型性能文件
    metric_files = [f for f in os.listdir(models_dir) if f.startswith('metrics_') and f.endswith('.json')]
    if not metric_files:
        print("没有找到模型性能文件")
        return
    
    # 按时间排序，取最新的
    metric_files.sort(reverse=True)
    latest_file = os.path.join(models_dir, metric_files[0])
    
    print(f"加载最新模型性能: {latest_file}")
    with open(latest_file, 'r', encoding='utf-8') as f:
        model_metrics = json.load(f)
    
    # 绘制模型性能柱状图
    metrics = list(model_metrics.keys())
    values = list(model_metrics.values())
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(metrics, values)
    plt.title('模型性能指标', fontsize=14)
    plt.xlabel('指标名称', fontsize=12)
    plt.ylabel('指标值', fontsize=12)
    plt.ylim(0, 1.1)  # 设置y轴范围
    
    # 在柱状图上显示数值
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.4f}', ha='center', va='bottom')
    
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('reports/model_performance.png', dpi=300)
    plt.close()
    print("模型性能图表已保存: reports/model_performance.png")

def create_summary_table(backtest_results):
    """创建回测结果汇总表格"""
    if backtest_results is None:
        return
    
    # 转换为DataFrame
    df = pd.DataFrame([backtest_results])
    
    # 格式化百分比列
    percent_cols = ['total_return', 'annual_return', 'max_drawdown', 'win_rate']
    for col in percent_cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: f"{x*100:.2f}%")
    
    # 绘制表格
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('tight')
    ax.axis('off')
    
    # 创建表格
    table = ax.table(cellText=df.values, colLabels=df.columns, 
                    cellLoc='center', loc='center')
    
    # 美化表格
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)  # 调整表格大小
    
    plt.title('回测结果汇总表', fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig('reports/backtest_summary_table.png', dpi=300)
    plt.close()
    print("回测结果汇总表已保存: reports/backtest_summary_table.png")

def main():
    """主函数"""
    print("=== 回测报告图形化展示 ===")
    
    # 加载回测结果
    backtest_results, perf_report, equity_curve = load_backtest_results()
    if backtest_results is None:
        return
    
    # 创建可视化图表
    print("\n生成可视化图表...")
    
    # 1. 资金曲线
    plot_equity_curve(equity_curve)
    
    # 2. 绩效指标雷达图
    plot_performance_metrics(backtest_results)
    
    # 3. 模型性能图表
    plot_model_performance()
    
    # 4. 回测结果汇总表
    create_summary_table(backtest_results)
    
    print("\n=== 可视化完成 ===")
    print("所有图表已保存到 reports 目录")
    print("\n生成的图表:")
    print("- reports/equity_curve_plot.png   # 资金曲线")
    print("- reports/performance_radar.png   # 绩效雷达图")
    print("- reports/model_performance.png   # 模型性能")
    print("- reports/backtest_summary_table.png  # 回测结果汇总表")

if __name__ == "__main__":
    main()
