# 股票涨停预测自动交易系统

## 项目概述

本项目是一个基于机器学习的股票涨停预测自动交易系统，主要功能包括：

1. **模型封装**：使用ModelWrapper类封装完整的模型组件，包括模型权重、预处理逻辑和特征工程
2. **自动数据获取**：从QMT获取当天的股票竞价数据
3. **模型预测**：使用训练好的模型预测可能涨停的股票
4. **定时任务**：每天9:26分自动启动预测流程
5. **结果输出**：将预测结果以CSV和JSON格式保存

## 目录结构

```
auto_trading/
├── config/           # 配置文件目录
├── data/             # 数据存储目录
├── models/           # 模型存储目录
├── src/              # 源代码目录
│   ├── data_acquisition/   # 数据获取模块
│   │   └── qmt_bid_data.py  # QMT竞价数据获取服务
│   ├── model_inference/    # 模型推理模块
│   │   └── prediction_service.py  # 模型预测服务
│   ├── scheduler/          # 调度服务模块
│   │   └── scheduler_service.py  # 定时任务调度服务
│   └── utils/             # 工具模块
├── output/           # 输出结果目录
│   └── logs/         # 日志目录
├── test_prediction_flow.py  # 测试脚本
└── README.md         # 项目文档
```

## 核心组件

### 1. ModelWrapper类

ModelWrapper类用于封装完整的模型组件，包括：
- 模型权重
- 数据处理器
- 特征工程器
- 配置信息

```python
class ModelWrapper:
    def __init__(self, model, data_processor=None, feature_engineer=None, config=None):
        self.model = model
        self.data_processor = data_processor
        self.feature_engineer = feature_engineer
        self.config = config or {}
        self.version = "1.0"
        self.created_at = pd.Timestamp.now().isoformat()
    
    def predict(self, data):
        # 预处理数据
        # 提取特征
        # 进行预测
        # 返回预测结果
```

### 2. QMTBidDataFetcher类

用于从QMT获取当天的股票竞价数据：

- 支持从xtdata获取实时竞价数据
- 对原始数据进行预处理和格式化
- 支持保存数据到本地文件

### 3. PredictionService类

用于加载模型并进行预测：

- 自动加载最新的模型文件
- 支持使用ModelWrapper进行预测
- 支持过滤预测结果
- 支持保存预测结果到CSV和JSON文件

### 4. SchedulerService类

用于定时执行预测任务：

- 使用APScheduler实现定时调度
- 每天9:26分自动启动预测流程
- 支持立即执行测试

## 使用方法

### 1. 模型训练

首先需要训练模型并保存为ModelWrapper格式：

```bash
python train_models.py
```

训练完成后，模型文件将保存在`models/trained_models/`目录下。

### 2. 启动定时任务

启动调度服务，每天9:26分自动执行预测：

```bash
python src/scheduler/scheduler_service.py
```

### 3. 立即执行测试

使用`--test`参数立即执行一次预测任务：

```bash
python src/scheduler/scheduler_service.py --test
```

### 4. 运行测试脚本

运行完整的测试流程：

```bash
python test_prediction_flow.py
```

## 配置说明

### 模型配置

模型配置主要在`model_train.py`中定义，包括：
- 模型类型（logistic, random_forest, xgboost）
- 模型参数
- 训练数据路径

### 调度配置

调度配置在`scheduler_service.py`中定义：
- 执行时间：每天9:26分
- 任务ID：daily_prediction

### 输出配置

输出路径在`prediction_service.py`中定义：
- 输出目录：`output/`
- 文件名格式：`prediction_results_YYYYMMDD_HHMMSS.csv/json`

## 依赖说明

- Python 3.7+
- pandas
- numpy
- scikit-learn
- xgboost
- apscheduler
- xtquant（QMT）

## 安装依赖

```bash
pip install pandas numpy scikit-learn xgboost apscheduler
```

## 日志说明

日志文件保存在`output/logs/`目录下，记录了：
- 模型加载信息
- 数据获取过程
- 预测结果
- 错误信息

## 预测结果格式

### CSV格式

```csv
stock_code,date,prediction,probability,predict_time,model_version,model_path
000001.SZ,20260108,1,0.85,2026-01-08T09:26:00,1.0,model_wrapper_xgboost_20260108_164001.joblib
600000.SH,20260108,0,0.32,2026-01-08T09:26:00,1.0,model_wrapper_xgboost_20260108_164001.joblib
```

### JSON格式

```json
[
  {
    "stock_code": "000001.SZ",
    "date": "20260108",
    "prediction": 1,
    "probability": 0.85,
    "predict_time": "2026-01-08T09:26:00",
    "model_version": "1.0",
    "model_path": "model_wrapper_xgboost_20260108_164001.joblib"
  },
  {
    "stock_code": "600000.SH",
    "date": "20260108",
    "prediction": 0,
    "probability": 0.32,
    "predict_time": "2026-01-08T09:26:00",
    "model_version": "1.0",
    "model_path": "model_wrapper_xgboost_20260108_164001.joblib"
  }
]
```

## 注意事项

1. 确保QMT客户端已启动，xtdata能够正常连接
2. 模型文件需要放在`models/`目录下
3. 第一次运行时，需要先训练模型生成ModelWrapper格式的模型文件
4. 定时任务需要保持脚本持续运行
5. 预测结果的准确性取决于模型的训练质量和数据质量

## 扩展说明

### 添加新的模型类型

1. 在`model_train.py`的`_init_model`方法中添加新的模型初始化逻辑
2. 在`_optimize_hyperparameters`方法中添加新模型的超参数优化逻辑

### 扩展数据来源

1. 在`data_acquisition/`目录下创建新的数据获取类
2. 实现`get_today_bid_data`方法
3. 在`scheduler_service.py`中更新数据获取模块的导入

### 自定义输出格式

在`prediction_service.py`的`save_prediction_results`方法中添加新的输出格式支持

## 故障排查

1. **模型加载失败**：检查`models/`目录下是否有可用的ModelWrapper格式模型文件
2. **数据获取失败**：检查QMT客户端是否已启动，xtdata连接是否正常
3. **预测结果为空**：检查模型是否训练正常，输入数据格式是否正确
4. **调度任务未执行**：检查APScheduler是否正常运行，系统时间是否正确

## 版本更新

### v1.0

- 实现ModelWrapper类封装完整模型组件
- 实现QMT竞价数据获取服务
- 实现模型预测服务
- 实现定时任务调度服务
- 支持CSV和JSON格式结果输出
