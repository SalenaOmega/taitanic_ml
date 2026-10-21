# 泰坦尼克号生存预测

用乘客信息预测泰坦尼克号沉船的生存情况，Kaggle Titanic 竞赛入门项目。

## 环境

使用 [uv](https://github.com/astral-sh/uv) 管理虚拟环境，Python 3.12。

```bash
# 创建虚拟环境并安装依赖
uv venv --python 3.12
uv pip install -r uv.lock
```

注册 Jupyter kernel：

```bash
.venv/Scripts/python.exe -m ipykernel install --user --name titanic-venv --display-name "Python (titanic uv)"
```

打开 `titanic_prediction.ipynb` 后，在右上角 kernel 选择器里选 **Python (titanic uv)**。

## 数据

```
data/titanic/
  train.csv            # 891 名乘客，含 Survived 标签
  test.csv             # 418 名乘客，无标签（待预测）
  gender_submission.csv  # Kaggle 基准提交示例
```

## 运行顺序

1. 按顺序执行 `titanic_prediction.ipynb` 的所有 cell
2. 执行完毕后 `submission.csv` 即为 Kaggle 提交文件

## 指标（来自最近一次执行）

| 评估口径 | 说明 |
|---------|------|
| 留出验证集（20%） | 训练/验证划分后的单次评估，较干净 |
| 调优后 CV | 网格搜索在 80% 训练集上的 5 折 CV，选参数用 |
| 全量 CV | 全部 891 样本的 5 折 CV，供参考，因调参后评估存在轻微乐观偏差 |

具体数字在 notebook 最后的汇总 cell 输出中可见。

## 数据泄露预防

- 年龄中位数、港口众数、票价分位数等统计量**只在训练集上计算**，再 transform 测试集
- `StandardScaler` 通过 `Pipeline` 集成，交叉验证每折独立 fit，不会把验证折分布泄露给模型

## 局限

- 特征工程较基础，Cabin 甲板字母未利用
- 未尝试 XGBoost / LightGBM 等更强实现
- 全量 CV 分数因调参后评估存在轻微乐观偏差，不能完全视为独立泛化估计
