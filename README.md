# 泰坦尼克号生存预测

用乘客信息预测泰坦尼克号沉船的生存情况，Kaggle Titanic 竞赛入门项目。

## 环境

使用 [uv](https://github.com/astral-sh/uv) 管理虚拟环境，Python 3.12。`requirements.lock` 为固定版本的 requirements 格式依赖清单。

```bash
# 创建虚拟环境并安装依赖
uv venv --python 3.12
uv pip install -r requirements.lock
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

网格搜索使用串行执行（`n_jobs=1`），避免本机 Windows/joblib 多进程退出时的资源清理错误；搜索范围和评分规则不变。

## 指标（来自最近一次执行）

| 评估口径 | 说明 |
|---------|------|
| 留出验证集（20%） | 原始数据先划分；不参与自动选择，但已在项目迭代中多次查看 |
| 调优后 CV | 网格搜索在 80% 训练集上的 5 折 CV，选参数用 |
| 全量 CV | 全部 891 样本的 5 折 CV，供参考，模型与参数选择可能造成乐观偏差 |

具体数字在 notebook 最后的汇总 cell 输出中可见。

## 数据泄露预防

- 原始数据先划分；`TitanicFeatures`、`StandardScaler` 和模型组成完整 Pipeline。交叉验证、网格搜索每折只用当折训练部分拟合所有统计量。
- 年龄缺失分组使用当次训练乘客的整体年龄中位数回退；预测不会重新计算统计量。
- 票价分箱保存训练四分位点，两端固定为无穷边界，单行和整批预测使用相同规则。

## 局限

- 特征工程较基础，Cabin 甲板字母未利用
- 未尝试 XGBoost / LightGBM 等更强实现
- 全量 CV 的模型、参数已通过同一数据集选出，不能视为独立泛化估计。留出集也已被多次查看，不能等同于未使用过的最终测试集。

## 回归检查

```bash
.venv/Scripts/python.exe -B -m unittest discover -s tests -v
```

测试直接加载 Notebook 中的定义，检查每折统计量隔离、单行/批量一致性、转换器实例隔离、极端票价、缺失年龄回退和分类报告真实人数。
