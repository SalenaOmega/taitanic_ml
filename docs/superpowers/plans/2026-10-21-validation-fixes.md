# Titanic Validation Fixes Implementation Plan

**Goal:** 修正已确认的预处理泄漏、分类报告参数顺序、批次相关分箱错误，保留现有 Notebook 分析结构。
**Architecture:** 在 Notebook 内定义 sklearn 兼容的 TitanicFeatures；原始乘客数据先划分，再交由包含特征工程的 Pipeline。统计量属于每个拟合后的转换器实例。
**Tech Stack:** 现有 Python 3.12 环境、pandas、scikit-learn、unittest、nbclient。
**Spec:** 当前聊天中用户要求修正上一轮只读复核发现的问题。

## Constraints and review focus

不安装依赖，不修改原始数据、use.ipynb。初始修复不提交；后续用户已明确授权在运行验证后提交 Git。保留已有图表与模型对比逻辑。
检查验证折统计隔离、单行/混合批次一致性、超出训练票价范围、无年龄分组回退、分类报告真实人数。

## Task 1: Regression tests and targeted fixes

- [x] 在 tests/test_notebook.py 中通过 AST 加载 Notebook 定义；先运行测试复现分箱、统计隔离、分类报告错误。
- [x] 修改 titanic_prediction.ipynb：新增 TitanicFeatures.fit/transform；将特征工程纳入 make_pipe；原始数据划分；修正报告；全量拟合后预测原始测试数据。
- [x] 运行 unittest 全部测试，确认训练统计不受预测数据和其他实例影响。

## Task 2: Reproduction and documentation

- [x] 将 requirements 格式清单改名为 requirements.lock，同步 README 的安装命令与评估边界。
- [x] 从新内核顺序执行 Notebook，刷新输出和 submission.csv，核对 418 行、ID、标签。
- [x] 检查最终 Git diff，报告真实指标与剩余局限。

## Verification results

4 个回归测试通过；原内核从头执行 21 个代码单元，无 error 类型输出；submission.csv 的 418 行、唯一 ID、顺序、0/1 标签通过检查。CV 最优为梯度提升树，调优 CV 0.8273，留出 0.8045，全量参考 CV 0.8260。Windows/joblib 并行退出资源清理 KeyError 在复跑时重现；网格搜索已改为串行（n_jobs=1），不改变参数空间和评分规则，不修改全局工具环境。

用户授权提交前复跑：串行版本完整执行约 62 秒，21 个代码单元顺序完成，无 error 输出；未再出现 joblib 清理 KeyError。4 个回归测试及提交文件检查通过，分数与上一轮一致。Jupyter 启动仍有事件循环兼容与本地 TCP 传输提示，不影响本次执行。
