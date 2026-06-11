# 视频审核实验指标

本文记录当前最后一版视频审核模型的评测集合构造方式和主要指标。

## 1. 模型与阈值

```text
checkpoint:
  outputs/checkpoints/multi_modal_cross_attention_model.pth

默认阈值:
  threshold = 0.5

当前推荐 API 阈值:
  threshold = 0.32
```

`score` 表示视频属于异常/PCL 类别的概率：

```text
is_normal = score < threshold
```

## 2. 评测集合构造

完整标注文件为：

```text
data/annotations/Annotation_Subset.csv
```

完整标注规模：

```text
总样本数：715
train：571
test：144
label 0：518
label 1：197
```

但当前最后一版实验只在已对齐四模态特征的视频上评测。现有可用四模态特征集中在 `disabled` 子集：

```text
总样本数：116
train：85   label 0 = 66, label 1 = 19
test：31    label 0 = 26, label 1 = 5
```

因此，下面指标反映的是当前 `disabled` 子集上的工程验证效果，不代表完整六类人群数据的最终泛化结论。

## 3. 最终指标

| 集合 | 阈值 | 样本数 | Accuracy | Macro F1 | PCL Precision | PCL Recall | PCL F1 | AUC | Confusion Matrix |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| train | 0.5 | 85 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | `[[66,0],[0,19]]` |
| test | 0.5 | 31 | 0.8387 | 0.5974 | 0.5000 | 0.2000 | 0.2857 | 0.8846 | `[[25,1],[4,1]]` |
| test | 0.32 | 31 | 0.8710 | 0.7130 | 0.6667 | 0.4000 | 0.5000 | 0.8846 | `[[25,1],[3,2]]` |
| test | 0.008 | 31 | 0.8387 | 0.7567 | 0.5000 | 0.8000 | 0.6154 | 0.8846 | `[[22,4],[1,4]]` |

## 4. 结果解读

`threshold=0.5` 更保守，误杀较少，但异常召回偏低。

`threshold=0.32` 在当前 test 集上取得最高准确率，兼顾了准确率和异常召回，因此作为当前 API 推荐阈值。

`threshold=0.008` 明显提高异常召回，但会带来更多正常视频误判，适合更偏安全拦截的策略。

训练集指标为 1.0，且测试集只有 31 条，因此当前结果应理解为 demo/API 接入阶段的可用性验证，而不是完整泛化评估。后续如果要做更严格结论，需要补齐其他人群子集的四模态特征并重新评测。

## 5. 指标来源

主要指标来自：

```text
outputs/metrics/fusion_eval_train_current.json
outputs/metrics/fusion_eval_test_current.json
outputs/metrics/fusion_eval_smoke.json
```

其中 `fusion_eval_smoke.json` 只用于流程冒烟测试，不作为性能指标。

