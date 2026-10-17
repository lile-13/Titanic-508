# 派生特征说明

由 `python -m scripts.export_features` 生成，与 `notebooks/*.ipynb` 中的口径一致。

当前导出策略：`rule_group`。人数计数使用公开 train+test 无标签特征池。

**用途：最终全量训练与测试预测，或检查数据。不要直接在此表上做正式 CV。**
交叉验证请从原始数据构建未填补特征，按 folds.csv 的 PassengerId 映射取行，再调用 `src.preprocess.prepare_fold_matrices`，每折独立拟合填补器。

## 清洗后表里的派生列

| 列名 | 含义 |
|---|---|
| `GroupId` | 同行组号（PassengerId 下划线前的部分） |
| `GroupNum` | 组内序号（1~8） |
| `GroupSize` | 同组人数：单人为 1。与目标强相关（单人 45% vs 四人组 64%） |
| `IsAlone` | 是否独自出行（GroupSize == 1） |
| `IsFirstInGroup` | 是否为组内第一个（GroupNum == 1） |
| `IsLastInGroup` | 是否为组内最后一个 |
| `GroupPos` | 组内相对位置 0~1，比绝对序号稳定 |
| `Deck` | Cabin 拆出的舱位牌面 A~G、T（T 仅 5 人） |
| `CabinNum` | Cabin 拆出的舱号（0~1894） |
| `Side` | Cabin 拆出的舷侧：P=Port、S=Starboard |
| `Surname` | 姓氏（同姓 ≠ 同组，两者互补） |
| `FirstName` | 名 |
| `SurnameSize` | 同姓氏人数。**必须合并 train+test 计算**，否则是数据集指纹（见 notebook 04） |
| `SurnameCount` | 同姓氏计数（同 SurnameSize，含缺失处理差异） |
| `GroupSurnameCount` | 本组内出现的不同姓氏数，>1 表示拼组 |
| `IsMixedGroup` | 是否拼组（GroupSurnameCount > 1） |
| `TotalSpend` | 五项消费合计 |
| `LogTotalSpend` | log1p(TotalSpend)。与目标相关性 -0.47，远强于原始值 |
| `ZeroSpend` | 五项消费是否为 0 |
| `LuxurySpend` | Spa + VRDeck + ShoppingMall |
| `BasicSpend` | RoomService + FoodCourt |
| `NZerosAmenities` | 五项中为 0 的项数 |
| `NAmenitiesUsed` | 五项中消费>0 的项数 |
| `LuxuryShare` | 豪华消费占比；总消费为 0 时记 0（用 ZeroSpend 区分） |
| `MaxAmenity` | 消费最多的项目；无消费时为 __None__ |
| `IsChild` | 是否 13 岁以下 |
| `AgeBand` | 年龄段分箱 |
| `CabinZone` | 舱号区间分箱（0-299/300-599/600-899/900-1199/1200-1499/1500+） |

## 建模矩阵（`src.preprocess.make_model_matrix`）

- 数值/布尔列（23 个）原样进入
- 分类列（HomePlanet, Destination, Deck, Side, AgeBand, MaxAmenity, CabinZone）做 one-hot
- 最终特征维度：**59**
- indicator 策略会额外纳入原始字段缺失标记（最多 12 列），标签不参与。

## 折划分（`folds.csv`）

按 `GroupId` 切的 5 折（`StratifiedGroupKFold`，seed=42）。
**不要自己重新随机切分**——同组乘客跨 fold 会造成泄漏，本地分数虚高。
按 PassengerId 关联，不依赖 CSV 当前行号；读取 GroupId 时使用字符串以保留前导零。
