# 派生特征说明

由 `export_features.py` 生成，与 `notebooks/*.ipynb` 中的口径一致。

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

## 建模矩阵（`preprocess.make_model_matrix`）

- 数值/布尔列（23 个）原样进入
- 分类列（HomePlanet, Destination, Deck, Side, AgeBand, MaxAmenity）做 one-hot
- 最终特征维度：**53**

## 折划分（`folds.csv`）

按 `GroupId` 切的 5 折（`StratifiedGroupKFold`，seed=42）。
**不要自己重新随机切分**——同组乘客跨 fold 会造成泄漏，本地分数虚高。
