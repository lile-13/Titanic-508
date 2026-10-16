"""Spaceship Titanic 可复用预处理模块（成员2 产出）。

设计原则
--------
1. **不碰标签**：这里所有填充/派生只用到特征列本身，读不到 `Transported`，
   因此不存在目标泄漏。附带好处是同一套代码可以直接用在 test 上。
2. **不在 train/test 之间传信息**：组号 `gggg` 在 train 与 test 中互不重叠
   （已核实交集为 0），所以"用同组其他人补缺"不会把 test 的信息带进 train。
3. **fit/transform 分离**：全局统计量（中位数、众数）只在 `fit` 阶段从训练集学，
   `transform` 复用；组内统计量是逐行局部的，无需持久化。

数据事实（均已用本数据集核实，见 notebooks/01、03）
--------------------------------------------------
- `PassengerId` = `gggg_pp`，`gggg` 为同行组号，`pp` 为组内序号（01–08）。
- `Cabin` = `Deck/Num/Side`，`Side` 在组内恒定；`Deck`、`Num` 在组内**可以不同**。
- `HomePlanet` 在组内恒定；`Destination` 在组内**可以不同**。
- `CryoSleep=True` 的乘客五项消费恒为 0（0 例外）；反之消费>0 可反推非冷冻睡眠。
- 同姓氏 ≠ 同组（2217 个姓氏中 1664 个跨多个组），姓氏与组号是两套信息。

用法
----
    import preprocess as pp
    train, test = pp.load_raw()
    train = pp.build_features(train)
    test = pp.build_features(test)
    prep = pp.Preprocessor("rule_group").fit(train)
    train_clean = prep.transform(train)
    X_train, y = pp.make_model_matrix(train_clean, target=True)
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

# --- 字段分组 ---------------------------------------------------------------
SPEND_COLS = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
# 这三项偏"个人享受"，另外两项偏"餐饮/购物"
LUXURY_COLS = ["Spa", "VRDeck", "ShoppingMall"]
BASIC_COLS = ["RoomService", "FoodCourt"]
RAW_CAT_COLS = ["HomePlanet", "Destination", "Deck", "Side"]
RAW_BOOL_COLS = ["CryoSleep", "VIP"]
TARGET = "Transported"
ID_COL = "PassengerId"

# 原始规模，用于断言数据没被换过
EXPECTED_SHAPES = {"train": (8693, 14), "test": (4277, 13)}


# --- 载入 -------------------------------------------------------------------
def find_data_dir(start: str | os.PathLike | None = None) -> Path:
    """定位存放 train.csv 的目录。

    依次尝试：给定目录 / 当前目录及其上两级（兼容在 notebooks/ 下运行）/
    仓库常见的 `csv/` 子目录 / 本机 D:\\Titanic-508\\csv 与临时目录。
    """
    candidates: list[Path] = []
    if start is not None:
        candidates.append(Path(start))
    here = Path.cwd()
    candidates += [here, here.parent, here.parent.parent]
    candidates += [here / "csv", here.parent / "csv", here.parent.parent / "csv"]

    for base in candidates:
        try:
            if (base / "train.csv").is_file():
                return base.resolve()
        except OSError:
            continue

    for fallback in (r"D:\Titanic-508\csv",
                     os.path.join(os.environ.get("TEMP", ""), "")):
        p = Path(fallback)
        if p and (p / "train.csv").is_file():
            return p.resolve()

    raise FileNotFoundError(
        "找不到 train.csv。请把 csv/ 放在当前目录或其上一级，或显式传 data_dir。"
    )


def load_raw(data_dir: str | os.PathLike | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """读取 train.csv / test.csv（原始列，未做任何加工）。"""
    d = find_data_dir(data_dir)
    train = pd.read_csv(d / "train.csv")
    test = pd.read_csv(d / "test.csv")
    return train, test


def load_submission_template(data_dir=None) -> pd.DataFrame:
    d = find_data_dir(data_dir)
    return pd.read_csv(d / "sample_submission.csv")


# --- 体检 -------------------------------------------------------------------
def basic_report(df: pd.DataFrame) -> pd.DataFrame:
    """逐列体检表：类型、缺失、唯一值、示例值。"""
    rows = []
    for col in df.columns:
        s = df[col]
        n_na = int(s.isna().sum())
        examples = s.dropna().unique()[:3]
        rows.append({
            "列名": col,
            "dtype": str(s.dtype),
            "非空数": int(s.notna().sum()),
            "缺失数": n_na,
            "缺失率%": round(n_na / len(df) * 100, 2),
            "唯一值": int(s.nunique(dropna=True)),
            "示例": ", ".join(map(str, examples)),
        })
    return pd.DataFrame(rows)


def duplicate_report(df: pd.DataFrame) -> dict:
    """重复记录检查：整行重复与主键重复分开看。"""
    return {
        "行数": len(df),
        "整行完全重复": int(df.duplicated().sum()),
        f"{ID_COL} 重复": int(df[ID_COL].duplicated().sum()),
        "列名重复": int(pd.Index(df.columns).duplicated().sum()),
    }


def consistency_report(df: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
    """组内一致性：同一个 `gggg` 组里，某列是否只有一个取值。

    这是判断"该列能否当组级特征用"的依据。
    """
    d = ensure_ids(df)
    # 只考察"有可能是组级属性"的字段；Age/CryoSleep/VIP/Name 是个人属性，
    # 本来就应该在组内不同，放进来没有意义。
    cols = cols or ["HomePlanet", "Destination", "Deck", "Side", "Surname"]
    rows = []
    for col in cols:
        if col not in d.columns:
            continue
        nun = d.groupby("GroupId", observed=True)[col].nunique(dropna=True)
        multi = d.groupby("GroupId", observed=True).size()
        multi_ids = multi[multi > 1].index
        rows.append({
            "列名": col,
            "多成员组数": len(multi_ids),
            "组内取值>1 的组数": int((nun.reindex(multi_ids) > 1).sum()),
        })
    return pd.DataFrame(rows)


# --- 结构解析 ---------------------------------------------------------------
def ensure_ids(df: pd.DataFrame) -> pd.DataFrame:
    """补上 GroupId / GroupNum（幂等）。"""
    if "GroupId" not in df.columns:
        df["GroupId"] = df[ID_COL].str.split("_").str[0]
    if "GroupNum" not in df.columns:
        df["GroupNum"] = df[ID_COL].str.split("_").str[1].astype(int)
    return df


def ensure_cabin_parts(df: pd.DataFrame) -> pd.DataFrame:
    """拆 Cabin -> Deck / CabinNum / Side（幂等）。"""
    if "Deck" in df.columns and "CabinNum" in df.columns and "Side" in df.columns:
        return df
    parts = df["Cabin"].str.split("/", expand=True)
    df["Deck"] = parts[0]
    df["CabinNum"] = pd.to_numeric(parts[1], errors="coerce")
    df["Side"] = parts[2]
    return df


def ensure_name_parts(df: pd.DataFrame) -> pd.DataFrame:
    """拆 Name -> FirstName / Surname（幂等）。"""
    if "Surname" not in df.columns:
        df["Surname"] = df["Name"].str.split().str[-1]
    if "FirstName" not in df.columns:
        df["FirstName"] = df["Name"].str.split().str[0]
    return df


def normalize_types(df: pd.DataFrame) -> pd.DataFrame:
    """字段类型规整。

    `CryoSleep` / `VIP` 在 CSV 里是 True/False/空，pandas 读成 object；
    换成可空布尔 `boolean` 才能干净地表达"三态"。分类列转 category 便于统计。
    """
    df = df.copy()
    for col in RAW_BOOL_COLS:
        if col in df.columns:
            df[col] = df[col].map({True: True, False: False, "True": True, "False": False}
                                  ).astype("boolean")
    for col in RAW_CAT_COLS:
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df


# --- 领域规则 ---------------------------------------------------------------
def apply_domain_rules(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """用已核实的确定性规则补 `CryoSleep`，并返回规则命中统计。

    规则：
      R1  五项消费合计 > 0  -> 必定不在冷冻睡眠（消费>0 的人 100% 是 CryoSleep=False）
      R2  CryoSleep=True     -> 五项消费必为 0

    只补"能由特征本身推出"的部分，不做任何猜测性填充。
    """
    df = df.copy()
    total = df[SPEND_COLS].sum(axis=1, min_count=1)
    df["TotalSpend"] = total.fillna(0.0)

    r1 = df["CryoSleep"].isna() & (total > 0)
    n_r1 = int(r1.sum())
    df.loc[r1, "CryoSleep"] = False

    r2 = (df["CryoSleep"] == True) & (total.isna())  # noqa: E712
    n_r2 = int(r2.sum())
    df.loc[df["CryoSleep"] == True, SPEND_COLS] = (  # noqa: E712
        df.loc[df["CryoSleep"] == True, SPEND_COLS].fillna(0.0)
    )

    stats = {"规则R1命中(消费>0判定非冷冻)": n_r1, "规则R2命中(冷冻者消费补0)": n_r2}
    return df, stats


# --- 特征派生 ---------------------------------------------------------------
def add_group_features(df: pd.DataFrame) -> pd.DataFrame:
    """同行组相关特征。组大小与目标强相关（单人 45% vs 四人组 64%）。"""
    d = ensure_ids(df)
    d["GroupSize"] = d.groupby("GroupId", observed=True)[ID_COL].transform("size")
    d["IsAlone"] = (d["GroupSize"] == 1)
    d["IsFirstInGroup"] = (d["GroupNum"] == 1)
    d["IsLastInGroup"] = (d["GroupNum"] == d["GroupSize"])
    # 组内相对序号（0~1），比绝对序号更稳
    d["GroupPos"] = np.where(d["GroupSize"] > 1,
                             (d["GroupNum"] - 1) / (d["GroupSize"] - 1), 0.0)
    return d


def add_name_features(df: pd.DataFrame) -> pd.DataFrame:
    """姓氏相关特征。姓氏 ≠ 组号，两者互补：同组可能跨姓氏（拼组）。"""
    d = ensure_name_parts(df)
    # 姓氏缺失时 groupby 会把该行排除在外，这里补 1（当作"独姓"），避免留下 NaN
    d["SurnameSize"] = d.groupby("Surname", observed=True)[ID_COL].transform("size").fillna(1)
    d["SurnameCount"] = d.groupby("Surname", observed=True)[ID_COL].transform("count").fillna(1)
    # 该组内出现的不同姓氏数量：>1 说明是"拼组"
    d["GroupSurnameCount"] = d.groupby("GroupId", observed=True)["Surname"].transform("nunique")
    d["IsMixedGroup"] = d["GroupSurnameCount"] > 1
    return d


def add_spend_features(df: pd.DataFrame) -> pd.DataFrame:
    """消费行为特征。原始消费极度右偏，取 log1p 更线性。"""
    d = df.copy()
    spend = d[SPEND_COLS]
    d["TotalSpend"] = spend.sum(axis=1, min_count=1).fillna(0.0)
    d["LogTotalSpend"] = np.log1p(d["TotalSpend"])
    d["ZeroSpend"] = d["TotalSpend"] == 0
    d["LuxurySpend"] = d[LUXURY_COLS].sum(axis=1, min_count=1).fillna(0.0)
    d["BasicSpend"] = d[BASIC_COLS].sum(axis=1, min_count=1).fillna(0.0)
    d["NZerosAmenities"] = (spend == 0).sum(axis=1)
    d["NAmenitiesUsed"] = (spend.fillna(0) > 0).sum(axis=1)
    # 总消费为 0 时"豪华占比"无定义：这里记 0，由 ZeroSpend 标志位负责区分"没花钱"与"全花在基础项"
    d["LuxuryShare"] = np.where(d["TotalSpend"] > 0,
                                d["LuxurySpend"] / d["TotalSpend"].replace(0, np.nan), 0.0)
    # 同理：没消费就没有"最大消费项"
    d["MaxAmenity"] = np.where(d["TotalSpend"] > 0,
                               spend.fillna(0.0).idxmax(axis=1), "__None__")
    return d


def add_age_features(df: pd.DataFrame) -> pd.DataFrame:
    """年龄段特征（Age 是连续值，但儿童/成人行为差异明显）。"""
    d = df.copy()
    d["IsChild"] = d["Age"] < 13
    d["AgeBand"] = pd.cut(
        d["Age"], bins=[-1, 0, 12, 17, 25, 35, 50, 65, 100],
        labels=["0", "1-12", "13-17", "18-25", "26-35", "36-50", "51-65", "66+"],
    )
    return d


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """一步到位：解析结构 + 类型规整 + 领域规则 + 全部派生特征。"""
    d = normalize_types(df)
    d = ensure_ids(d)
    d = ensure_cabin_parts(d)
    d = ensure_name_parts(d)
    d = add_group_features(d)
    d = add_name_features(d)
    d = add_spend_features(d)
    d = add_age_features(d)
    return d


def build_features_combined(train: pd.DataFrame,
                            test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """**推荐入口**：把 train/test 拼在一起算派生特征，再拆回两个表。

    为什么必须这样——`SurnameSize` / `SurnameCount` 是"同一姓氏有多少人"的计数：

    - 组号 `gggg` 在 train/test 之间**不重叠**（已核实交集 0），所以 `GroupSize` 怎么算都对；
    - 但**姓氏是重叠的**：1536 个姓氏同时出现在两边，覆盖 train 6137 人、test 3759 人。
      若两边各算各的，train 里这些姓氏的计数天然偏大（train 人数约是 test 的 2 倍）：
      实测 train 的 `SurnameSize` 均值 5.33，test 只有 3.51。

    后果不是"分数低一点"这么简单——这个特征变成了**"我来自哪个数据集"的指纹**：
    notebook 04 的对抗验证 AUC 从 0.52 直接涨到 **0.73**。树模型会在
    `SurnameSize` 上切出一个阈值，而这个阈值在两边含义完全不同。

    合并计算后两侧口径一致（train 均值 7.085 vs test 7.079，对抗 AUC 回落到 0.545）。

    这里只用到特征列，完全不碰 `Transported`，所以**不存在目标泄漏**；
    实际上用的是"test 也有这些列"这一公开信息，Kaggle 允许。
    """
    a = train.copy()
    b = test.copy()
    a["_is_test"] = False
    b["_is_test"] = True
    both = pd.concat([a, b], ignore_index=True, sort=False)
    both = build_features(both)
    tr = both[~both["_is_test"]].drop(columns=["_is_test"]).reset_index(drop=True)
    te = both[both["_is_test"]].drop(columns=["_is_test"]).reset_index(drop=True)

    # 拼接会让标签列在 test 侧全为 NaN、并把它降级成 object。这里收拾干净：
    # 训练集标签还原成 bool；测试集**整列删掉**——留着会是个坑，
    # 例如 Preprocessor("indicator").fit(test) 会给它加一个全 True 的 `Transported_na`，
    # 那就等于把"这条来自 test"直接写进特征里。
    if TARGET in tr.columns:
        tr[TARGET] = tr[TARGET].astype(bool)
    if TARGET in te.columns and te[TARGET].isna().all():
        te = te.drop(columns=[TARGET])
    return tr, te


# --- 缺失值处理 -------------------------------------------------------------
STRATEGIES = ("drop", "simple", "indicator", "rule_group")


class Preprocessor:
    """缺失值处理。

    strategy:
        "drop"       删除任何含缺失的行（只对训练集有意义，test 不能删）
        "simple"     数值取中位数、分类取众数
        "indicator"  在 simple 基础上，为每个原本有缺失的列加 `<列>_na` 标志位
        "rule_group" 领域规则 + 同组信息 + 兜底统计量（推荐）
    """

    def __init__(self, strategy: str = "rule_group", use_domain_rules: bool = True):
        if strategy not in STRATEGIES:
            raise ValueError(f"strategy 必须是 {STRATEGIES} 之一，收到 {strategy!r}")
        self.strategy = strategy
        # 关掉它可做消融实验，量化"领域规则本身"带来多少收益
        self.use_domain_rules = use_domain_rules
        self.numeric_fill_: dict[str, float] = {}
        self.cat_fill_: dict[str, object] = {}
        self.age_by_planet_: dict[str, float] = {}
        self.feature_fill_: dict[str, float] = {}
        self.na_columns_: list[str] = []
        self._cols: list[str] = []

    # -- 需要填充的列 --------------------------------------------------------
    @staticmethod
    def _numeric_cols(df: pd.DataFrame) -> list[str]:
        return [c for c in SPEND_COLS + ["Age", "CabinNum"] if c in df.columns]

    @staticmethod
    def _categorical_cols(df: pd.DataFrame) -> list[str]:
        return [c for c in RAW_CAT_COLS + ["CryoSleep", "VIP"] if c in df.columns]

    def fit(self, train: pd.DataFrame) -> "Preprocessor":
        df = train
        self._cols = list(df.columns)
        for col in self._numeric_cols(df):
            self.numeric_fill_[col] = float(df[col].median())
        for col in self._categorical_cols(df):
            mode = df[col].mode(dropna=True)
            self.cat_fill_[col] = mode.iloc[0] if len(mode) else np.nan
        if "HomePlanet" in df.columns:
            self.age_by_planet_ = (df.groupby("HomePlanet", observed=True)["Age"]
                                   .median().to_dict())
        # 兜底统计量：保证 transform 之后"模型要用的列"一个缺失都没有
        for col in NUM_FEATURES:
            if col in df.columns:
                v = pd.to_numeric(df[col], errors="coerce").median()
                self.feature_fill_[col] = float(v) if pd.notna(v) else 0.0
        self.na_columns_ = [c for c in df.columns if df[c].isna().any()]
        return self

    def _finalize(self, d: pd.DataFrame) -> pd.DataFrame:
        """收尾：模型特征列不许有缺失。只动特征列，原始文本列原样保留。

        这是对下游的硬保证——`make_model_matrix` 拿到的数值/布尔/分类列一定干净。
        """
        for col in NUM_FEATURES:
            if col in d.columns:
                s = pd.to_numeric(d[col], errors="coerce")
                if s.isna().any():
                    s = s.fillna(self.feature_fill_.get(col, 0.0))
                d[col] = s
        for col in BOOL_FEATURES:
            if col in d.columns:
                d[col] = d[col].astype("object").fillna(False).astype(bool)
        for col in CAT_FEATURES:
            if col in d.columns:
                d[col] = d[col].astype("object").fillna(self.cat_fill_.get(col, "__missing__"))
        return d

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()

        # 标志位要在填充之前算
        if self.strategy == "indicator":
            for col in self.na_columns_:
                if col in d.columns:
                    d[f"{col}_na"] = d[col].isna()

        if self.strategy == "drop":
            before = len(d)
            d = d.dropna(subset=[c for c in d.columns if c in self.na_columns_]).reset_index(drop=True)
            print(f"[drop] {before} -> {len(d)} 行（丢弃 {before - len(d)}，"
                  f"{100 * (before - len(d)) / before:.1f}%）")
            return d

        if self.strategy in ("simple", "indicator"):
            for col, v in self.numeric_fill_.items():
                if col in d.columns:
                    d[col] = d[col].fillna(v)
            for col, v in self.cat_fill_.items():
                if col in d.columns:
                    d[col] = d[col].astype("object").fillna(v)

        elif self.strategy == "rule_group":
            d = ensure_ids(d)
            spend_total = d[SPEND_COLS].sum(axis=1, min_count=1)

            # 1) 分类列：同组取值 -> 全局众数。HomePlanet/Side 组内恒定，效果最好。
            for col in ["HomePlanet", "Side", "Destination", "Deck"]:
                if col in d.columns:
                    known = d.dropna(subset=[col]).groupby("GroupId", observed=True)[col].first()
                    d[col] = d[col].fillna(d["GroupId"].map(known))
                    d[col] = d[col].astype("object").fillna(self.cat_fill_.get(col, np.nan))

            # 2) CryoSleep：消费>0 -> 非冷冻；其余用同组多数，再退到众数
            if "CryoSleep" in d.columns:
                if self.use_domain_rules:
                    d.loc[d["CryoSleep"].isna() & (spend_total > 0), "CryoSleep"] = False
                grp = (d.dropna(subset=["CryoSleep"])
                       .groupby("GroupId", observed=True)["CryoSleep"]
                       .agg(lambda s: s.mode().iloc[0] if len(s.mode()) else pd.NA))
                d["CryoSleep"] = d["CryoSleep"].fillna(d["GroupId"].map(grp))
                d["CryoSleep"] = d["CryoSleep"].astype("object").fillna(
                    self.cat_fill_.get("CryoSleep", False))
                d["CryoSleep"] = d["CryoSleep"].astype("boolean")

            # 3) 五项消费：冷冻者必为 0；其余用同组中位数，再退到全局中位数
            for col in SPEND_COLS:
                if col not in d.columns:
                    continue
                if self.use_domain_rules and "CryoSleep" in d.columns:
                    d.loc[(d[col].isna()) & (d["CryoSleep"] == True), col] = 0.0  # noqa: E712
                grp_med = (d.dropna(subset=[col])
                           .groupby("GroupId", observed=True)[col].median())
                d[col] = d[col].fillna(d["GroupId"].map(grp_med))
                d[col] = d[col].fillna(self.numeric_fill_.get(col, 0.0))

            # 4) Age：同组中位数 -> 出发星球中位数 -> 全局中位数
            if "Age" in d.columns:
                grp_med = (d.dropna(subset=["Age"])
                           .groupby("GroupId", observed=True)["Age"].median())
                d["Age"] = d["Age"].fillna(d["GroupId"].map(grp_med))
                if "HomePlanet" in d.columns and self.age_by_planet_:
                    d["Age"] = d["Age"].fillna(
                        d["HomePlanet"].map(self.age_by_planet_).astype(float))
                d["Age"] = d["Age"].fillna(self.numeric_fill_.get("Age", 0.0))

            # 4b) CabinNum：Cabin 缺失时用同组中位数（同组常在同一舱号附近）
            if "CabinNum" in d.columns:
                grp_med = (d.dropna(subset=["CabinNum"])
                           .groupby("GroupId", observed=True)["CabinNum"].median())
                d["CabinNum"] = d["CabinNum"].fillna(d["GroupId"].map(grp_med))
                d["CabinNum"] = d["CabinNum"].fillna(self.numeric_fill_.get("CabinNum", np.nan))

            # 5) VIP：同组多数 -> 众数
            if "VIP" in d.columns:
                grp = (d.dropna(subset=["VIP"])
                       .groupby("GroupId", observed=True)["VIP"]
                       .agg(lambda s: s.mode().iloc[0] if len(s.mode()) else pd.NA))
                d["VIP"] = d["VIP"].fillna(d["GroupId"].map(grp))
                d["VIP"] = d["VIP"].astype("object").fillna(
                    self.cat_fill_.get("VIP", False))
                d["VIP"] = d["VIP"].astype("boolean")

        # 派生特征在填充之后重算，保证它们和填充后的值一致
        d = add_spend_features(d)
        d = add_age_features(d)
        return self._finalize(d)

    def fit_transform(self, train: pd.DataFrame) -> pd.DataFrame:
        return self.fit(train).transform(train)


# --- 给建模用 ---------------------------------------------------------------
def group_folds(df: pd.DataFrame, n_splits: int = 5, seed: int = 42, shuffle: bool = True):
    """按 `GroupId` 分组切分，返回 (train_idx, valid_idx) 列表。

    **必须按组切**：同组乘客高度相关（同舱、同行程、常同结果），
    随机切会让同一组的人同时出现在训练和验证集里，本地分数虚高。
    """
    from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

    d = ensure_ids(df)
    if TARGET not in d.columns:
        raise ValueError(f"df 里没有 {TARGET} 列，无法做带分层的分组切分")
    # 拼接/填充后标签可能变成 float 或 object，StratifiedGroupKFold 只认离散标签
    y_num = pd.to_numeric(d[TARGET], errors="coerce")
    if y_num.isna().any():
        raise ValueError(f"{TARGET} 含有缺失值，无法切分")
    y = y_num.astype(int).values
    if shuffle:
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(cv.split(d, y, groups=d["GroupId"]))
    cv = GroupKFold(n_splits=n_splits)
    return list(cv.split(d, y, groups=d["GroupId"]))


CAT_FEATURES = ["HomePlanet", "Destination", "Deck", "Side", "AgeBand", "MaxAmenity"]
NUM_FEATURES = (SPEND_COLS + ["Age", "TotalSpend", "LogTotalSpend", "LuxurySpend",
                              "BasicSpend", "NZerosAmenities", "NAmenitiesUsed",
                              "LuxuryShare", "GroupSize", "GroupPos", "SurnameSize",
                              "GroupSurnameCount"])
BOOL_FEATURES = ["CryoSleep", "VIP", "ZeroSpend", "IsAlone", "IsChild", "IsMixedGroup"]


def make_model_matrix(df: pd.DataFrame, target: bool = False):
    """把清洗后的表变成模型可用的稠密矩阵。

    分类列 one-hot（`handle_unknown` 语义：未见过的取值整列为 0），
    布尔列转 0/1，数值列原样。返回 (X, 特征名) 或 (X, y, 特征名)。
    """
    d = df.copy()
    cat_cols = [c for c in CAT_FEATURES if c in d.columns]
    bool_cols = [c for c in BOOL_FEATURES if c in d.columns]
    num_cols = [c for c in NUM_FEATURES if c in d.columns]

    parts = []
    for col in bool_cols + num_cols:
        s = d[col]
        if str(s.dtype) == "boolean":
            s = s.astype("float")
        parts.append(pd.to_numeric(s, errors="coerce").rename(col))
    num_df = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=d.index)

    if cat_cols:
        cat_df = pd.get_dummies(
            d[cat_cols].astype("object").fillna("__missing__"),
            columns=cat_cols, dummy_na=False, dtype=float,
        )
        X = pd.concat([num_df, cat_df], axis=1)
    else:
        X = num_df

    X = X.replace([np.inf, -np.inf], np.nan)
    if target:
        return X, d[TARGET].astype(int).values
    return X


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """缺失总览（数量/比例/类型），按缺失率降序。"""
    na = df.isna().sum()
    out = pd.DataFrame({
        "缺失数": na,
        "缺失率%": (na / len(df) * 100).round(2),
        "dtype": df.dtypes.astype(str),
    })
    return out[out["缺失数"] > 0].sort_values("缺失率%", ascending=False)


__all__ = [
    "SPEND_COLS", "LUXURY_COLS", "BASIC_COLS", "RAW_CAT_COLS", "RAW_BOOL_COLS",
    "TARGET", "ID_COL", "STRATEGIES",
    "find_data_dir", "load_raw", "load_submission_template",
    "basic_report", "duplicate_report", "consistency_report", "missing_summary",
    "ensure_ids", "ensure_cabin_parts", "ensure_name_parts", "normalize_types",
    "apply_domain_rules", "add_group_features", "add_name_features",
    "add_spend_features", "add_age_features", "build_features",
    "build_features_combined",
    "Preprocessor", "group_folds", "make_model_matrix",
]
