"""并行补算同一确定性搜索计划的尾部任务，不改变试验配置或选择规则。"""
import argparse
from contextlib import redirect_stdout
from io import StringIO

from src.optimization import SearchRunner


class ScheduleRunner(SearchRunner):
    def evaluate(self, config, stage, folds=(1, 2, 3, 4), early=True):
        if stage == "tune":
            self.scheduled.append(config)
            return None
        return super().evaluate(config, stage, folds, early)

    def export_table(self, family=None):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="optimization_20261025")
    parser.add_argument("--family", choices=["catboost", "lightgbm"], default="catboost")
    parser.add_argument("--trials", type=int, default=32)
    parser.add_argument("--start", type=int, default=16, help="0-based start, inclusive")
    parser.add_argument("--stop", type=int, default=32, help="0-based stop, exclusive")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if not 0 <= args.start < args.stop <= args.trials:
        raise ValueError("要求 0 <= start < stop <= trials")
    scheduler = ScheduleRunner(args.name, args.threads)
    # 先由原搜索函数生成相同计划；筛选/Profile阶段必须已完成，避免重复开跑。
    if len(scheduler.records(args.family, ["profile"])) < 3:
        raise ValueError("先等主进程完成三个完整开发折的特征方案比较")
    scheduler.scheduled = []
    with redirect_stdout(StringIO()):
        scheduler.search(args.family, args.trials)
    worker = SearchRunner(args.name, args.threads)
    for index in reversed(range(args.start, args.stop)):
        print(f"Worker trial {index+1}/{args.trials}", flush=True)
        worker.evaluate(scheduler.scheduled[index], "tune")


if __name__ == "__main__":
    main()
