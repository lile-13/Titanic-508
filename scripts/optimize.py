"""高分候选搜索：python -m scripts.optimize --phase search --family catboost。"""
import argparse
import json

from src.optimization import SearchRunner


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["search","freeze","finalize"], required=True)
    parser.add_argument("--family", choices=["catboost","lightgbm"], default="catboost")
    parser.add_argument("--trials", type=int, default=32)
    parser.add_argument("--threads",type=int,default=4)
    parser.add_argument("--name",default="optimization_20261025")
    args=parser.parse_args()
    runner=SearchRunner(args.name,args.threads)
    if args.phase=="search":runner.search(args.family,args.trials)
    elif args.phase=="freeze":print(json.dumps(runner.freeze(),ensure_ascii=False,indent=2))
    else:print(json.dumps(runner.finalize(),ensure_ascii=False,indent=2))


if __name__=="__main__":main()
