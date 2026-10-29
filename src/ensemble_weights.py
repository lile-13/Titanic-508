"""按组件身份重设已保存模型权重，不依赖模型在列表中的顺序。"""
import numpy as np


def component_key(member):
    return (member["encoder"].family,member["builder"].feature_set,member["builder"].strategy)


def reweight_members(members,plan,weights):
    weights=np.asarray(weights,dtype=float)
    if len(weights)!=len(plan["members"]) or not np.isfinite(weights).all() or (weights<0).any() or not np.isclose(weights.sum(),1.):
        raise ValueError("新权重必须非负、有限、合计为1并对应全部组件")
    mapping={}
    for i,choice in enumerate(plan["members"]):
        c=choice["config"];key=(c["family"],c["feature_set"],c["strategy"])
        if key in mapping or plan["weights"][i]<=0:raise ValueError("组件身份重复或旧权重为零")
        mapping[key]=i
    old_totals=np.zeros(len(weights));new_totals=np.zeros(len(weights));result=[]
    for member in members:
        key=component_key(member)
        if key not in mapping:raise ValueError("模型组件与冻结配置不匹配")
        i=mapping[key];old_totals[i]+=member["weight"]
        new=dict(member,weight=float(member["weight"]*weights[i]/plan["weights"][i]))
        new_totals[i]+=new["weight"];result.append(new)
    np.testing.assert_allclose(old_totals,plan["weights"],atol=1e-12,rtol=0)
    np.testing.assert_allclose(new_totals,weights,atol=1e-12,rtol=0)
    return result
