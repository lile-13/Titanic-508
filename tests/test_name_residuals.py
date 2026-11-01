import unittest

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from src.name_residuals import NameResidual,objective


class NameResidualTests(unittest.TestCase):
    def test_analytic_gradient(self):
        D=csr_matrix([[1,1,0],[1,0,1],[1,1,0],[1,0,0]],dtype=float)
        y=np.array([1,0,0,1]);offset=np.array([-.3,.5,.4,-.2]);beta=np.array([.1,-.2,.3])
        _,gradient=objective(beta,D,y,offset,4.)
        numeric=[]
        for i in range(3):
            delta=np.zeros(3);delta[i]=1e-6
            numeric.append((objective(beta+delta,D,y,offset,4.)[0]-objective(beta-delta,D,y,offset,4.)[0])/2e-6)
        np.testing.assert_allclose(gradient,numeric,rtol=0,atol=1e-7)

    def test_unknown_missing_names_batch_and_target_column(self):
        frame=pd.DataFrame(dict(FirstName=["A"]*30+["B"]*30,Surname=["same"]*60,Transported=np.zeros(60)))
        model=NameResidual().fit(frame,np.array([1]*30+[0]*30),np.full(60,.5))
        query=pd.DataFrame(dict(FirstName=["A","B","unseen",None],Surname=["same","same","unseen",None]))
        p=model.predict(query,np.full(4,.5))
        self.assertGreater(p[0],.5);self.assertLess(p[1],.5)
        np.testing.assert_allclose(p[2:],[.5,.5],rtol=0,atol=1e-8)
        np.testing.assert_allclose(model.predict(query.iloc[[0]],np.array([.5])),p[[0]],rtol=0,atol=1e-12)
        query["Transported"]=[1,0,1,0]
        np.testing.assert_array_equal(p,model.predict(query,np.full(4,.5)))


if __name__=="__main__":unittest.main()
