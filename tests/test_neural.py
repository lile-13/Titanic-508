import unittest
import numpy as np

from src.neural_models import estimator,fit_group_mlp


class NeuralTests(unittest.TestCase):
    def test_epoch_selection_uses_disjoint_inner_groups_and_refits_all_training_rows(self):
        X=np.arange(80,dtype=float).reshape(40,2)/80
        y=np.tile([0,1],20);groups=np.repeat(np.arange(20),2)
        learner=estimator(dict(variant=0),42,[]).set_params(batch_size=8)
        model=fit_group_mlp(learner,X,y,groups,max_epochs=3,patience=2)
        record=model.epoch_selection_
        self.assertEqual(record["inner_group_overlap"],0)
        self.assertEqual(record["inner_train_rows"]+record["inner_valid_rows"],40)
        self.assertGreaterEqual(record["selected_epoch"],1)
        self.assertLessEqual(record["selected_epoch"],3)
        self.assertEqual(model.n_iter_,record["selected_epoch"])
        self.assertEqual(model.predict_proba(X).shape,(40,2))


if __name__=="__main__":unittest.main()
