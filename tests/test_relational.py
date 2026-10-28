import unittest

import numpy as np
import pandas as pd

from src.relational_features import GroupCrossFitEncoder


class RelationalTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({"GroupId": np.repeat(["a","b","c","d"], 2),
                                   "Surname": ["Shared","Shared","Shared","B","C","C","D",None],
                                   "Deck": ["A"]*8,"CabinNum": np.arange(8)*80.,"Side": ["P"]*8})
        self.y = np.array([0,1,1,0,1,1,0,0])

    def test_entire_own_group_targets_are_excluded(self):
        first = GroupCrossFitEncoder(spatial=True).fit_transform(self.frame,self.y)
        for group in self.frame.GroupId.unique():
            mask = self.frame.GroupId == group
            changed = self.y.copy();changed[mask] = 1-changed[mask]
            second = GroupCrossFitEncoder(spatial=True).fit_transform(self.frame,changed)
            pd.testing.assert_frame_equal(first.loc[mask],second.loc[mask])

    def test_unseen_keys_use_training_prior_and_transform_ignores_labels(self):
        encoder = GroupCrossFitEncoder().fit(self.frame,self.y)
        frame = self.frame.iloc[:2].assign(Surname=["Unseen",None],Transported=[True,False])
        out = encoder.transform(frame)
        np.testing.assert_allclose(out.te_Surname, self.y.mean())
        np.testing.assert_array_equal(out.te_count_Surname, 0)
        pd.testing.assert_frame_equal(out,encoder.transform(frame.assign(Transported=[False,True])))


if __name__ == "__main__":
    unittest.main()
