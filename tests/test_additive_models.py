import unittest
import numpy as np

from src.additive_models import group_bags


class GroupBagsTest(unittest.TestCase):
    def test_whole_groups_stay_together_in_every_internal_bag(self):
        groups=np.repeat(np.arange(30),np.arange(1,31)%5+1)
        bags=group_bags(groups,42)
        self.assertEqual(bags.shape,(len(groups),6))
        for bag in bags.T:
            self.assertEqual(set(bag),{-1,1})
            for group in np.unique(groups):
                self.assertEqual(len(np.unique(bag[groups==group])),1)
        np.testing.assert_array_equal(bags,group_bags(groups,42))
        self.assertFalse(np.array_equal(bags,group_bags(groups,103)))

    def test_invalid_groups_rejected(self):
        for groups in [[],[1,1,1],[[1,2],[3,4]]]:
            with self.assertRaises(ValueError):group_bags(groups,42)


if __name__=="__main__":unittest.main()
