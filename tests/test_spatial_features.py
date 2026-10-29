import unittest
import numpy as np
import pandas as pd

from src.spatial_features import SpatialTargetEncoder


def sample():
    return pd.DataFrame(dict(GroupId=np.repeat(np.arange(8),2),Deck="F",Side="P",
        CryoSleep=False,CabinNum=np.repeat(np.arange(8)*5.,2)))


class SpatialEncoderTest(unittest.TestCase):
    def test_own_entire_group_labels_cannot_change_own_training_features(self):
        frame=sample();y=np.arange(len(frame))%2
        baseline=SpatialTargetEncoder().fit_transform(frame,y)
        for group in np.unique(frame.GroupId):
            altered=y.copy();mask=frame.GroupId.eq(group).to_numpy();altered[mask]=1-altered[mask]
            changed=SpatialTargetEncoder().fit_transform(frame,altered)
            np.testing.assert_array_equal(baseline.loc[mask],changed.loc[mask])

    def test_window_boundaries_missing_and_unseen_regions(self):
        frame=sample();encoder=SpatialTargetEncoder().fit(frame,np.ones(len(frame)))
        query=frame.iloc[:3].copy();query.CabinNum=[-10.,np.nan,999.];query.loc[query.index[-1],"Deck"]="unseen"
        encoded=encoder.transform(query)
        self.assertAlmostEqual(encoded.spatial_log_count_10.iloc[0],np.log1p(2))
        self.assertEqual(encoded.spatial_log_count_10.iloc[1],0.)
        self.assertEqual(encoded.spatial_log_count_10.iloc[2],0.)
        np.testing.assert_array_equal(encoded.spatial_mean_10,np.ones(3))

    def test_target_column_is_ignored_at_transform(self):
        frame=sample();encoder=SpatialTargetEncoder().fit(frame,np.arange(len(frame))%2)
        original=encoder.transform(frame);frame["Transported"]=np.arange(len(frame))
        pd.testing.assert_frame_equal(original,encoder.transform(frame))


if __name__=="__main__":unittest.main()
