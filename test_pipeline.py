import unittest

import joblib
import numpy as np

from data import HISTORY, RUN_DIR, TARGET, build_features, load_trips, split_trips
from inference import forecast, metrics


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.blocks = split_trips(load_trips())

    def test_chronological_split(self):
        self.assertEqual([len(x) for x in self.blocks.values()], [23100, 4616, 2798, 4616])
        blocks = list(self.blocks.values())
        for earlier, later in zip(blocks, blocks[1:]):
            self.assertLessEqual(earlier["load_available_at"].max(), later["forecast_at"].min())

    def test_future_reports_cannot_change_features(self):
        train = self.blocks["train"]
        first_day = train.loc[train["service_date"].eq(train["service_date"].min())].copy()
        prepared = build_features(first_day).set_index("trip_id")
        sample = prepared.loc[prepared[HISTORY].notna().any(axis=1)].iloc[0]

        later_reports = first_day["load_available_at"].gt(sample["forecast_at"])
        self.assertTrue(later_reports.any())
        first_day.loc[later_reports, TARGET] = 0
        changed = build_features(first_day).set_index("trip_id").loc[sample.name]

        self.assertTrue(sample[HISTORY].equals(changed[HISTORY]))
        self.assertLessEqual(sample["latest_history_release"], sample["forecast_at"])

    def test_saved_model_inference(self):
        bundle = joblib.load(RUN_DIR / "winner.joblib")
        sample = build_features(self.blocks["test"]).iloc[:10].drop(columns=TARGET)
        result = forecast(bundle, sample)

        self.assertTrue(np.isfinite(result["prediction"]).all())
        self.assertTrue(result["lower"].le(result["prediction"]).all())
        self.assertTrue(result["prediction"].le(result["upper"]).all())
        self.assertTrue(result["upper"].le(sample["total_capacity"]).all())

    def test_zero_total_wape(self):
        self.assertIsNone(metrics([0, 0], [1, 2])["wape_pct"])


if __name__ == "__main__":
    unittest.main()