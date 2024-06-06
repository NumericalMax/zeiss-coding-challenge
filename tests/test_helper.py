import unittest
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from io import StringIO
from contextlib import redirect_stdout
from src.helper import Helper


class TestHelper(unittest.TestCase):

    # TODO: Add more test-cases
    def setUp(self):
        self.df = pd.DataFrame({
            'A': [1, 2, 2, 3, 4, 5, 6, 6, 7, 8],
            'B': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'],
            'C': [1, 1, 2, 2, 2, 2, 2, 2, 2, 2]
        })

    def test_print_unique_column_entries(self):
        with StringIO() as buf, redirect_stdout(buf):
            Helper.print_unique_column_entries(self.df, threshold=3)
            output = buf.getvalue()
        print(output)
        expected_output = ("Column 'A' of type 'int64': Has more than 3 unique entries\n"
                           "Column 'B' of type 'object': Has more than 3 unique entries\n"
                           "Column 'C' of type 'int64': [1 2]\n")
        self.assertIn(expected_output, output)


    def test_print_distinct_values(self):
        values = [24, 24, 12, 617.6, 628.9, 12, 606.6, 315.9, 217.6, 318.8, 640.7]
        power = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        threshold = 100
        result = Helper.print_distinct_values(values, power, threshold)
        self.assertEqual(result, [(24, 1), (618, 4), (316.0, 8)])


    def test_fill_nan_from_seasonality(self):
        series = pd.Series([1, np.nan, 2, np.nan, 1, 2, np.nan, 2, 1, 1, 1, 1, 1, 10])
        distribution = GaussianMixture(n_components=1, random_state=42)
        distribution.fit(np.array([1, 2, 1, 2, 1, 1, 1, 1, 10]).reshape(-1, 1))
        result = Helper.fill_nan_from_seasonality(series, window=2, distribution=distribution, threshold_quantile=0.05)
        expected_result = pd.Series([1, np.nan, 2, np.nan, 1, 2, 1, 2, 1, 1, 1, 1, 1, 10])
        pd.testing.assert_series_equal(result, expected_result, check_exact=False)
    

    def test_fill_nan_from_distribution(self):
        series = pd.Series([1, np.nan, 3, np.nan, 5, 6, np.nan, 8, 9, 10])
        distribution = GaussianMixture(n_components=1, random_state=42)
        distribution.fit(np.array([2, 2, 2, 2, 2, 2, 2]).reshape(-1, 1))
        result = Helper.fill_nan_from_distribution(series, distribution)
        expected_result = pd.Series([1, 2, 3, 2, 5, 6, 2, 8, 9, 10])
        pd.testing.assert_series_equal(result.astype(int), expected_result, check_exact=False)

    def test_calculate_difference(self):
        df = pd.DataFrame({
            'property_name': ['Heating', 'Cooling', 'Heating', 'Cooling'],
            'temperature': [100, 80, 90, 85],
            'datetime': ['2021-01-01', '2021-01-01', '2021-01-02', '2021-01-02']
        })
        result = Helper.calculate_difference_timeseries(df)
        expected = pd.DataFrame({
            'datetime': ['2021-01-01', '2021-01-02'],
            'temperature_difference': [20, 5],
            'cooling_temperature': [100, 90],
            'heating_temperature': [80, 85]
        })
        pd.testing.assert_frame_equal(result, expected)


if __name__ == '__main__':
    unittest.main()
