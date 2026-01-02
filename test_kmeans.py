import unittest
import pandas as pd
from kmeans_model import run_kmeans

class TestKMeans(unittest.TestCase):

    def test_cluster_count(self):
        sample_data = pd.DataFrame({
            'TEMP': [20, 22, 25, 10, 12],
            'WDSP': [5, 6, 7, 3, 2],
            'SLP': [1010, 1012, 1013, 1008, 1009]
        })

        labels, model = run_kmeans(sample_data, k=2)
        self.assertEqual(len(set(labels)), 2)

if __name__ == '__main__':
    unittest.main()
