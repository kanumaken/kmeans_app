"""
test_kmeans.py

This file contains unit tests for the K-Means clustering logic.
It verifies that the model creates the expected number of clusters.
"""

import unittest
import pandas as pd
from kmeans_model import run_kmeans


class TestKMeans(unittest.TestCase):
    """
    Test class for K-Means clustering functionality.
    """

    def test_cluster_count(self):
        """
        Test to ensure the correct number of clusters is created.
        """

        # Create a small sample dataset for testing
        sample_data = pd.DataFrame({
            'TEMP': [20, 22, 25, 10, 12],
            'WDSP': [5, 6, 7, 3, 2],
            'SLP': [1010, 1012, 1013, 1008, 1009]
        })

        # Run K-Means clustering with 2 clusters
        labels, model = run_kmeans(sample_data, k=2)

        # Assert that exactly 2 unique clusters are created
        self.assertEqual(len(set(labels)), 2)


# Allows the test file to be run directly
if __name__ == '__main__':
    unittest.main()
