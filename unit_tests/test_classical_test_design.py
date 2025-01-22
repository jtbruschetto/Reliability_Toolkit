import unittest

from reliability_toolkit.tools.classical_test_design import calculate_reliability

class TestCalculateReliability(unittest.TestCase):

    def test_no_failures(self):
        # Test with no failures
        reliability = calculate_reliability(confidence=0.9, beta=1.0, failures=0, sample_size=20, life_ratio=1.0)
        self.assertAlmostEqual(reliability, 0.89125093, places=6)  # Expected value for these inputs

    def test_some_failures(self):
        # Test with some failures
        reliability = calculate_reliability(confidence=0.95, beta=2.0, failures=2, sample_size=50, life_ratio=0.5)
        self.assertAlmostEqual(reliability, 0.604312706, places=6)  # Expected value for these inputs

    def test_high_confidence(self):
        # Test with high confidence level
        reliability = calculate_reliability(confidence=0.99, beta=1.5, failures=1, sample_size=20, life_ratio=2.0)
        self.assertAlmostEqual(reliability, 0.88927336, places=6)  # Expected value for these inputs

    def test_low_life_ratio(self):
        # Test with a low life ratio
        reliability = calculate_reliability(confidence=0.9, beta=0.8, failures=5, sample_size=100, life_ratio=0.1)
        self.assertAlmostEqual(reliability, 0.55699870, places=6)  # Expected value for these inputs

    def test_edge_case_zero_beta(self):
        # Test with beta = 0 (should not result in an error)
        reliability = calculate_reliability(confidence=0.95, beta=0.0, failures=1, sample_size=10, life_ratio=1.0)
        self.assertAlmostEqual(reliability, 0.62226672, places=6)

    def test_kwargs_ignored(self):
        # Test that extra keyword arguments are ignored
        reliability1 = calculate_reliability(confidence=0.9, beta=1.0, failures=0, sample_size=10, life_ratio=1.0)
        reliability2 = calculate_reliability(confidence=0.9, beta=1.0, failures=0, sample_size=10, life_ratio=1.0, extra_arg=123)
        self.assertEqual(reliability1, reliability2)