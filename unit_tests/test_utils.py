import unittest

import numpy as np

from source.tools import utils

class TestUtils(unittest.TestCase):
    def test_get_range_array(self):
        self.assertEqual(
            utils.get_range_array(.9, 'confidence' )[0],0.8)
        self.assertEqual(
            len(utils.get_range_array(.9, 'confidence')),100)
        self.assertEqual(
            utils.get_range_array(.99, 'r_ts', levels=50)[0],0.97)
        self.assertEqual(
            len(utils.get_range_array(.99, 'r_ts', levels=50)),50)
        self.assertEqual(
            utils.get_range_array(.8, levels=50)[0],0.4)


    def test_seconds_to_minutes(self):
        self.assertEqual(utils.seconds_to_minutes(60), 1)

    def test_seconds_to_hours(self):
        self.assertEqual(utils.seconds_to_hours(60*60), 1)

    def test_minutes_to_hours(self):
        self.assertEqual(utils.minutes_to_hours(60), 1)

    def test_minutes_to_seconds(self):
        self.assertEqual(utils.minutes_to_seconds(1), 60)

    def test_hours_to_minutes(self):
        self.assertEqual(utils.hours_to_minutes(1), 60)

    def test_hours_to_seconds(self):
        self.assertEqual(utils.hours_to_seconds(1), 60*60)