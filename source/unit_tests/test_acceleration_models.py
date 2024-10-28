import unittest

from source.tools import acceleration_models


class TestAccelerationModels(unittest.TestCase):
    def test_coffin_manson_acceleration_factor(self):
        self.assertEqual(acceleration_models.coffin_manson_acceleration_factor(dt_acc=100, dt_use=100, cm_exp=2), 1)
        self.assertEqual(acceleration_models.coffin_manson_acceleration_factor(dt_acc=120, dt_use=100, cm_exp=2), 1.44)
        self.assertEqual(acceleration_models.coffin_manson_acceleration_factor(dt_acc=70, dt_use=100, cm_exp=2), 0.48999999999999994)

    def test_arrhenius_acceleration_factor(self):
        self.assertEqual(acceleration_models.arrhenius_acceleration_factor(t_acc=80, t_use=80, ea=.5), 1)
        self.assertEqual(acceleration_models.arrhenius_acceleration_factor(t_acc=100, t_use=80, ea=.5), 1)
        self.assertEqual(acceleration_models.arrhenius_acceleration_factor(t_acc=60, t_use=80, ea=.5), 1)


if __name__ == '__main__':
    unittest.main()