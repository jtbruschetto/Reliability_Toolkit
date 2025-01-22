import unittest

from reliability_toolkit.tools.acceleration_models import CoffinManson, Arrhenius


class TestAccelerationModels(unittest.TestCase):
    def test_coffin_manson_acceleration_factor(self):
        cm = CoffinManson(cm_exp=2)
        self.assertEqual(cm.acceleration_factor(dt_acc=100, dt_use=100), 1)
        self.assertEqual(cm.acceleration_factor(dt_acc=120, dt_use=100), 1.44)
        self.assertEqual(cm.acceleration_factor(dt_acc=70, dt_use=100), 0.48999999999999994)

    def test_arrhenius_acceleration_factor(self):
        arr = Arrhenius(ea=.5)
        self.assertEqual(arr.acceleration_factor(t_acc=80, t_use=80), 1)
        self.assertEqual(arr.acceleration_factor(t_acc=100, t_use=80), 2.412457081817392)
        self.assertEqual(arr.acceleration_factor(t_acc=60, t_use=80), 0.37292371023584664)


if __name__ == '__main__':
    unittest.main()