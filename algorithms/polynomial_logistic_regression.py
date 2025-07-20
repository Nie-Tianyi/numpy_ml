import unittest

from test_data_set.mnist import mnist


class Unittest(unittest.TestCase):
    def test_mnist(self):
        data = mnist()
        self.assertEqual(1 + 1, 2)

if __name__ == "__main__":
    unittest.main()
