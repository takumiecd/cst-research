"""One next-token shift and split-boundary checks, without network access."""

import unittest


class FFNBatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ImportError as error:
            raise unittest.SkipTest('optional Torch required') from error

    def test_next_token_targets_are_shifted_once_and_never_wrap(self):
        import numpy as np
        from cst_research.experiments.ffn_run import token_batch
        x,y = token_batch(np.arange(10,dtype=np.uint32), [0,4], 4, 'cpu')
        self.assertEqual(x.tolist(), [[0,1,2,3],[4,5,6,7]])
        self.assertEqual(y.tolist(), [[1,2,3,4],[5,6,7,8]])
        with self.assertRaises(ValueError):
            token_batch(np.arange(10,dtype=np.uint32), [7], 4, 'cpu')
