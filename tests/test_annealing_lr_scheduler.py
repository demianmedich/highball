import unittest

from highball.optim.scheduler import AnnealingLrSchedulerConfig


class AnnealingLrSchedulerTestCase(unittest.TestCase):
    def test_cosine_decay(self):
        cfg = AnnealingLrSchedulerConfig(max_lr=1e-3,
                                         min_lr=1e-5,
                                         warmup_steps=100,
                                         decay_steps=4000,
                                         decay_style='cosine')

        class DummyOptimizer:
            param_groups = []

        dummy_optimizer = DummyOptimizer()
        scheduler = cfg.instantiate(dummy_optimizer)

        for i in range(5000):
            lr = scheduler.get_lr()
            scheduler.step(1)
            print(f'step {i} lr {lr}')


if __name__ == '__main__':
    unittest.main()
