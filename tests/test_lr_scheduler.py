import unittest

from torch import nn
from torch.optim import SGD

from highball.optim.scheduler import get_linear_schedule_with_warmup, \
    get_cosine_schedule_with_warmup


class LrSchedulerTestCase(unittest.TestCase):

    def setUp(self) -> None:
        class _Dummy(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(5, 3)

        self.dummy_model = _Dummy()

    def test_linear_schedule_with_warmup(self):
        lr = 0.1
        warmup_steps = 10
        training_steps = 100
        optimizer = SGD(self.dummy_model.parameters(), lr)
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, training_steps)

        for current_step in range(1, training_steps):

            scheduler.step()
            params_lr = optimizer.param_groups[0]['lr']
            print(params_lr)

            if current_step < warmup_steps:
                self.assertEqual(True, params_lr < lr)
            elif current_step == warmup_steps:
                self.assertEqual(lr, params_lr)
            else:
                self.assertEqual(True, params_lr < lr)

    def test_cosine_schedule_with_warmup(self):
        lr = 0.1
        warmup_steps = 10
        training_steps = 100

        optimizer = SGD(self.dummy_model.parameters(), lr)
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, training_steps)

        for current_step in range(1, training_steps):

            scheduler.step()
            params_lr = optimizer.param_groups[0]['lr']
            print(params_lr)

            if current_step < warmup_steps:
                self.assertEqual(True, params_lr < lr)
            elif current_step == warmup_steps:
                self.assertEqual(lr, params_lr)
            else:
                self.assertEqual(True, params_lr < lr)


if __name__ == '__main__':
    unittest.main()
