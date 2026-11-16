import torch
import pytest

from tchg.training.losses import trust_calibration_loss
from tchg.training.trainer import Trainer

from test_model import make_batch, make_tiny_model


def test_one_training_step() -> None:
    model = make_tiny_model()
    batch = make_batch()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    prediction = model(batch)
    loss, _ = trust_calibration_loss(prediction, batch, {}, {})
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    assert torch.isfinite(loss)


def test_ema_validation_weights_preserve_optimizer_weights_even_on_failure():
    trainer=Trainer.__new__(Trainer);trainer.cfg={'training':{'ema_decay':.5}}
    trainer.model=torch.nn.Linear(1,1,bias=False)
    with torch.no_grad():trainer.model.weight.zero_()
    trainer._start_ema()
    with torch.no_grad():trainer.model.weight.fill_(2.)
    trainer._update_ema()
    with trainer._validation_weights():
        torch.testing.assert_close(trainer.model.weight,torch.ones(1,1))
    torch.testing.assert_close(trainer.model.weight,torch.full((1,1),2.))
    with torch.no_grad():trainer.model.weight.fill_(4.)
    trainer._update_ema()
    with pytest.raises(RuntimeError,match='validation failed'):
        with trainer._validation_weights():
            torch.testing.assert_close(trainer.model.weight,torch.full((1,1),2.5))
            raise RuntimeError('validation failed')
    torch.testing.assert_close(trainer.model.weight,torch.full((1,1),4.))
