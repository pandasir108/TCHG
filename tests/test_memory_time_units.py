from copy import deepcopy
import math

import pytest
import torch

from tchg.models.masked_memory import ComponentDecoupledMaskedTrustMemory


@pytest.mark.parametrize('days', [1., 7., 30.])
def test_elapsed_days_preserve_distinct_memory_decay(days):
    memory = ComponentDecoupledMaskedTrustMemory(4, 4, memory_key='src_node',
                                                time_scale_seconds=86400.)
    z = torch.zeros(1, 4)
    state = memory._ensure_state(2, z)
    for component in ('E', 'B', 'C'):
        state['M_' + component].fill_(1.)
    output = memory(2, torch.tensor([0]), torch.tensor([1]), torch.ones(1),
                    torch.tensor([days * 86400.]), z, z, z,
                    timestamp=torch.ones(1), update_mask=torch.zeros(1))
    for component, rate in zip(('E', 'B', 'C'), (.015, .045, .09)):
        torch.testing.assert_close(output['M_' + component], torch.full_like(z, math.exp(-rate * days)))
    assert output['M_E'][0, 0] > output['M_B'][0, 0] > output['M_C'][0, 0] > 0.


def test_seconds_and_days_are_equivalent_for_reads_writes_and_gradients():
    torch.manual_seed(23)
    second_input = ComponentDecoupledMaskedTrustMemory(4, 4, memory_key='src_node',
                                                      time_scale_seconds=86400.)
    day_input = deepcopy(second_input)
    day_input.time_scale_seconds = 1.
    z = torch.randn(3, 4)
    arguments = [2, torch.zeros(3, dtype=torch.long), torch.ones(3, dtype=torch.long),
                 torch.tensor([0., 1., 1.])]
    kwargs = {'timestamp': torch.arange(3.), 'update_mask': torch.ones(3)}
    seconds = second_input(*arguments, torch.tensor([86400., 172800., 259200.]), z, z, z, **kwargs)
    days = day_input(*arguments, torch.tensor([1., 2., 3.]), z, z, z, **kwargs)
    for key in seconds:
        torch.testing.assert_close(seconds[key], days[key])
    seconds['memory'].sum().backward()
    days['memory'].sum().backward()
    for (name, value), (_, other) in zip(second_input.named_parameters(), day_input.named_parameters()):
        if value.grad is not None:
            torch.testing.assert_close(value.grad, other.grad)
    for component in ('E', 'B', 'C'):
        assert second_input.cells[component].gru.weight_ih.grad.abs().sum() > 0.
