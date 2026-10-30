from dataclasses import asdict
import json

import numpy as np
import pytest

from taucatch.action_h96 import SPEC as SPEC96, CONTRACT as CONTRACT96
from taucatch.action_h192_195536 import SPEC, CONTRACT, require_checkpoint, require_spec
from taucatch.action_h192_data_195536 import exact_teacher_grid


def test_changing_timestep_cannot_promote_an_untrained_checkpoint():
    with pytest.raises(ValueError, match='12x16ms'):
        require_spec(asdict(SPEC96))
    require_spec(asdict(SPEC))
    retimed = dict(action_spec=asdict(SPEC), time_contract=CONTRACT96, dataset={})
    with pytest.raises(ValueError, match='provenance'):
        require_checkpoint(retimed)
    retimed['time_contract'] = CONTRACT
    with pytest.raises(ValueError, match='dataset'):
        require_checkpoint(retimed)


@pytest.mark.parametrize('split', ['train', 'val'])
def test_grid_reconstruction_checks_overlapping_physical_timestamps(tmp_path, split):
    source, labels = tmp_path/'source.npz', tmp_path/'labels.npz'
    observed = np.array([.008, .016])
    actions = np.array([[[t+.008*(j+1)]*7 for j in range(12)] for t in observed], np.float32)
    valid = np.ones((2,12), bool)
    np.savez(source, split=split, observed_t=observed)
    np.savez(labels, action_spec=json.dumps(asdict(SPEC96)), observed_t=observed, actions=actions, valid=valid)
    refs = exact_teacher_grid(source, labels)
    np.testing.assert_array_equal(refs[24], actions[0,1])
    assert len(refs)==13
    actions[1,0,0] += .01  # Same absolute24ms label must not silently disagree.
    np.savez(labels, action_spec=json.dumps(asdict(SPEC96)), observed_t=observed, actions=actions, valid=valid)
    with pytest.raises(AssertionError, match='Conflicting'):
        exact_teacher_grid(source, labels)
