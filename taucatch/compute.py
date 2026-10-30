"""User-authorized physical GPU assignments for this project's independent jobs."""
import os

AUTHORIZED_GPUS=(0,5,6)


def selected_gpu():
    value=os.environ.get('CUDA_VISIBLE_DEVICES','')
    if value not in tuple(map(str,AUTHORIZED_GPUS)):
        raise ValueError('Set CUDA_VISIBLE_DEVICES to one authorized GPU: 0, 5 or 6')
    return int(value)
