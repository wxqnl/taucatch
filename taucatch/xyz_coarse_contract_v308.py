"""Pure coarse XYZ ABI; importing it never imports simulation or CUDA."""
XYZ_CONTRACT = dict(
    name='coarse_fast_relative_xyz_only_6cm_rate08_v308', horizon=12, dt=.016,
    action_dim=3, translation_m=.06, translation_rate_mps=.8,
    normal='fast_unchanged', closure='fast_timer_unchanged',
    expiry='observation_plus_192ms_then_slew_to_zero')
