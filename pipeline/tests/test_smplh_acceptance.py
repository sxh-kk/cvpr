import importlib.util
from pathlib import Path

spec=importlib.util.spec_from_file_location("accept_smplh_job",Path(__file__).parents[1]/"scripts/accept_smplh_job.py")
accept=importlib.util.module_from_spec(spec)
spec.loader.exec_module(accept)


def _decision(**changes):
    args=dict(policy={"body_observation_mean_max_m":.01,"body_observation_p95_max_m":.03,
                      "native_hand_to_fitted_wrist_p95_max_m":.01,
                      "continuous_local_step_over45deg_fraction_max":.001},
              audit={"passed":True,"episodes_checked":60,"frames_checked":154473},
              preservation={"passed":True},body={"count":100,"mean":.008,"p95":.025},
              wrists={"count":100,"mean":.002,"p95":.007},jump_count=1,pair_count=10000,
              expected_episodes=60,expected_frames=154473)
    args.update(changes)
    return accept.evaluate_gates(**args)


def test_good_residuals_do_not_override_failed_preservation_or_independent_audit():
    assert not _decision()[0]
    assert _decision(preservation={"passed":False})[0]
    assert _decision(audit={"passed":False,"episodes_checked":60,"frames_checked":154473})[0]


def test_partial_population_or_missing_supervision_cannot_open_full_processing_gate():
    assert _decision(expected_episodes=61)[0]
    assert _decision(expected_frames=154474)[0]
    assert _decision(wrists={"count":0,"mean":None,"p95":None})[0]
    assert _decision(pair_count=0,jump_count=0)[0]


def test_rotation_jump_and_wrist_failure_are_not_hidden_by_good_body_fit():
    assert _decision(jump_count=20,pair_count=10000)[0]
    assert _decision(wrists={"count":100,"mean":.002,"p95":.02})[0]
    assert _decision(body={"count":100,"mean":.015,"p95":.025})[0]
