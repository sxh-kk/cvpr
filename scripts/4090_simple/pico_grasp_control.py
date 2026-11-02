"""Measured-FK mode1 upper-body PICO input with SONIC standing leg references.

The exact same live-packet freshness checks apply. Joystick motion intent is
recorded but ignored for this tabletop task; no actor/root is fixed or edited.
The official planner still generates leg references and the decoder/physics
determine the actual motion. This is not AMO control equivalence.
"""
from pico_mobile_control import MobilePlannerControl

CONTRACT = 'cvpr-pico-grasp-standing-live-vr-mode1-v1'


class GraspStandingControl(MobilePlannerControl):
    def __init__(self, model_path, **kwargs):
        super().__init__(model_path, reference_policy='continuous_vr_mode1', **kwargs)
        self.contract = CONTRACT

    def _input_command(self, frame, now, tracking_valid):
        heading = self.yaw
        _, evidence = super()._input_command(frame, now, tracking_valid)
        self.yaw = heading
        evidence.update(applied_axes_left_x_y_right_x_y=[0., 0., 0., 0.],
            ignored_axes=['left_axis_x','left_axis_y','right_axis_x','right_axis_y'],
            ignored_reason=['tabletop_grasp_standing_reference'],
            axis_policy='grasp_standing_leg_reference_all_joystick_axes_ignored')
        return self._idle(), evidence
