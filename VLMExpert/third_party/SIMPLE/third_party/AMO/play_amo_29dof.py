# -----------------------------------------------------------------------------
# Copyright [2025] [Jialong Li, Xuxin Cheng, Tianshu Huang, Xiaolong Wang]

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This script is based on an initial draft generously provided by Zixuan Chen.
# -----------------------------------------------------------------------------

import types
import numpy as np
import mujoco, mujoco_viewer
import glfw
from collections import deque
import torch

def quatToEuler(quat):
    eulerVec = np.zeros(3)
    qw = quat[0] 
    qx = quat[1] 
    qy = quat[2]
    qz = quat[3]
    # roll (x-axis rotation)
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    eulerVec[0] = np.arctan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2 * (qw * qy - qz * qx)
    if np.abs(sinp) >= 1:
        eulerVec[1] = np.copysign(np.pi / 2, sinp)  # use 90 degrees if out of range
    else:
        eulerVec[1] = np.arcsin(sinp)

    # yaw (z-axis rotation)
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    eulerVec[2] = np.arctan2(siny_cosp, cosy_cosp)
    
    return eulerVec

def _key_callback(self, window, key, scancode, action, mods):
    if action != glfw.PRESS:
        return
    if key == glfw.KEY_S:
        self.commands[0] -= 0.05
    elif key == glfw.KEY_W:
        self.commands[0] += 0.05
    elif key == glfw.KEY_A:
        self.commands[1] += 0.1
    elif key == glfw.KEY_D:
        self.commands[1] -= 0.1
    elif key == glfw.KEY_Q:
        self.commands[2] += 0.05
    elif key == glfw.KEY_E:
        self.commands[2] -= 0.05
    elif key == glfw.KEY_Z:
        self.commands[3] += 0.05
    elif key == glfw.KEY_X:
        self.commands[3] -= 0.05
    elif key == glfw.KEY_J:
        self.commands[4] += 0.1
    elif key == glfw.KEY_U:
        self.commands[4] -= 0.1
    elif key == glfw.KEY_K:
        self.commands[5] += 0.05
    elif key == glfw.KEY_I:
        self.commands[5] -= 0.05
    elif key == glfw.KEY_L:
        self.commands[6] += 0.05
    elif key == glfw.KEY_O:
        self.commands[6] -= 0.1
    elif key == glfw.KEY_T:
        self.commands[7] = not self.commands[7]
        if self.commands[7]:
            print("Toggled arm control ON")
        else:
            print("Toggled arm control OFF")
    elif key == glfw.KEY_ESCAPE:
        print("Pressed ESC")
        print("Quitting.")
        glfw.set_window_should_close(self.window, True)
        return
    print(
        f"vx: {self.commands[0]:<{8}.2f}"
        f"vy: {self.commands[2]:<{8}.2f}"
        f"yaw: {self.commands[1]:<{8}.2f}"
        f"height: {(0.75 + self.commands[3]):<{8}.2f}"
        f"torso yaw: {self.commands[4]:<{8}.2f}"
        f"torso pitch: {self.commands[5]:<{8}.2f}"
        f"torso roll: {self.commands[6]:<{8}.2f}"
    )

class HumanoidEnv:
    def __init__(self, policy_jit, robot_type="g1", device="cuda"):
        self.robot_type = robot_type
        self.device = device
        
        if robot_type == "g1":
            model_path = "/home/wenke/SIMPLE/third_party/AMO/g1_29dof.xml"
            self.stiffness = np.array([
                150, 150, 150, 300, 80, 20,
                150, 150, 150, 300, 80, 20,
                400, 400, 400,
                80, 80, 40, 60,40,40,40,
                80, 80, 40, 60,40,40,40,
            ])
            self.damping = np.array([
                2, 2, 2, 4, 2, 1,
                2, 2, 2, 4, 2, 1,
                15, 15, 15,
                2, 2, 1, 1,1,1,1,
                2, 2, 1, 1,1,1,1,
            ])
            #FIXME
            self.num_actions = 15
            self.num_dofs = 43

            # self.default_dof_pos = np.array([
            #     -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,#left_hip_pitch, left_hip_roll, left_hip_yaw, left_knee, left_ankle_pitch, left_ankle_roll
            #     -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,#right_hip_pitch, right_hip_roll, right_hip_yaw, right_knee, right_ankle_pitch, right_ankle_roll
               
            #     0.0, 0.0, 0.0,#waist_yaw, waist_roll, waist_pitch
            #     0.0, 0.0, 0.0, 0.0,0.0,0.0,0.0,#left_shoulder_pitch, left_shoulder_roll, left_shoulder_yaw, left_elbow, left_wrist_roll, left_wrist_pitch, left_wrist_yaw
            #     0.0, 0.0, 0.0, 0.0,0.0,0.0,0.0,#right_shoulder_pitch, right_shoulder_roll, right_shoulder_yaw, right_elbow, right_wrist_roll, right_wrist_pitch, right_wrist_yaw
            #     0.0, 0.0, 0.0, 0.0,0.0,0.0,0.0,#left_hand_thumb_0_joint, left_hand_thumb_1_joint, left_hand_thumb_2_joint, left_hand_index_0_joint, left_hand_index_1_joint, left_hand_middle_0_joint, left_hand_middle_1_joint
            #     0.0, 0.0, 0.0, 0.0,0.0,0.0,0.0,#right_hand_thumb_0_joint, right_hand_thumb_1_joint, right_hand_thumb_2_joint, right_hand_index_0_joint, right_hand_index_1_joint, right_hand_middle_0_joint, right_hand_middle_1_joint
            # ])

            self.default_dof_pos = np.array([
                -0, 0.0, 0.0, 0.0, -0.0, 0.0,#left_hip_pitch, left_hip_roll, left_hip_yaw, left_knee, left_ankle_pitch, left_ankle_roll
                -0.0, 0.0, 0.0, 0.0, -0.0, 0.0,#right_hip_pitch, right_hip_roll, right_hip_yaw, right_knee, right_ankle_pitch, right_ankle_roll
               
                0.0, 0.0, 0.0,#waist_yaw, waist_roll, waist_pitch
                0.0, 0.0, 0.0, 0.0,0.0,0.0,0.0,#left_shoulder_pitch, left_shoulder_roll, left_shoulder_yaw, left_elbow, left_wrist_roll, left_wrist_pitch, left_wrist_yaw
                0.0, 0.0, 0.0, 0.0,0.0,0.0,0.0,#right_shoulder_pitch, right_shoulder_roll, right_shoulder_yaw, right_elbow, right_wrist_roll, right_wrist_pitch, right_wrist_yaw
                0.0, 0.0, 0.0, 0.0,0.0,0.0,0.0,#left_hand_thumb_0_joint, left_hand_thumb_1_joint, left_hand_thumb_2_joint, left_hand_index_0_joint, left_hand_index_1_joint, left_hand_middle_0_joint, left_hand_middle_1_joint
                0.0, 0.0, 0.0, 0.0,0.0,0.0,0.0,#right_hand_thumb_0_joint, right_hand_thumb_1_joint, right_hand_thumb_2_joint, right_hand_index_0_joint, right_hand_index_1_joint, right_hand_middle_0_joint, right_hand_middle_1_joint
            ])

            self.default_dof_pos_for_policy = np.concatenate([
                self.default_dof_pos[:15],
                self.default_dof_pos[15:19],
                self.default_dof_pos[22:26]
            ])
            self.torque_limits = np.array([
                88, 139, 88, 139, 50, 50,
                88, 139, 88, 139, 50, 50,
                88, 50, 50,
                25, 25, 25, 25,25,25,25,
                25, 25, 25, 25,25,25,25,
            ])


        else:
            raise ValueError(f"Robot type {robot_type} not supported!")
        
        self.obs_indices = np.arange(self.num_dofs)
        
        self.sim_duration = 100 * 20.0
        self.sim_dt = 0.002
        self.sim_decimation = 10
        self.control_dt = self.sim_dt * self.sim_decimation
        
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.model.opt.timestep = 0.002
        self.model.opt.gravity= (0,0,-9.81)
        self.model.opt.impratio = 10
        self.model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        self.model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        self.model.opt.noslip_iterations = 2



        data = mujoco.MjData(self.model)
        
        # gravity_link_names = ["pelvis",
        #                       "left_hip_pitch_link", 
        #                       "left_hip_roll_link", 
        #                       "left_hip_yaw_link",
        #                        "left_knee_link",
        #                        "left_ankle_pitch_link",
        #                        "left_ankle_roll_link",
        #                        "pelvis_contour_link",
        #                        "right_hip_pitch_link",
        #                        "right_hip_roll_link",
        #                        "right_hip_yaw_link",
        #                        "right_knee_link",
        #                        "right_ankle_pitch_link",
        #                        "right_ankle_roll_link",

        #                        "waist_yaw_link",
        #                        "waist_roll_link",
        #                        "torso_link",
        #                       ]
        # for link_name in gravity_link_names:
        #     data.body(link_name).xfrc_applied = np.array([0, 0, -98.1,0,0,0])
                            
        self.data = data
        mujoco.mj_forward(self.model, self.data)
        
        # mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_step(self.model, self.data)
        self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
        self.viewer.commands = np.zeros(8, dtype=np.float32)
        self.viewer.commands[0] = 0.0
        self.viewer.commands[1] = 1.57
        self.viewer.cam.distance = 2.5 # 5.0
        self.viewer.cam.elevation = 0.0
        self.viewer._key_callback = types.MethodType(_key_callback, self.viewer)
        glfw.set_key_callback(self.viewer.window, self.viewer._key_callback)
        
        # self.last_action = np.zeros(self.num_actions, dtype=np.float32)
        self.action_scale = 0.25


        # self.arm_action = self.default_dof_pos[15:19]+self.default_dof_pos[22:26]
        self.arm_action = np.concatenate([
                                self.default_dof_pos[15:19],
                                self.default_dof_pos[22:26]
                            ])
        # self.prev_arm_action = self.default_dof_pos[15:]
        self.prev_arm_action = np.concatenate([
                                self.default_dof_pos[15:19],
                                self.default_dof_pos[22:26]
                            ])
        self.arm_blend = 0.0
        self.toggle_arm = False

        self.scales_ang_vel = 0.25
        self.scales_dof_vel = 0.05
        
        #TODO
        self.nj = 43
        self.n_priv = 3
        self.n_proprio = 3 + 2 + 2 + 23 * 3 + 2 + 15
        self.history_len = 10
        self.extra_history_len = 25
        self._n_demo_dof = 8

        # not used
        # self.dof_pos = np.zeros(self.nj, dtype=np.float32)
        # self.dof_vel = np.zeros(self.nj, dtype=np.float32)
        
        self.quat = np.zeros(4, dtype=np.float32)
        self.ang_vel = np.zeros(3, dtype=np.float32)
        self.last_action = np.zeros(self.nj)

        self.last_action_for_policy = np.concatenate([self.last_action[:15], self.last_action[15:19], self.last_action[22:26]])

        self.demo_obs_template = np.zeros((8 + 3 + 3 + 3, ))
        self.demo_obs_template[:self._n_demo_dof] = np.concatenate([
                                self.default_dof_pos[15:19],
                                self.default_dof_pos[22:26]
                            ])
        self.demo_obs_template[self._n_demo_dof+6:self._n_demo_dof+9] = 0.75

        self.target_yaw = 0.0 

        self._in_place_stand_flag = True
        self.gait_cycle = np.array([0.25, 0.25])
        self.gait_freq = 1.3

        self.proprio_history_buf = deque(maxlen=self.history_len)
        self.extra_history_buf = deque(maxlen=self.extra_history_len)
        for i in range(self.history_len):
            self.proprio_history_buf.append(np.zeros(self.n_proprio))
        for i in range(self.extra_history_len):
            self.extra_history_buf.append(np.zeros(self.n_proprio))
    
        self.policy_jit = policy_jit

        self.adapter = torch.jit.load("/home/wenke/SIMPLE/third_party/AMO/adapter_jit.pt", map_location=self.device)
        self.adapter.eval()
        for param in self.adapter.parameters():
            param.requires_grad = False
        
        norm_stats = torch.load("/home/wenke/SIMPLE/third_party/AMO/adapter_norm_stats.pt", weights_only=False)
        self.input_mean = torch.tensor(norm_stats['input_mean'], device=self.device, dtype=torch.float32)
        self.input_std = torch.tensor(norm_stats['input_std'], device=self.device, dtype=torch.float32)
        self.output_mean = torch.tensor(norm_stats['output_mean'], device=self.device, dtype=torch.float32)
        self.output_std = torch.tensor(norm_stats['output_std'], device=self.device, dtype=torch.float32)

        self.adapter_input = torch.zeros((1, 8 + 4), device=self.device, dtype=torch.float32)
        self.adapter_output = torch.zeros((1, 15), device=self.device, dtype=torch.float32)

    def extract_data(self):
        self.dof_pos = self.data.qpos.astype(np.float32)[-self.num_dofs:]
        self.dof_vel = self.data.qvel.astype(np.float32)[-self.num_dofs:]
        #for amo policy input
        self.dof_pos = np.concatenate([self.dof_pos[:15], self.dof_pos[15:19], self.dof_pos[22:26]]) 
        self.dof_vel = np.concatenate([self.dof_vel[:15], self.dof_vel[15:19], self.dof_vel[22:26]]) 
        self.quat = self.data.sensor('orientation').data.astype(np.float32)
        self.ang_vel = self.data.sensor('angular-velocity').data.astype(np.float32)
        
    def get_observation(self):
        rpy = quatToEuler(self.quat)

        self.target_yaw = self.viewer.commands[1]
        dyaw = rpy[2] - self.target_yaw
        dyaw = np.remainder(dyaw + np.pi, 2 * np.pi) - np.pi
        if self._in_place_stand_flag:
            dyaw = 0.0

        obs_dof_vel = self.dof_vel.copy()
        obs_dof_vel[[4, 5, 10, 11, 13, 14]] = 0.0

        gait_obs = np.sin(self.gait_cycle * 2 * np.pi)

        self.adapter_input = np.concatenate([np.zeros(4), self.dof_pos[15:]])

        self.adapter_input[0] = 0.75 + self.viewer.commands[3]
        self.adapter_input[1] = self.viewer.commands[4]
        self.adapter_input[2] = self.viewer.commands[5]
        self.adapter_input[3] = self.viewer.commands[6]

        self.adapter_input = torch.tensor(self.adapter_input).to(self.device, dtype=torch.float32).unsqueeze(0)
            
        self.adapter_input = (self.adapter_input - self.input_mean) / (self.input_std + 1e-8)
        self.adapter_output = self.adapter(self.adapter_input.view(1, -1))
        self.adapter_output = self.adapter_output * self.output_std + self.output_mean

        obs_prop = np.concatenate([
                    self.ang_vel * self.scales_ang_vel,
                    rpy[:2],
                    (np.sin(dyaw),
                    np.cos(dyaw)),
                    (self.dof_pos - self.default_dof_pos_for_policy),
                    self.dof_vel * self.scales_dof_vel,
                    self.last_action_for_policy,
                    gait_obs,
                    self.adapter_output.cpu().numpy().squeeze(),
        ])

        obs_priv = np.zeros((self.n_priv, ))
        obs_hist = np.array(self.proprio_history_buf).flatten()

        obs_demo = self.demo_obs_template.copy()
        obs_demo[:self._n_demo_dof] = self.dof_pos[15:]
        obs_demo[self._n_demo_dof] = self.viewer.commands[0]
        obs_demo[self._n_demo_dof+1] = self.viewer.commands[2]
        self._in_place_stand_flag = np.abs(self.viewer.commands[0]) < 0.1
        obs_demo[self._n_demo_dof+3] = self.viewer.commands[4]
        obs_demo[self._n_demo_dof+4] = self.viewer.commands[5]
        obs_demo[self._n_demo_dof+5] = self.viewer.commands[6]
        obs_demo[self._n_demo_dof+6:self._n_demo_dof+9] = 0.75 + self.viewer.commands[3]

        self.proprio_history_buf.append(obs_prop)
        self.extra_history_buf.append(obs_prop)
        
        return np.concatenate((obs_prop, obs_demo, obs_priv, obs_hist))
        
    def run(self):        
        for i in range(int(self.sim_duration / self.sim_dt)):
            self.extract_data()
            self.sim_decimation=1
            if i % self.sim_decimation == 0:
                obs = self.get_observation()
                
                obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    extra_hist = torch.tensor(np.array(self.extra_history_buf).flatten().copy(), dtype=torch.float).view(1, -1).to(self.device)
                    raw_action = self.policy_jit(obs_tensor, extra_hist).cpu().numpy().squeeze()
                
                raw_action = np.clip(raw_action, -40., 40.)
                self.last_action_for_policy = np.concatenate([raw_action.copy(), (self.dof_pos - self.default_dof_pos_for_policy)[15:] / self.action_scale])
                scaled_actions = raw_action * self.action_scale
                
            

                pd_target = scaled_actions + self.default_dof_pos_for_policy[:15]
             
                

                self.gait_cycle = np.remainder(self.gait_cycle + self.control_dt * self.gait_freq, 1.0)
                if self._in_place_stand_flag and ((np.abs(self.gait_cycle[0] - 0.25) < 0.05) or (np.abs(self.gait_cycle[1] - 0.25) < 0.05)):
                    self.gait_cycle = np.array([0.25, 0.25])
                if (not self._in_place_stand_flag) and ((np.abs(self.gait_cycle[0] - 0.25) < 0.05) and (np.abs(self.gait_cycle[1] - 0.25) < 0.05)):
                    self.gait_cycle = np.array([0.25, 0.75])
                
                # self.viewer.cam.azimuth += 0.1
                self.viewer.cam.lookat = self.data.qpos.astype(np.float32)[:3]
                self.viewer.render()
            pd_target[12:15] = 0
            torque = (pd_target[:15] - self.dof_pos[:15]) * self.stiffness[:15] - self.dof_vel[:15] * self.damping[:15]
            torque = np.clip(torque, -self.torque_limits[:15], self.torque_limits[:15])
            
            self.data.ctrl[:15] = torque[:15]
      

                       
            mujoco.mj_step(self.model, self.data,nstep=1)

            for i in range(10):
                self.extract_data()
                torque = (pd_target[:15] - self.dof_pos[:15]) * self.stiffness[:15] - self.dof_vel[:15] * self.damping[:15]
                torque = np.clip(torque, -self.torque_limits[:15], self.torque_limits[:15])
                
                self.data.ctrl[:15] = torque[:15]
                mujoco.mj_step(self.model, self.data,nstep=1)
        
        self.viewer.close()

if __name__ == "__main__":

    robot = "g1"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    policy_pth = '/home/wenke/SIMPLE/third_party/AMO/amo_jit.pt'
    
    policy_jit = torch.jit.load(policy_pth, map_location=device)
    
    env = HumanoidEnv(policy_jit=policy_jit, robot_type=robot, device=device)
    
    env.run()
        
        
