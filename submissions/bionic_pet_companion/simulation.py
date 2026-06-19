"""
仿生四足 + 人形宠物陪护机器人家庭互动仿真
==============================================
双机器人共生交互：机器狗（四足仿生）跟随人形陪护机器人，
共同完成递物、躲避障碍物、抚摸交互，柔性毛绒碰撞模型。

物理亮点：
  - 柔性沙发形变碰撞（composite particle）
  - 玩具滚动摩擦
  - 机器狗爪子柔性接触
  - 全身触觉传感器，触碰后自动跟随运动
"""

import mujoco
import numpy as np
import cv2
import time
import os
import sys
from pathlib import Path
from collections import deque


# ============================================================
# 控制器基类
# ============================================================

class Controller:
    """机器人控制器基类"""

    def __init__(self, model, data, name):
        self.model = model
        self.data = data
        self.name = name
        self.actuator_ids = []
        self._init_actuators()

    def _init_actuators(self):
        """初始化执行器 ID 列表"""
        raise NotImplementedError

    def set_control(self, targets):
        """设置控制目标"""
        for act_id, target in zip(self.actuator_ids, targets):
            self.data.ctrl[act_id] = target

    def get_joint_positions(self, joint_names):
        """获取关节位置"""
        return np.array([self.data.joint(name).qpos[0] for name in joint_names])


# ============================================================
# 机器狗控制器 (Unitree Go1)
# ============================================================

class DogController(Controller):
    """Unitree Go1 四足机器狗控制器 - 对角小跑步态"""

    def __init__(self, model, data):
        super().__init__(model, data, "robot_dog")

        # 关节名
        self.joint_names = [
            "dog_fl_hip_x", "dog_fl_hip_z", "dog_fl_knee",
            "dog_fr_hip_x", "dog_fr_hip_z", "dog_fr_knee",
            "dog_rl_hip_x", "dog_rl_hip_z", "dog_rl_knee",
            "dog_rr_hip_x", "dog_rr_hip_z", "dog_rr_knee",
        ]

        # 默认站立姿态
        self.stand_pose = np.array([
            # FL: hip_x, hip_z, knee
            0.0, -5.0, 30.0,
            # FR: hip_x, hip_z, knee
            0.0, 5.0, 30.0,
            # RL: hip_x, hip_z, knee
            0.0, -5.0, 30.0,
            # RR: hip_x, hip_z, knee
            0.0, 5.0, 30.0,
        ]) * np.pi / 180.0

        # 步态参数
        self.gait_phase = 0.0
        self.gait_frequency = 2.5  # Hz
        self.step_height = 0.06    # 抬腿高度 (m)
        self.step_length = 0.08    # 步幅 (rad)

        # 触摸传感器
        self.touch_sensor_names = [
            "dog_fl_touch_sensor", "dog_fr_touch_sensor",
            "dog_rl_touch_sensor", "dog_rr_touch_sensor",
            "dog_body_touch_sensor", "dog_head_touch_sensor",
        ]

        # 行为状态
        self.mode = "idle"  # idle, walking, touched, playing
        self.touch_cooldown = 0
        self.wag_phase = 0.0
        self.turn_target = 0.0

    def _init_actuators(self):
        act_names = [
            "dog_fl_hip_x_act", "dog_fl_hip_z_act", "dog_fl_knee_act",
            "dog_fr_hip_x_act", "dog_fr_hip_z_act", "dog_fr_knee_act",
            "dog_rl_hip_x_act", "dog_rl_hip_z_act", "dog_rl_knee_act",
            "dog_rr_hip_x_act", "dog_rr_hip_z_act", "dog_rr_knee_act",
        ]
        self.actuator_ids = [self.model.actuator(name).id for name in act_names]

    def read_touch_sensors(self):
        """读取所有触觉传感器值"""
        values = []
        for name in self.touch_sensor_names:
            try:
                sensor_id = self.model.sensor(name).id
                values.append(self.data.sensordata[sensor_id])
            except KeyError:
                values.append(0.0)
        return np.array(values)

    def is_touched(self, threshold=0.01):
        """检测是否被触摸"""
        touch_values = self.read_touch_sensors()
        return np.any(touch_values > threshold)

    def get_trot_gait(self, phase, frequency, step_length, step_height, turn=0.0):
        """
        对角小跑 (trot) 步态生成
        FL+RR 对角同步，FR+RL 对角同步，相位差 180°
        phase: 步态相位 [0, 2*pi)
        """
        targets = self.stand_pose.copy()

        # 对角组: FL(0,1,2) + RR(9,10,11) 同相
        #          FR(3,4,5) + RL(6,7,8) 反相
        swing1 = 0.5 * (np.sin(phase) + 1.0)  # FL+RR 摆动相
        swing2 = 0.5 * (np.sin(phase + np.pi) + 1.0)  # FR+RL 摆动相

        # 抬腿通过 hip_x 旋转实现
        targets[0] += step_height * swing1 * 0.5   # FL hip_x
        targets[3] += step_height * swing2 * 0.5   # FR hip_x
        targets[6] += step_height * swing1 * 0.5   # RL hip_x
        targets[9] += step_height * swing2 * 0.5   # RR hip_x

        # 前进/后退通过 hip_z 旋转实现
        targets[1] += step_length * np.sin(phase)      # FL hip_z
        targets[4] += step_length * np.sin(phase + np.pi)  # FR hip_z
        targets[7] -= step_length * np.sin(phase)      # RL hip_z (rear opposite)
        targets[10] -= step_length * np.sin(phase + np.pi)  # RR hip_z

        # 转向
        targets[1] += turn * swing1
        targets[4] -= turn * swing2
        targets[7] += turn * swing1
        targets[10] -= turn * swing2

        # 膝关节弯曲 - 摆动相时收起
        knee_bend = 20.0 * np.pi / 180.0
        targets[2] += knee_bend * swing1   # FL knee
        targets[5] += knee_bend * swing2   # FR knee
        targets[8] += knee_bend * swing1   # RL knee
        targets[11] += knee_bend * swing2  # RR knee

        return targets

    def step(self, dt, dog_pos, human_pos, ball_pos):
        """每步更新"""
        self.touch_cooldown = max(0, self.touch_cooldown - dt)

        touch_values = self.read_touch_sensors()
        is_touched = np.any(touch_values > 0.01)

        # 行为状态机
        if is_touched and self.touch_cooldown <= 0 and self.mode != "touched":
            self.mode = "touched"
            self.touch_cooldown = 2.0  # 2秒冷却
            self.wag_phase = 0.0

        if self.mode == "touched":
            self.wag_phase += dt * 8.0
            if self.wag_phase > 2.0 * np.pi:
                self.mode = "walking"
                self.wag_phase = 0.0

        # 根据模式生成控制
        if self.mode == "idle":
            targets = self.stand_pose.copy()
            # 微小呼吸效果
            targets[0] += 0.02 * np.sin(time.time() * 3.0)
            targets[3] += 0.02 * np.sin(time.time() * 3.0)
            targets[6] += 0.02 * np.sin(time.time() * 3.0 + np.pi)
            targets[9] += 0.02 * np.sin(time.time() * 3.0 + np.pi)

        elif self.mode == "touched":
            # 被抚摸 - 开心摇摆
            targets = self.stand_pose.copy()
            wag = 0.15 * np.sin(self.wag_phase * 2.0)
            targets[1] += wag
            targets[4] += wag
            targets[7] += wag * 0.5
            targets[10] += wag * 0.5
            # 身体微微下蹲
            targets[2] += 0.1
            targets[5] += 0.1
            targets[8] += 0.1
            targets[11] += 0.1

        elif self.mode == "playing":
            # 玩耍模式 - 追球
            direction = ball_pos[:2] - dog_pos[:2]
            dist = np.linalg.norm(direction)
            if dist > 0.3:
                direction = direction / (dist + 0.01)
                # 计算转向
                dog_forward = np.array([1.0, 0.0, 0.0])
                cross = np.cross(dog_forward[:2], direction)[-1] if len(direction) >= 2 else 0.0
                turn = np.clip(cross * 0.3, -0.15, 0.15)
                self.gait_phase += dt * self.gait_frequency * 2.0 * np.pi
                targets = self.get_trot_gait(
                    self.gait_phase, self.gait_frequency,
                    self.step_length * 0.6, self.step_height * 0.5, turn
                )
            else:
                targets = self.stand_pose.copy()
        else:
            # walking - 跟随人形机器人
            direction = human_pos[:2] - dog_pos[:2]
            dist = np.linalg.norm(direction)
            if dist > 1.0:
                direction = direction / (dist + 0.01)
                dog_forward = np.array([1.0, 0.0, 0.0])
                cross = np.cross(dog_forward[:2], direction[:2])
                turn = np.clip(cross * 0.2, -0.1, 0.1)
                speed = min(dist * 0.5, 1.0)
                self.gait_phase += dt * self.gait_frequency * speed * 2.0 * np.pi
                targets = self.get_trot_gait(
                    self.gait_phase, self.gait_frequency,
                    self.step_length * speed, self.step_height * speed, turn
                )
            else:
                self.mode = "idle"
                targets = self.stand_pose.copy()

        self.set_control(targets)


# ============================================================
# 人形陪护机器人控制器
# ============================================================

class HumanoidController(Controller):
    """小型人形陪护机器人控制器"""

    def __init__(self, model, data):
        super().__init__(model, data, "humanoid")

        self.joint_names = [
            "human_neck",
            "human_la_shoulder_x", "human_la_shoulder_z", "human_la_elbow",
            "human_ra_shoulder_x", "human_ra_shoulder_z", "human_ra_elbow",
            "human_ll_hip_x", "human_ll_hip_z", "human_ll_knee",
            "human_rl_hip_x", "human_rl_hip_z", "human_rl_knee",
        ]

        # 默认站立姿态
        self.stand_pose = np.array([
            0.0,      # neck
            -30.0, 0.0, -30.0,  # left arm
            -30.0, 0.0, -30.0,  # right arm
            0.0, 0.0, 0.0,      # left leg
            0.0, 0.0, 0.0,      # right leg
        ]) * np.pi / 180.0

        # 挥手姿态
        self.wave_pose = self.stand_pose.copy()
        self.wave_pose[1] = -90.0 * np.pi / 180.0  # left shoulder up
        self.wave_pose[2] = -30.0 * np.pi / 180.0  # left shoulder forward
        self.wave_pose[3] = -60.0 * np.pi / 180.0  # left elbow

        # 递物姿态
        self.offer_pose = self.stand_pose.copy()
        self.offer_pose[1] = -60.0 * np.pi / 180.0
        self.offer_pose[2] = -90.0 * np.pi / 180.0
        self.offer_pose[3] = -20.0 * np.pi / 180.0
        self.offer_pose[4] = -60.0 * np.pi / 180.0
        self.offer_pose[5] = 90.0 * np.pi / 180.0
        self.offer_pose[6] = -20.0 * np.pi / 180.0

        self.touch_sensor_names = [
            "human_head_touch_sensor", "human_la_touch_sensor",
            "human_ra_touch_sensor", "human_ll_touch_sensor",
            "human_rl_touch_sensor", "human_body_touch_sensor",
        ]

        self.mode = "idle"
        self.behavior_phase = 0.0
        self.behavior_timer = 0.0
        self.touch_cooldown = 0.0
        self.walk_phase = 0.0

    def _init_actuators(self):
        act_names = [
            "human_neck_act",
            "human_la_shoulder_x_act", "human_la_shoulder_z_act", "human_la_elbow_act",
            "human_ra_shoulder_x_act", "human_ra_shoulder_z_act", "human_ra_elbow_act",
            "human_ll_hip_x_act", "human_ll_hip_z_act", "human_ll_knee_act",
            "human_rl_hip_x_act", "human_rl_hip_z_act", "human_rl_knee_act",
        ]
        self.actuator_ids = [self.model.actuator(name).id for name in act_names]

    def read_touch_sensors(self):
        values = []
        for name in self.touch_sensor_names:
            try:
                sensor_id = self.model.sensor(name).id
                values.append(self.data.sensordata[sensor_id])
            except KeyError:
                values.append(0.0)
        return np.array(values)

    def is_touched(self, threshold=0.01):
        touch_values = self.read_touch_sensors()
        return np.any(touch_values > threshold)

    def get_walk_gait(self, phase):
        """人形机器人行走步态"""
        targets = self.stand_pose.copy()

        # 左右腿交替摆动
        l_swing = 0.5 * (np.sin(phase) + 1.0)
        r_swing = 0.5 * (np.sin(phase + np.pi) + 1.0)

        # 抬腿
        hip_lift = 30.0 * np.pi / 180.0
        targets[7] += hip_lift * l_swing   # left hip_x
        targets[10] += hip_lift * r_swing  # right hip_x

        # 膝关节
        knee_bend = 40.0 * np.pi / 180.0
        targets[9] += knee_bend * l_swing
        targets[12] += knee_bend * r_swing

        # 手臂摆动
        arm_swing = 20.0 * np.pi / 180.0
        targets[1] += arm_swing * np.sin(phase)      # left shoulder
        targets[4] += arm_swing * np.sin(phase + np.pi)  # right shoulder

        return targets

    def step(self, dt, human_pos, dog_pos, ball_pos, cup_pos):
        """每步更新"""
        self.touch_cooldown = max(0, self.touch_cooldown - dt)
        self.behavior_timer += dt

        touch_values = self.read_touch_sensors()
        is_touched = np.any(touch_values > 0.01)

        # 行为状态机
        if is_touched and self.touch_cooldown <= 0:
            self.mode = "touched"
            self.touch_cooldown = 2.0
            self.behavior_phase = 0.0

        if self.mode == "touched" and self.behavior_timer > 2.0:
            self.mode = "walking"
            self.behavior_timer = 0.0

        # 周期性行为切换
        if self.mode == "idle":
            self.behavior_phase += dt
            cycle = self.behavior_phase % 15.0

            if cycle < 3.0:
                # 站立
                targets = self.stand_pose.copy()
                # 头部微微转动
                targets[0] = 0.2 * np.sin(self.behavior_phase * 1.5)
            elif cycle < 6.0:
                # 挥手
                targets = self.wave_pose.copy()
                wave = 0.3 * np.sin(self.behavior_phase * 5.0)
                targets[1] += wave
                targets[0] = 0.3 * np.sin(self.behavior_phase * 2.0)
            elif cycle < 9.0:
                # 递物
                t = (cycle - 6.0) / 3.0
                targets = self.stand_pose * (1 - t) + self.offer_pose * t
                targets[0] = 0.2 * np.sin(self.behavior_phase * 2.0)
            elif cycle < 12.0:
                # 收回手臂
                t = (cycle - 9.0) / 3.0
                targets = self.offer_pose * (1 - t) + self.wave_pose * t
            else:
                # 回到站立
                targets = self.stand_pose.copy()

        elif self.mode == "touched":
            # 被抚摸 - 开心反应
            targets = self.stand_pose.copy()
            # 头部快速转动
            targets[0] = 0.6 * np.sin(self.behavior_phase * 8.0)
            # 手臂微微抬起
            targets[1] += 0.2 * np.sin(self.behavior_phase * 6.0)
            targets[4] += 0.2 * np.sin(self.behavior_phase * 6.0 + np.pi)

        elif self.mode == "walking":
            # 走向目标位置
            direction = dog_pos[:2] - human_pos[:2]
            dist = np.linalg.norm(direction)

            if dist > 0.8:
                self.walk_phase += dt * 2.0 * np.pi
                targets = self.get_walk_gait(self.walk_phase)
            else:
                self.mode = "idle"
                targets = self.stand_pose.copy()

        self.set_control(targets)


# ============================================================
# 电影级运镜摄像机
# ============================================================

class CinematicCamera:
    """电影级运镜：环绕、推拉、摇移，自动跟踪双机器人"""

    def __init__(self):
        self.time = 0.0
        # 基础参数
        self.base_distance = 5.0    # 基础距离
        self.base_height = 2.5      # 基础高度
        self.base_azimuth = 0.0     # 基础方位角

    def update(self, dt, dog_pos, human_pos):
        """更新摄像机位姿，返回 (lookat, distance, azimuth, elevation)"""
        self.time += dt

        # 场景中心：狗和人形机器人的中点
        center = (dog_pos + human_pos) / 2.0
        # 让中心稍微偏向人形机器人，方便展示交互
        center = center * 0.7 + human_pos * 0.3

        # 30秒的电影运镜脚本
        t = self.time

        # 阶段划分
        if t < 5.0:
            # 阶段1：远景建立镜头 - 从高处俯瞰整个房间，缓慢下降
            progress = t / 5.0
            distance = 6.0 - progress * 1.5  # 6.0 -> 4.5
            height = 4.0 - progress * 1.0     # 4.0 -> 3.0
            azimuth = progress * 30.0         # 0 -> 30度，缓慢旋转展示场景
            elevation = -25.0 - progress * 5.0  # 俯视角度渐减

        elif t < 10.0:
            # 阶段2：中景环绕 - 围绕两个机器人旋转
            progress = (t - 5.0) / 5.0
            distance = 4.5 - progress * 0.5    # 4.5 -> 4.0
            height = 3.0 - progress * 0.5      # 3.0 -> 2.5
            azimuth = 30.0 + progress * 90.0   # 30 -> 120度，绕半圈
            elevation = -30.0 + progress * 5.0  # 逐渐变平

        elif t < 15.0:
            # 阶段3：低角度跟随 - 展示机器狗行走
            progress = (t - 10.0) / 5.0
            distance = 4.0 - progress * 1.0    # 4.0 -> 3.0，推近
            height = 2.5 - progress * 1.0      # 2.5 -> 1.5，降低
            azimuth = 120.0 + progress * 40.0  # 120 -> 160
            elevation = -25.0 + progress * 15.0  # 变平，接近平视

        elif t < 20.0:
            # 阶段4：特写人形机器人 - 展示挥手/递物
            progress = (t - 15.0) / 5.0
            distance = 3.0 + progress * 0.5     # 3.0 -> 3.5，微微拉远
            height = 1.5 + progress * 0.3       # 1.5 -> 1.8
            azimuth = 160.0 + progress * 30.0   # 160 -> 190
            elevation = -10.0 + progress * 5.0  # 微调

        elif t < 25.0:
            # 阶段5：侧面环绕 - 展示沙发和整体环境
            progress = (t - 20.0) / 5.0
            distance = 3.5 + progress * 1.5    # 3.5 -> 5.0，拉远
            height = 1.8 + progress * 1.0      # 1.8 -> 2.8
            azimuth = 190.0 + progress * 80.0  # 190 -> 270
            elevation = -5.0 - progress * 15.0 # 俯视

        else:
            # 阶段6：结束 - 拉远全景
            progress = (t - 25.0) / 5.0
            distance = 5.0 + progress * 1.5    # 5.0 -> 6.5
            height = 2.8 + progress * 1.2      # 2.8 -> 4.0
            azimuth = 270.0 + progress * 45.0  # 270 -> 315
            elevation = -20.0 - progress * 10.0

        return center, distance, azimuth, elevation


# ============================================================
# 仿真场景管理器
# ============================================================

class Simulation:
    """家庭互动仿真主控制器"""

    def __init__(self, scene_path, render_mode="offscreen"):
        self.scene_path = scene_path
        self.render_mode = render_mode

        # 加载模型
        print(f"Loading scene: {scene_path}")
        self.model = mujoco.MjModel.from_xml_path(str(scene_path))
        self.data = mujoco.MjData(self.model)

        # 渲染器
        if render_mode == "offscreen":
            self.renderer = mujoco.Renderer(self.model, 720, 1280)
        else:
            self.renderer = None

        # 初始化控制器
        self.dog = DogController(self.model, self.data)
        self.humanoid = HumanoidController(self.model, self.data)

        # 仿真参数
        self.dt = self.model.opt.timestep
        self.sim_time = 0.0
        self.frame_count = 0

        # 摄像机
        self.cinematic_cam = CinematicCamera()
        self.camera_name = "tracking"  # fallback

        # 性能统计
        self.touch_events = 0
        self.dog_mode_history = deque(maxlen=100)

        print("Simulation initialized successfully!")
        print(f"  Timestep: {self.dt}s")
        print(f"  Dog joints: {len(self.dog.joint_names)}")
        print(f"  Humanoid joints: {len(self.humanoid.joint_names)}")
        print(f"  Touch sensors: dog={len(self.dog.touch_sensor_names)}, humanoid={len(self.humanoid.touch_sensor_names)}")

    def get_sensor_data(self):
        """获取传感器数据"""
        def get_sensor(name):
            try:
                sensor_id = self.model.sensor(name).id
                adr = self.model.sensor_adr[sensor_id]
                dim = self.model.sensor(sensor_id).dim[0]
                return self.data.sensordata[adr:adr + dim].copy()
            except KeyError:
                return None

        return {
            "dog_pos": get_sensor("dog_pos"),
            "human_pos": get_sensor("human_pos"),
            "ball_pos": get_sensor("ball_pos"),
            "cup_pos": get_sensor("cup_pos"),
        }

    def update_camera(self, dog_pos, human_pos):
        """电影级运镜：计算 free camera 位姿"""
        lookat, distance, azimuth, elevation = self.cinematic_cam.update(
            self.dt, dog_pos, human_pos
        )

        # 球坐标转笛卡尔坐标
        az_rad = np.deg2rad(azimuth)
        el_rad = np.deg2rad(elevation)

        cam_x = lookat[0] + distance * np.cos(el_rad) * np.cos(az_rad)
        cam_y = lookat[1] + distance * np.cos(el_rad) * np.sin(az_rad)
        cam_z = lookat[2] + distance * np.sin(el_rad)

        cam_pos = np.array([cam_x, cam_y, cam_z])

        # 计算摄像机朝向矩阵
        forward = lookat - cam_pos
        forward = forward / (np.linalg.norm(forward) + 1e-10)

        # 世界 up 向量
        world_up = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, world_up)
        right = right / (np.linalg.norm(right) + 1e-10)
        up = np.cross(right, forward)

        # 构建旋转矩阵 (列主序)
        rot = np.column_stack([right, up, -forward])

        # 设置摄像机
        cam = self.data.camera("tracking")
        cam.xpos = cam_pos
        cam.xmat = rot.T.flatten()  # MuJoCo 使用行主序

    def render_frame(self):
        """渲染当前帧"""
        if self.renderer is None:
            return None

        self.renderer.update_scene(self.data, camera=self.camera_name)
        pixels = self.renderer.render()
        return pixels

    def step(self):
        """执行一步仿真"""
        # 获取传感器数据
        sensors = self.get_sensor_data()
        dog_pos = sensors["dog_pos"] if sensors["dog_pos"] is not None else np.zeros(3)
        human_pos = sensors["human_pos"] if sensors["human_pos"] is not None else np.zeros(3)
        ball_pos = sensors["ball_pos"] if sensors["ball_pos"] is not None else np.zeros(3)
        cup_pos = sensors["cup_pos"] if sensors["cup_pos"] is not None else np.zeros(3)

        # 更新控制器
        self.dog.step(self.dt, dog_pos, human_pos, ball_pos)
        self.humanoid.step(self.dt, human_pos, dog_pos, ball_pos, cup_pos)

        # 检测触摸事件
        if self.dog.is_touched() or self.humanoid.is_touched():
            self.touch_events += 1

        # 推进物理仿真
        for _ in range(5):  # 子步
            mujoco.mj_step(self.model, self.data)

        self.sim_time += self.dt
        self.frame_count += 1

        # 更新摄像机
        self.update_camera(dog_pos, human_pos)

        # 记录狗的模式
        self.dog_mode_history.append(self.dog.mode)

    def run(self, duration=30.0, record_video=False, video_path=None):
        """运行仿真"""
        print(f"\nStarting simulation for {duration}s...")
        print(f"Recording video: {record_video}")

        frames = []
        total_steps = int(duration / self.dt)
        fps = int(1.0 / self.dt)
        frame_interval = max(1, fps // 30)  # 30fps 输出

        start_time = time.time()
        for step_i in range(total_steps):
            self.step()

            if record_video and step_i % frame_interval == 0:
                frame = self.render_frame()
                if frame is not None:
                    frames.append(frame)

            # 进度显示
            if step_i % 1000 == 0:
                elapsed = time.time() - start_time
                progress = step_i / total_steps * 100
                print(f"\r  Progress: {progress:.1f}% | "
                      f"Time: {self.sim_time:.1f}s | "
                      f"Dog: {self.dog.mode:8s} | "
                      f"Humanoid: {self.humanoid.mode:8s} | "
                      f"Touches: {self.touch_events}",
                      end="", flush=True)

        elapsed = time.time() - start_time
        print(f"\n\nSimulation complete!")
        print(f"  Duration: {elapsed:.1f}s wall time")
        print(f"  Simulated: {self.sim_time:.1f}s")
        print(f"  Touch events: {self.touch_events}")
        print(f"  Frames captured: {len(frames)}")

        # 保存视频
        if record_video and frames and video_path:
            self._save_video(frames, video_path, fps=30)

        return frames

    def _save_video(self, frames, video_path, fps=30):
        """保存渲染帧为 MP4 视频"""
        print(f"\nSaving video to {video_path}...")
        if not frames:
            print("  No frames to save!")
            return

        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(video_path), fourcc, fps, (w, h))

        for frame in frames:
            # MuJoCo 渲染输出是 RGB，OpenCV 需要 BGR
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(bgr)

        out.release()
        print(f"  Video saved: {video_path} ({len(frames)} frames, {fps}fps)")


# ============================================================
# 主程序
# ============================================================

def main():
    # 项目路径
    project_dir = Path(__file__).parent
    scene_path = project_dir / "assets" / "scene.xml"
    video_path = project_dir / "demo_video.mp4"

    if not scene_path.exists():
        print(f"ERROR: Scene file not found: {scene_path}")
        sys.exit(1)

    # 创建仿真
    sim = Simulation(str(scene_path), render_mode="offscreen")

    # 运行仿真并录制视频
    duration = 30.0  # 30秒
    sim.run(duration=duration, record_video=True, video_path=str(video_path))

    # 转码为 H.264 通用格式
    if video_path.exists():
        import subprocess
        tmp_path = video_path.with_suffix(".tmp.mp4")
        video_path.rename(tmp_path)
        result = subprocess.run([
            "ffmpeg", "-y", "-i", str(tmp_path),
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(video_path)
        ], capture_output=True)
        if result.returncode == 0:
            tmp_path.unlink()
            print(f"Video transcoded to H.264: {video_path}")
        else:
            tmp_path.rename(video_path)
            print("Warning: H.264 transcode failed, kept original mp4v")

    print("\nDone! All outputs saved to:", project_dir)


if __name__ == "__main__":
    main()