"""
仿生四足 + 人形宠物陪护机器人家庭互动仿真
==============================================
MuJoCo 物理仿真 + Matplotlib 2D 可视化
双机器人共生交互：机器狗跟随人形陪护机器人，
共同完成递物、躲避障碍物、抚摸交互。
"""
import mujoco
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Ellipse, FancyBboxPatch, Polygon, Arc
from matplotlib.animation import FFMpegWriter
import matplotlib.patches as mpatches
from pathlib import Path
import subprocess
import time


# ============================================================
# 控制器基类
# ============================================================

class Controller:
    def __init__(self, model, data, name):
        self.model = model
        self.data = data
        self.name = name
        self.actuator_ids = []
        self._init_actuators()

    def _init_actuators(self):
        raise NotImplementedError

    def set_control(self, targets):
        for act_id, target in zip(self.actuator_ids, targets):
            self.data.ctrl[act_id] = target


# ============================================================
# 机器狗控制器 (Unitree Go1) - 对角小跑步态
# ============================================================

class DogController(Controller):
    def __init__(self, model, data):
        super().__init__(model, data, "robot_dog")
        self.stand_pose = np.array([
            0.0, -5.0, 30.0,
            0.0, 5.0, 30.0,
            0.0, -5.0, 30.0,
            0.0, 5.0, 30.0,
        ]) * np.pi / 180.0
        self.gait_phase = 0.0
        self.gait_frequency = 2.5
        self.step_height = 0.06
        self.step_length = 0.08
        self.touch_sensor_names = [
            "dog_fl_touch_sensor", "dog_fr_touch_sensor",
            "dog_rl_touch_sensor", "dog_rr_touch_sensor",
            "dog_body_touch_sensor", "dog_head_touch_sensor",
        ]
        self.mode = "walking"
        self.touch_cooldown = 0
        self.wag_phase = 0.0

    def _init_actuators(self):
        names = [
            "dog_fl_hip_x_act", "dog_fl_hip_z_act", "dog_fl_knee_act",
            "dog_fr_hip_x_act", "dog_fr_hip_z_act", "dog_fr_knee_act",
            "dog_rl_hip_x_act", "dog_rl_hip_z_act", "dog_rl_knee_act",
            "dog_rr_hip_x_act", "dog_rr_hip_z_act", "dog_rr_knee_act",
        ]
        self.actuator_ids = [self.model.actuator(n).id for n in names]

    def read_touch_sensors(self):
        values = []
        for name in self.touch_sensor_names:
            try:
                values.append(self.data.sensordata[self.model.sensor(name).id])
            except KeyError:
                values.append(0.0)
        return np.array(values)

    def get_trot_gait(self, phase, step_length, step_height, turn=0.0):
        targets = self.stand_pose.copy()
        swing1 = 0.5 * (np.sin(phase) + 1.0)
        swing2 = 0.5 * (np.sin(phase + np.pi) + 1.0)
        targets[0] += step_height * swing1 * 0.5
        targets[3] += step_height * swing2 * 0.5
        targets[6] += step_height * swing1 * 0.5
        targets[9] += step_height * swing2 * 0.5
        targets[1] += step_length * np.sin(phase)
        targets[4] += step_length * np.sin(phase + np.pi)
        targets[7] -= step_length * np.sin(phase)
        targets[10] -= step_length * np.sin(phase + np.pi)
        targets[1] += turn * swing1
        targets[4] -= turn * swing2
        targets[7] += turn * swing1
        targets[10] -= turn * swing2
        knee_bend = 20.0 * np.pi / 180.0
        targets[2] += knee_bend * swing1
        targets[5] += knee_bend * swing2
        targets[8] += knee_bend * swing1
        targets[11] += knee_bend * swing2
        return targets

    def step(self, dt, dog_pos, human_pos, ball_pos):
        self.touch_cooldown = max(0, self.touch_cooldown - dt)
        if self.mode == "touched":
            self.wag_phase += dt * 8.0
            if self.wag_phase > 2.0 * np.pi:
                self.mode = "walking"
            targets = self.stand_pose.copy()
            wag = 0.15 * np.sin(self.wag_phase * 2.0)
            for i in [1, 4, 7, 10]:
                targets[i] += wag * (0.5 if i > 6 else 1.0)
            for i in [2, 5, 8, 11]:
                targets[i] += 0.1
        elif self.mode == "idle":
            targets = self.stand_pose.copy()
            t = time.time()
            targets[0] += 0.02 * np.sin(t * 3.0)
            targets[3] += 0.02 * np.sin(t * 3.0)
            targets[6] += 0.02 * np.sin(t * 3.0 + np.pi)
            targets[9] += 0.02 * np.sin(t * 3.0 + np.pi)
        else:
            direction = human_pos[:2] - dog_pos[:2]
            dist = np.linalg.norm(direction)
            if dist > 0.5:
                direction = direction / (dist + 0.01)
                fwd = np.array([1.0, 0.0, 0.0])
                cross = np.cross(fwd[:2], direction[:2])
                turn = np.clip(cross * 0.2, -0.1, 0.1)
                speed = min(dist * 0.8, 1.5)
                self.gait_phase += dt * self.gait_frequency * speed * 2.0 * np.pi
                targets = self.get_trot_gait(self.gait_phase, self.step_length * speed, self.step_height * speed, turn)
            else:
                self.mode = "idle"
                targets = self.stand_pose.copy()
        self.set_control(targets)


# ============================================================
# 人形陪护机器人控制器
# ============================================================

class HumanoidController(Controller):
    def __init__(self, model, data):
        super().__init__(model, data, "humanoid")
        self.stand_pose = np.array([
            0.0,
            -30.0, 0.0, -30.0,
            -30.0, 0.0, -30.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
        ]) * np.pi / 180.0
        self.wave_pose = self.stand_pose.copy()
        self.wave_pose[1] = -90.0 * np.pi / 180.0
        self.wave_pose[2] = -30.0 * np.pi / 180.0
        self.wave_pose[3] = -60.0 * np.pi / 180.0
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
        self.behavior_phase = 3.0
        self.behavior_timer = 0.0
        self.walk_phase = 0.0

    def _init_actuators(self):
        names = [
            "human_neck_act",
            "human_la_shoulder_x_act", "human_la_shoulder_z_act", "human_la_elbow_act",
            "human_ra_shoulder_x_act", "human_ra_shoulder_z_act", "human_ra_elbow_act",
            "human_ll_hip_x_act", "human_ll_hip_z_act", "human_ll_knee_act",
            "human_rl_hip_x_act", "human_rl_hip_z_act", "human_rl_knee_act",
        ]
        self.actuator_ids = [self.model.actuator(n).id for n in names]

    def step(self, dt, human_pos, dog_pos, ball_pos, cup_pos):
        self.behavior_timer += dt
        if self.mode == "idle":
            self.behavior_phase += dt
            cycle = self.behavior_phase % 10.0
            if cycle < 2.0:
                targets = self.stand_pose.copy()
                targets[0] = 0.2 * np.sin(self.behavior_phase * 1.5)
            elif cycle < 4.5:
                targets = self.wave_pose.copy()
                targets[1] += 0.3 * np.sin(self.behavior_phase * 5.0)
                targets[0] = 0.3 * np.sin(self.behavior_phase * 2.0)
            elif cycle < 7.0:
                t = (cycle - 4.5) / 2.5
                targets = self.stand_pose * (1 - t) + self.offer_pose * t
                targets[0] = 0.2 * np.sin(self.behavior_phase * 2.0)
            elif cycle < 9.0:
                t = (cycle - 7.0) / 2.0
                targets = self.offer_pose * (1 - t) + self.wave_pose * t
            else:
                targets = self.stand_pose.copy()
        elif self.mode == "touched":
            targets = self.stand_pose.copy()
            targets[0] = 0.6 * np.sin(self.behavior_phase * 8.0)
            targets[1] += 0.2 * np.sin(self.behavior_phase * 6.0)
            targets[4] += 0.2 * np.sin(self.behavior_phase * 6.0 + np.pi)
        else:
            targets = self.stand_pose.copy()
        self.set_control(targets)


# ============================================================
# 2D 场景绘制
# ============================================================

def draw_scene(ax, data, model, sim_time):
    ax.clear()
    ax.set_xlim(-4, 5)
    ax.set_ylim(-3, 4)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_facecolor('#F5E6D3')

    # Carpet
    carpet = Rectangle((-2.5, -1.8), 5, 3.6, facecolor='#A8C9A5', edgecolor='#8AB887', linewidth=2, alpha=0.7, zorder=1)
    ax.add_patch(carpet)

    # Sofa
    sofa = FancyBboxPatch((2.5, -0.9), 2.0, 1.8, boxstyle="round,pad=0.1", facecolor='#C4956A', edgecolor='#8B6914', linewidth=2, zorder=2)
    ax.add_patch(sofa)
    ax.text(3.5, 0, 'Sofa', ha='center', va='center', fontsize=9, color='#5C3A1E', zorder=3, weight='bold')

    # Pillows
    p1 = Ellipse((2.8, 0.5), 0.3, 0.2, angle=15, facecolor='#E8C9A0', edgecolor='#C4956A', linewidth=1.5, zorder=3)
    p2 = Ellipse((4.2, 0.5), 0.3, 0.2, angle=-15, facecolor='#E8C9A0', edgecolor='#C4956A', linewidth=1.5, zorder=3)
    ax.add_patch(p1); ax.add_patch(p2)

    # Toy ball
    bp = data.body('toy_ball').xpos
    ax.add_patch(Circle((bp[0], bp[1]), 0.12, facecolor='#FF6B6B', edgecolor='#CC4444', linewidth=2, zorder=5))
    ax.text(bp[0], bp[1], '*', ha='center', va='center', fontsize=14, color='white', zorder=6, weight='bold')

    # Cup
    cp = data.body('cup').xpos
    ax.add_patch(Circle((cp[0], cp[1]), 0.06, facecolor='#87CEEB', edgecolor='#5555AA', linewidth=2, alpha=0.8, zorder=5))
    ax.text(cp[0], cp[1], 'C', ha='center', va='center', fontsize=8, color='white', zorder=6, weight='bold')

    # Obstacles
    ob = data.body('obstacle_box').xpos
    ax.add_patch(Rectangle((ob[0]-0.15, ob[1]-0.15), 0.3, 0.3, facecolor='#999999', edgecolor='#666666', linewidth=1.5, zorder=4))
    oc = data.body('obstacle_cylinder').xpos
    ax.add_patch(Circle((oc[0], oc[1]), 0.08, facecolor='#66AA66', edgecolor='#448844', linewidth=1.5, zorder=4))

    # ===== Robot Dog =====
    dx, dy = data.body('robot_dog').xpos[0], data.body('robot_dog').xpos[1]
    ax.add_patch(FancyBboxPatch((dx-0.35, dy-0.12), 0.7, 0.24, boxstyle="round,pad=0.05", facecolor='#4A6FA5', edgecolor='#2C4A7C', linewidth=2, zorder=10))
    ax.add_patch(Rectangle((dx+0.28, dy-0.08), 0.16, 0.16, facecolor='#4A6FA5', edgecolor='#2C4A7C', linewidth=2, zorder=10))
    ax.add_patch(Circle((dx+0.38, dy+0.04), 0.02, facecolor='#00FFCC', zorder=11))
    ax.add_patch(Circle((dx+0.38, dy-0.04), 0.02, facecolor='#00FFCC', zorder=11))
    ax.add_patch(Polygon([(dx+0.32, dy+0.06), (dx+0.3, dy+0.14), (dx+0.36, dy+0.08)], facecolor='#3A5A8A', edgecolor='#2C4A7C', linewidth=1, zorder=9))
    ax.add_patch(Polygon([(dx+0.32, dy-0.06), (dx+0.3, dy-0.14), (dx+0.36, dy-0.08)], facecolor='#3A5A8A', edgecolor='#2C4A7C', linewidth=1, zorder=9))
    ax.add_patch(Arc((dx-0.35, dy+0.05), 0.2, 0.15, angle=30, theta1=0, theta2=180, color='#3A5A8A', linewidth=2, zorder=9))
    for lx, ly in [(dx+0.22, dy+0.1), (dx+0.22, dy-0.1), (dx-0.22, dy+0.1), (dx-0.22, dy-0.1)]:
        ax.add_patch(Rectangle((lx-0.03, ly-0.07), 0.06, 0.14, facecolor='#3A5A8A', edgecolor='#2C4A7C', linewidth=1, zorder=8))
        ax.add_patch(Circle((lx, ly-0.08), 0.04, facecolor='#222222', edgecolor='#111111', linewidth=1, zorder=8))
    ax.text(dx, dy-0.3, 'Unitree Go1', ha='center', fontsize=8, color='#4A6FA5', weight='bold', zorder=12)

    # ===== Humanoid Robot =====
    hx, hy = data.body('humanoid').xpos[0], data.body('humanoid').xpos[1]
    ax.add_patch(FancyBboxPatch((hx-0.08, hy-0.06), 0.16, 0.12, boxstyle="round,pad=0.03", facecolor='#E8E8E8', edgecolor='#999999', linewidth=2, zorder=10))
    ax.add_patch(Circle((hx, hy+0.12), 0.06, facecolor='#E8E8E8', edgecolor='#999999', linewidth=2, zorder=10))
    ax.add_patch(Circle((hx+0.02, hy+0.14), 0.012, facecolor='#00AAFF', zorder=11))
    ax.add_patch(Circle((hx+0.02, hy+0.10), 0.012, facecolor='#00AAFF', zorder=11))
    ant = mpatches.FancyArrow(hx, hy+0.18, 0, 0.06, width=0.015, head_width=0.03, head_length=0.02, facecolor='#FF6600', edgecolor='#CC4400', zorder=11)
    ax.add_patch(ant)
    for off in [0.1, -0.1]:
        ax.add_patch(Rectangle((hx+0.06, hy+off-0.01), 0.1, 0.04, facecolor='#D0D0D0', edgecolor='#999999', linewidth=1, zorder=9))
        ax.add_patch(Circle((hx+0.17, hy+off+0.01), 0.025, facecolor='#E8E8E8', edgecolor='#999999', linewidth=1, zorder=9))
    for off in [0.03, -0.03]:
        ax.add_patch(Rectangle((hx+off-0.02, hy-0.12), 0.04, 0.1, facecolor='#D0D0D0', edgecolor='#999999', linewidth=1, zorder=8))
        ax.add_patch(Rectangle((hx+off-0.03, hy-0.22), 0.06, 0.03, facecolor='#333333', edgecolor='#111111', linewidth=1, zorder=8))
    ax.text(hx, hy-0.3, 'Companion Bot', ha='center', fontsize=8, color='#666666', weight='bold', zorder=12)

    # Title and legend
    ax.set_title('Bionic Quadruped Pet Companion - Home Interaction Simulation', fontsize=14, weight='bold', color='#333333', pad=10)
    ax.text(0.02, 0.98, 'Time: {:.1f}s'.format(sim_time), transform=ax.transAxes, fontsize=10, color='#666666', va='top')
    legend = [
        mpatches.Patch(color='#4A6FA5', label='Unitree Go1 Dog'),
        mpatches.Patch(color='#E8E8E8', label='Companion Humanoid'),
        mpatches.Patch(color='#C4956A', label='Flexible Sofa'),
        mpatches.Patch(color='#FF6B6B', label='Toy Ball'),
        mpatches.Patch(color='#87CEEB', label='Water Cup'),
    ]
    ax.legend(handles=legend, loc='lower right', fontsize=7, ncol=3, framealpha=0.8, edgecolor='#CCCCCC')


# ============================================================
# Main
# ============================================================

def main():
    project_dir = Path(__file__).parent
    scene_path = project_dir / "assets" / "scene.xml"
    video_path = project_dir / "demo_video.mp4"

    print("Loading MuJoCo model...")
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)

    dog = DogController(model, data)
    humanoid = HumanoidController(model, data)

    dt = model.opt.timestep
    duration = 20.0
    fps = 30
    total_frames = int(duration * fps)
    steps_per_frame = int(1.0 / (dt * fps))

    print("Generating {} frames at {}fps...".format(total_frames, fps))

    fig, ax = plt.subplots(figsize=(12, 8), dpi=120)
    writer = FFMpegWriter(fps=fps, codec='libx264', bitrate=2000)

    def get_sensor(name):
        sid = model.sensor(name).id
        adr = model.sensor_adr[sid]
        dim = model.sensor(sid).dim[0]
        return data.sensordata[adr:adr + dim].copy()

    sim_time = 0.0
    with writer.saving(fig, str(video_path), dpi=120):
        for frame_i in range(total_frames):
            dog_pos = get_sensor("dog_pos")
            human_pos = get_sensor("human_pos")
            ball_pos = get_sensor("ball_pos")
            cup_pos = get_sensor("cup_pos")

            dog.step(dt, dog_pos, human_pos, ball_pos)
            humanoid.step(dt, human_pos, dog_pos, ball_pos, cup_pos)

            for _ in range(steps_per_frame):
                for _ in range(5):
                    mujoco.mj_step(model, data)

            draw_scene(ax, data, model, sim_time)
            writer.grab_frame()
            sim_time += steps_per_frame * dt

            if frame_i % 150 == 0:
                print("  Frame {}/{} ({:.1f}s)".format(frame_i, total_frames, sim_time))

    plt.close()
    print("Video saved:", video_path)

    # Transcode to H.264
    if video_path.exists():
        tmp = video_path.with_suffix(".tmp.mp4")
        video_path.rename(tmp)
        subprocess.run(["ffmpeg", "-y", "-i", str(tmp), "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(video_path)], capture_output=True)
        tmp.unlink()
        print("Transcoded to H.264")

    print("Done!")


if __name__ == "__main__":
    main()