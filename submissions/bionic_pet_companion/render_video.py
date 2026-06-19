"""
仿生四足宠物陪护机器人家庭互动 —— 2D 可视化视频生成
========================================================
MuJoCo 物理仿真 + Matplotlib 2D 伪3D 渲染
"""
import mujoco
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Ellipse, FancyBboxPatch, Polygon, Arc, Wedge
from matplotlib.animation import FFMpegWriter
import matplotlib.patches as mpatches
from pathlib import Path
import sys

# ============================================================
# 场景绘制
# ============================================================

def draw_scene(ax, data, model, sim_time):
    """绘制家庭场景俯视图"""
    ax.clear()
    ax.set_xlim(-4, 5)
    ax.set_ylim(-3, 4)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_facecolor('#F5E6D3')  # 暖色地板

    # 地毯
    carpet = Rectangle((-2.5, -1.8), 5, 3.6, facecolor='#A8C9A5', edgecolor='#8AB887',
                       linewidth=2, alpha=0.7, zorder=1)
    ax.add_patch(carpet)

    # 沙发
    sofa = FancyBboxPatch((2.5, -0.9), 2.0, 1.8, boxstyle="round,pad=0.1",
                          facecolor='#C4956A', edgecolor='#8B6914', linewidth=2, zorder=2)
    ax.add_patch(sofa)
    ax.text(3.5, 0, 'Sofa', ha='center', va='center', fontsize=9, color='#5C3A1E', zorder=3, weight='bold')

    # 靠枕
    pillow1 = Ellipse((2.8, 0.5), 0.3, 0.2, angle=15, facecolor='#E8C9A0', edgecolor='#C4956A', linewidth=1.5, zorder=3)
    pillow2 = Ellipse((4.2, 0.5), 0.3, 0.2, angle=-15, facecolor='#E8C9A0', edgecolor='#C4956A', linewidth=1.5, zorder=3)
    ax.add_patch(pillow1)
    ax.add_patch(pillow2)

    # 玩具球
    ball_pos = data.body('toy_ball').xpos
    ball = Circle((ball_pos[0], ball_pos[1]), 0.12, facecolor='#FF6B6B', edgecolor='#CC4444',
                  linewidth=2, zorder=5)
    ax.add_patch(ball)
    # 球上的星星图案
    ax.text(ball_pos[0], ball_pos[1], '*', ha='center', va='center', fontsize=14, color='white', zorder=6, weight='bold')

    # 水杯
    cup_pos = data.body('cup').xpos
    cup = Circle((cup_pos[0], cup_pos[1]), 0.06, facecolor='#87CEEB', edgecolor='#5555AA',
                 linewidth=2, alpha=0.8, zorder=5)
    ax.add_patch(cup)
    ax.text(cup_pos[0], cup_pos[1], 'C', ha='center', va='center', fontsize=8, color='white', zorder=6, weight='bold')

    # 障碍物
    box_pos = data.body('obstacle_box').xpos
    ob = Rectangle((box_pos[0]-0.15, box_pos[1]-0.15), 0.3, 0.3,
                   facecolor='#999999', edgecolor='#666666', linewidth=1.5, zorder=4)
    ax.add_patch(ob)

    cyl_pos = data.body('obstacle_cylinder').xpos
    cyl = Circle((cyl_pos[0], cyl_pos[1]), 0.08, facecolor='#66AA66', edgecolor='#448844',
                 linewidth=1.5, zorder=4)
    ax.add_patch(cyl)

    # ===== 机器狗 =====
    dog_pos = data.body('robot_dog').xpos
    dog_x, dog_y = dog_pos[0], dog_pos[1]
    # 身体
    dog_body = FancyBboxPatch((dog_x-0.35, dog_y-0.12), 0.7, 0.24,
                              boxstyle="round,pad=0.05", facecolor='#4A6FA5', edgecolor='#2C4A7C',
                              linewidth=2, zorder=10)
    ax.add_patch(dog_body)
    # 头
    dog_head = Rectangle((dog_x+0.28, dog_y-0.08), 0.16, 0.16, facecolor='#4A6FA5',
                         edgecolor='#2C4A7C', linewidth=2, zorder=10)
    ax.add_patch(dog_head)
    # 眼睛
    eye_l = Circle((dog_x+0.38, dog_y+0.04), 0.02, facecolor='#00FFCC', zorder=11)
    eye_r = Circle((dog_x+0.38, dog_y-0.04), 0.02, facecolor='#00FFCC', zorder=11)
    ax.add_patch(eye_l)
    ax.add_patch(eye_r)
    # 耳朵
    ear_l = Polygon([(dog_x+0.32, dog_y+0.06), (dog_x+0.3, dog_y+0.14), (dog_x+0.36, dog_y+0.08)],
                    facecolor='#3A5A8A', edgecolor='#2C4A7C', linewidth=1, zorder=9)
    ear_r = Polygon([(dog_x+0.32, dog_y-0.06), (dog_x+0.3, dog_y-0.14), (dog_x+0.36, dog_y-0.08)],
                    facecolor='#3A5A8A', edgecolor='#2C4A7C', linewidth=1, zorder=9)
    ax.add_patch(ear_l)
    ax.add_patch(ear_r)
    # 尾巴
    tail = Arc((dog_x-0.35, dog_y+0.05), 0.2, 0.15, angle=30, theta1=0, theta2=180,
               color='#3A5A8A', linewidth=2, zorder=9)
    ax.add_patch(tail)
    # 腿
    leg_color = '#3A5A8A'
    for lx, ly in [(dog_x+0.22, dog_y+0.1), (dog_x+0.22, dog_y-0.1),
                   (dog_x-0.22, dog_y+0.1), (dog_x-0.22, dog_y-0.1)]:
        leg = Rectangle((lx-0.03, ly-0.07), 0.06, 0.14, facecolor=leg_color,
                        edgecolor='#2C4A7C', linewidth=1, zorder=8)
        ax.add_patch(leg)
        # 爪子
        paw = Circle((lx, ly-0.08), 0.04, facecolor='#222222', edgecolor='#111111', linewidth=1, zorder=8)
        ax.add_patch(paw)

    # 狗标签
    ax.text(dog_x, dog_y-0.3, 'Unitree Go1', ha='center', fontsize=8, color='#4A6FA5', weight='bold', zorder=12)

    # ===== 人形机器人 =====
    human_pos = data.body('humanoid').xpos
    hx, hy = human_pos[0], human_pos[1]
    # 身体
    human_body = FancyBboxPatch((hx-0.08, hy-0.06), 0.16, 0.12,
                                boxstyle="round,pad=0.03", facecolor='#E8E8E8', edgecolor='#999999',
                                linewidth=2, zorder=10)
    ax.add_patch(human_body)
    # 头
    human_head = Circle((hx, hy+0.12), 0.06, facecolor='#E8E8E8', edgecolor='#999999',
                        linewidth=2, zorder=10)
    ax.add_patch(human_head)
    # 眼睛
    heye_l = Circle((hx+0.02, hy+0.14), 0.012, facecolor='#00AAFF', zorder=11)
    heye_r = Circle((hx+0.02, hy+0.10), 0.012, facecolor='#00AAFF', zorder=11)
    ax.add_patch(heye_l)
    ax.add_patch(heye_r)
    # 天线
    ant = mpatches.FancyArrow(hx, hy+0.18, 0, 0.06, width=0.015, head_width=0.03,
                              head_length=0.02, facecolor='#FF6600', edgecolor='#CC4400', zorder=11)
    ax.add_patch(ant)
    # 手臂
    arm_color = '#D0D0D0'
    for ax_off in [0.1, -0.1]:
        arm = Rectangle((hx+0.06, hy+ax_off-0.01), 0.1, 0.04, facecolor=arm_color,
                        edgecolor='#999999', linewidth=1, zorder=9)
        ax.add_patch(arm)
        hand = Circle((hx+0.17, hy+ax_off+0.01), 0.025, facecolor='#E8E8E8',
                      edgecolor='#999999', linewidth=1, zorder=9)
        ax.add_patch(hand)
    # 腿
    for lx_off in [0.03, -0.03]:
        leg = Rectangle((hx+lx_off-0.02, hy-0.12), 0.04, 0.1, facecolor=arm_color,
                        edgecolor='#999999', linewidth=1, zorder=8)
        ax.add_patch(leg)
        foot = Rectangle((hx+lx_off-0.03, hy-0.22), 0.06, 0.03, facecolor='#333333',
                         edgecolor='#111111', linewidth=1, zorder=8)
        ax.add_patch(foot)

    # 人形标签
    ax.text(hx, hy-0.3, 'Companion Bot', ha='center', fontsize=8, color='#666666', weight='bold', zorder=12)

    # 标题
    ax.set_title('Bionic Quadruped Pet Companion - Home Interaction Simulation', fontsize=14, weight='bold',
                 color='#333333', pad=10)

    # 时间戳
    ax.text(0.02, 0.98, f'Time: {sim_time:.1f}s', transform=ax.transAxes,
            fontsize=10, color='#666666', va='top')

    # 图例
    legend_elements = [
        mpatches.Patch(color='#4A6FA5', label='Unitree Go1 Dog'),
        mpatches.Patch(color='#E8E8E8', label='Companion Humanoid'),
        mpatches.Patch(color='#C4956A', label='Flexible Sofa'),
        mpatches.Patch(color='#FF6B6B', label='Toy Ball'),
        mpatches.Patch(color='#87CEEB', label='Water Cup'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=7, ncol=3,
              framealpha=0.8, edgecolor='#CCCCCC')


# ============================================================
# 主程序
# ============================================================

def main():
    project_dir = Path(__file__).parent
    scene_path = project_dir / "assets" / "scene.xml"
    video_path = project_dir / "demo_video.mp4"

    print("Loading MuJoCo model...")
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)

    # 初始化传感器
    mujoco.mj_step(model, data)

    # 导入控制器
    sys.path.insert(0, str(project_dir))
    from simulation import DogController, HumanoidController

    dog = DogController(model, data)
    dog.mode = "walking"
    humanoid = HumanoidController(model, data)
    humanoid.behavior_phase = 3.0

    dt = model.opt.timestep
    duration = 20.0
    fps = 30
    total_frames = int(duration * fps)
    steps_per_frame = int(1.0 / (dt * fps))

    print(f"Generating {total_frames} frames at {fps}fps...")

    # 设置 matplotlib
    fig, ax = plt.subplots(figsize=(12, 8), dpi=120)
    writer = FFMpegWriter(fps=fps, codec='libx264', bitrate=2000)

    sim_time = 0.0
    with writer.saving(fig, str(video_path), dpi=120):
        for frame_i in range(total_frames):
            # 获取传感器数据
            def get_sensor(name):
                sid = model.sensor(name).id
                adr = model.sensor_adr[sid]
                dim = model.sensor(sid).dim[0]
                return data.sensordata[adr:adr+dim].copy()

            dog_pos = get_sensor("dog_pos")
            human_pos = get_sensor("human_pos")
            ball_pos = get_sensor("ball_pos")
            cup_pos = get_sensor("cup_pos")

            # 更新控制器
            dog.step(dt, dog_pos, human_pos, ball_pos)
            humanoid.step(dt, human_pos, dog_pos, ball_pos, cup_pos)

            # 物理步进
            for _ in range(steps_per_frame):
                for _ in range(5):
                    mujoco.mj_step(model, data)

            # 绘制
            draw_scene(ax, data, model, sim_time)
            writer.grab_frame()

            sim_time += steps_per_frame * dt

            if frame_i % 150 == 0:
                print(f"  Frame {frame_i}/{total_frames} ({sim_time:.1f}s)")

    plt.close()
    print(f"Video saved: {video_path}")

    # 确保 H.264
    if video_path.exists():
        import subprocess
        tmp = video_path.with_suffix(".tmp.mp4")
        video_path.rename(tmp)
        subprocess.run(["ffmpeg", "-y", "-i", str(tmp), "-c:v", "libx264",
                        "-preset", "fast", "-crf", "23", "-pix_fmt", "yuv420p",
                        "-movflags", "+faststart", str(video_path)],
                       capture_output=True)
        tmp.unlink()
        print("Transcoded to H.264")

    print("Done!")


if __name__ == "__main__":
    main()