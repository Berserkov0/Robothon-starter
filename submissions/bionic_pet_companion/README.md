# 仿生四足 + 人形宠物陪护机器人家庭互动仿真

## Project name
Bionic Quadruped & Humanoid Pet Companion Robot — Home Interaction Simulation

## Robot platform
- **Unitree Go1** 四足仿生机器狗（12 自由度，对角小跑步态）
- **小型人形陪护机器人**（13 自由度，双臂 + 双腿 + 头部）

## Task goal
在家庭场景中实现双机器人共生交互：机器狗跟随人形陪护机器人，共同完成递物、躲避障碍物、抚摸交互，展示柔性碰撞与全身触觉反馈。

## Technical approach

### 仿真引擎
基于 MuJoCo 3.x 物理引擎，使用 MJCF XML 格式描述场景。

### 场景构成
| 元素 | 描述 |
|:---|:---|
| 地板 | 木质纹理，含地毯区域（高摩擦） |
| 柔性沙发 | 软接触参数（solref/solimp）模拟坐垫形变，含独立物理靠枕 |
| 玩具球 | 滚动摩擦模型，可被机器人推动 |
| 水杯 | 半透明杯体 + 水面 |
| 障碍物 | 盒子、圆柱体，用于避障演示 |

### 控制器架构
- **机器狗控制器**：状态机驱动（idle → walking → touched → playing），对角小跑（trot）步态生成，支持速度调节与转向
- **人形机器人控制器**：周期性行为切换（站立 → 挥手 → 递物 → 休息），支持行走步态与目标跟随

### 触觉交互
- 机器狗：6 个触觉传感器（4 脚掌 + 身体 + 头部）
- 人形机器人：6 个触觉传感器（头部 + 双手 + 双脚 + 身体）
- 触碰后触发开心摇摆 / 跟随行为

## Core features
- 双机器人共生交互：机器狗自动跟随人形陪护机器人
- 对角小跑步态：Unitree Go1 风格四足运动
- 全身触觉传感器：触碰触发情感化反馈
- 柔性沙发形变碰撞：软接触参数模拟毛绒质感
- 多摄像机自动切换：追踪 / 前方 / 特写 / 俯视
- 递物行为演示：人形机器人伸臂递物姿态

## Highlights
- 柔性爪子接触：狗脚掌使用软 solref/solimp 参数，模拟肉垫柔性
- 毛绒玩具球：低刚度碰撞参数，模拟毛绒玩具滚动
- 情感化交互：被抚摸后机器人产生摇摆、下蹲等开心反应
- 完整的家庭场景：沙发、地毯、靠枕、玩具、水杯一应俱全

## Current limitations
- 机器狗模型为简化版本，12 自由度而非真实 Go1 的 12 电机 + 附加关节
- 人形机器人未实现精确抓取，递物为姿态演示
- 柔性沙发为软接触参数模拟，非真实有限元形变
- 触觉传感器为接触力检测，非分布式触觉阵列

## Future improvements
- 接入真实 Unitree Go1 URDF/MJCF 模型
- 增加强化学习步态控制器
- 实现精确抓取与物体搬运
- 增加语音/手势交互
- 引入视觉传感器（RGB-D 相机）实现目标检测

## How to run

### 环境要求
- Python 3.10+
- MuJoCo 3.0+
- NumPy
- OpenCV-Python

### 安装
```bash
cd submissions/bionic_pet_companion
pip install -r requirements.txt
```

### 运行仿真
```bash
# 有显示器环境
python3 simulation.py

# 无显示器环境（使用虚拟显示）
MUJOCO_GL=glfw xvfb-run -a python3 simulation.py
```

### 输出
- `demo_video.mp4`：1280×720 H.264 编码的演示视频（30 fps）

## Demo video
`demo_video.mp4`（31 秒，1280×720，H.264）

视频展示了完整的交互流程：机器狗从静止状态开始，跟随人形机器人移动，人形机器人周期性展示挥手、递物等行为，玩具球和水杯参与物理交互，摄像机在不同角度间自动切换。