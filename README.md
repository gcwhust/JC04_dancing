# JC04 Dancing / 小楚机器人舞蹈仿真

[**Homepage · 项目主页**](https://gcwhust.github.io/JC04_dancing/) · [演示视频](https://gcwhust.github.io/JC04_dancing/#swan-lake) · [设计说明 PDF](https://gcwhust.github.io/JC04_dancing/media/design.pdf) · [流程图 PDF](https://gcwhust.github.io/JC04_dancing/media/workflow.pdf)

**2026 全国大学生机器人与算法大赛 · 机器人舞蹈赛道**

参赛队伍：ROBOT261937J

本项目基于 MuJoCo 实现小楚 JC04 轮足机器人的《天鹅湖》舞蹈仿真。双臂展翼、前伸与屈肘配合髋膝起伏，通过双轮滑行和差速转向完成连续的舞蹈动作。

[![天鹅湖舞蹈演示](docs/media/swan-lake-preview.gif)](https://gcwhust.github.io/JC04_dancing/#swan-lake)

视频时长 **99 秒**，分辨率 **1920 × 1080**，帧率 **30 FPS**。点击上方预览可观看有声视频。

## 机器人与控制方案

![小楚机器人仿真模型](docs/model-initial.png)

轮足机器人 JC04 具有 **12 个主动自由度**：下肢包括左右髋关节、左右膝关节和左右主动轮，共 6 个自由度；上肢包括左右肩部俯仰、肩部横滚及肘部俯仰关节，共 6 个自由度。头部下方为躯干，两侧为双臂；髋膝连杆连接躯干与主动轮，辅助轮用于四轮模式的支撑。

仿真模型采用 **jc04_desc4 V1.3**，包含 URDF、MJCF 和配套网格。机身采用自由浮动基座，运动由关节力矩和轮地接触共同决定。10 个臂腿关节使用 PD 力矩控制，两个主轮使用轮速闭环控制，辅助轮被动滚动。物理步长为 **0.0005 秒**，摄像机跟随机身水平位置。

动作以 CSV 记录，包含 `time_s`、10 个臂腿关节角度和 2 个主轮角速度通道。角度单位为度，轮速单位为 rad/s；关节轨迹采用五次样条插值。完整时间轴由 2 秒准备、94 秒舞蹈、2 秒回位和 1 秒保持组成。

## 环境安装

| 项目 | 配置 |
|---|---|
| 操作系统 | Ubuntu 22.04 x86_64 |
| Python | 3.10.22 |
| 仿真平台 | MuJoCo 3.3.7 |
| 离屏渲染 | EGL |
| 视频与音频 | H.264 / AAC |

```bash
git clone https://github.com/gcwhust/JC04_dancing.git
cd JC04_dancing
sudo apt-get update
sudo apt-get install -y libegl1 libgl1 libglfw3 libosmesa6
conda env create -f environment.yml
conda activate jc04_dancing
python -m pip install .
```

依赖版本见 [environment.yml](environment.yml) 和 [requirements-lock.txt](requirements-lock.txt)。机器人模型、动作数据、配乐和字体均随安装包提供，视频编码使用 `imageio-ffmpeg` 随附的 FFmpeg。

## 运行仿真

以下命令在仓库根目录执行。

```bash
python -m jc04_dancing list
MUJOCO_GL=egl python -m jc04_dancing check --dance swan_lake --mode free
```

`check` 执行完整时间轴，并输出关节跟踪误差、轮速、机身位移和稳定性统计。有图形桌面时，可打开交互窗口：

```bash
python -m jc04_dancing play --dance swan_lake --mode free
```

交互窗口用于查看动作；有声视频通过下面的录制与配乐步骤获得。

## 录制有声视频

### 1. 录制舞蹈画面

```bash
MUJOCO_GL=egl python -m jc04_dancing record \
  --dance swan_lake --mode free \
  --output outputs/swan_lake_run/dance.mp4
```

默认录制完整的 99 秒舞蹈，共 2970 帧，同时保存运行数据 `run.json` 和截图 `stills/`。输出目录须为空。服务器离屏渲染需要可用的 EGL/显卡驱动，模型加载与录制通常需要数分钟。

### 2. 加入配乐

```bash
python -m jc04_dancing add-audio \
  --run-dir outputs/swan_lake_run \
  --output outputs/swan_lake_audio
```

有声视频位于 `outputs/swan_lake_audio/dance.mp4`，可使用浏览器或视频播放器观看。配乐选自柴可夫斯基《天鹅湖》，从视频第 2 秒开始，持续 94 秒；视频画面保持原有编码，音轨采用 AAC 立体声。配乐输出目录同样须为空。

## 核心文件

| 文件 | 内容 |
|---|---|
| [choreography.py](src/jc04_dancing/choreography.py) | 舞蹈姿态、段落与时间参数 |
| [swan_lake.csv](src/jc04_dancing/assets/motions/swan_lake.csv) | 臂、腿和轮部的完整动作数据 |
| [robot.yaml](src/jc04_dancing/configs/robot.yaml) | 关节控制、轮速控制、场景与摄像机参数 |
| [motion.py](src/jc04_dancing/motion.py) | 动作读取、插值和时间轴 |
| [simulation.py](src/jc04_dancing/simulation.py) | MuJoCo 仿真与闭环控制 |
| [recording.py](src/jc04_dancing/recording.py) | 视频录制与运行数据 |
| [audio.py](src/jc04_dancing/audio.py) | 视频配乐与音轨处理 |

## 资料与来源

[设计说明](docs/media/design.pdf) · [运行流程图](docs/media/workflow.pdf) · [模型说明](docs/model-conversion.md)

机器人模型来源为 **jc04_desc4 V1.3**。配乐为柴可夫斯基《天鹅湖》节选，录音来源和使用信息见[配乐说明](docs/audio-provenance.md)。模型、录音和字体保留各自许可，详见[素材声明](src/jc04_dancing/assets/licenses/ASSET-NOTICES.md)。
