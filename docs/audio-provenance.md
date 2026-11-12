# 配乐来源与合成

`record` 生成无声仿真录像；`add-audio` 将音乐加入已完成的录像。当前新年成片使用包内 `assets/audio/happy_new_year.mp3`，保留视频的 H.264 码流、全部帧与动作时间轴，声音编码为 AAC。

## 来源

音频从快照 `2d224d13d624e07102a2c6eedbaf111694555e16` 按原字节提取；资产相对来源路径与校验清单见包内 `assets/provenance.json`。源文件与包内文件的 SHA-256 均为：

```text
a6afc7649b5c4e2833368abccd7730829e6989e17fea50120ae8e51386619925
```

模型或软件包的 Apache-2.0 许可不自动适用于这段录音或音乐作品。音乐身份、权利人及额外授权尚未核实；相关说明保存在 `assets/licenses/ASSET-NOTICES.md`。

## 默认合成参数

| 参数 | 值 |
|---|---|
| 音乐源起点 | 0秒 |
| 视频内开始时刻 | 2秒，与舞蹈主体起点相同 |
| 播放速度 | 原速，不循环 |
| 淡入 / 淡出 | 0.4秒 / 0.8秒 |
| 音频输出 | AAC、48,000 Hz、立体声 |
| 视频总时长 | 83.333333秒 |

FFmpeg 7.0.2 完整解码音源得到约76.32秒。新年舞蹈主体为 `1880/24 = 78.333333` 秒，音乐约短2.013333秒。因此音乐约在完整视频第78.32秒结束，剩余舞蹈与收尾阶段使用静音补足全片。合成不调整编舞时序。

```bash
python -m jc04_dancing add-audio --run-dir outputs/reproduced --output outputs/reproduced_audio
```

输出目录包含带音轨的 `dance.mp4`、更新后的 `run.json` 与截图证据。元数据记录音源哈希、起点、偏移、淡入淡出、实际时长与解码检查。合成流程核验视频帧保持一致，并检查音轨可完整解码且有效音乐区段不是静音。

## 验证范围

音视频技术检查已通过，节拍或旋律与动作的配合尚未完成主观验收，D04 保持待验证。包内原始来源清单仍保留提取时的 `candidate_pending_audio_review` 状态；实际成片是否添加音轨及其参数以对应的 `run.json` 为准。

豆豆龙尚未提供对应音乐，当前85秒完整成片为无声 H.264 视频。新年与豆豆龙均可独立进行仿真与无声录制；上述默认合成参数及已完成的音频检查仅对应新年。

为豆豆龙加入配乐时，请提供对应音乐文件并明确指定 `--audio`，例如：

```bash
python -m jc04_dancing add-audio \
  --run-dir outputs/doudoulong \
  --audio /path/to/doudoulong-music.mp3 \
  --output outputs/doudoulong_audio
```

该操作不会完成音乐权利或节拍配合的验证，合成结果应另行试听核对。
