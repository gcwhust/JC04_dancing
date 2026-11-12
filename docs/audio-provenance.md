# 《天鹅湖》配乐

采用柴可夫斯基《天鹅湖》Op.20 第二幕主题录音的连续节选。录音来源为 [Wikimedia Commons / European Archive / Musopen](https://commons.wikimedia.org/wiki/File:Tchaikovsky_-_Swan_Lake_Op.20_-_Act_II_Pt.1.ogg)，来源页标注 [Public Domain Mark 1.0](https://creativecommons.org/publicdomain/mark/1.0/)。原页面未注明具体乐团、指挥和录制日期，本项目不补造这些信息。

节选原录音 **55.2–149.2 秒**，共 **94 秒**，不变速、不循环、不调换段落。音量乘以 2，开头淡入 0.4 秒、结尾淡出 2.8 秒，已写入包内 MP3。默认合成不再重复淡入淡出；配乐从视频第 2 秒开始，到第 96 秒结束。

编舞时间点根据波形能量与频谱选取：主题回应、低力度段、增强段、高潮、回落与致意。它们是人工选择的音乐结构提示，不是经乐谱校验的小节线。动作生成器使用同一份时间标记，音乐和动作保持同一原速时间轴；尚未进行人工试听评价。

完整来源链接、原文件及节选 SHA256、音频参数、可复现 FFmpeg 参数和各时间点依据，见 [swan_lake_source.json](../src/jc04_dancing/assets/audio/swan_lake_source.json)。

合成输出为 AAC 48kHz 立体声。工具逐帧验证合成前后视频像素一致，实际解码声音并检测活动段非静音、准备段及片尾静音。数值检查不代替艺术效果评价。
