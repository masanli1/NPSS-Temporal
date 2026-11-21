# NPSS-Burst

作者：[masanli1](https://github.com/masanli1) · 仓库：[NPSS-Burst](https://github.com/masanli1/NPSS-Burst)

[English](README.md)

> **把一段连续帧融合成一张干净照片。**
> 连续帧是互补的：内容相同、噪声独立。对齐后平均，噪声按 √N 下降，细节保留——
> 这是任何单帧降噪都做不到的。

核心纯 numpy，只有从视频取帧时才需要 ffmpeg。

## 流程

```
N 张连续帧
  ↓  逐帧仿射配准（分块相位相关，亚像素 + SSD 精修）
  ↓  鲁棒加权融合（中位数参考 → 高斯权重 → 加权平均）
  ↓  单帧降噪（NPSS）作为兜底
  ↓  时间一致性混合（运动区域回退单帧）
一张干净照片
```

## 实测

`python3 benchmark.py`：384×384 纹理场景，每帧随机位移 ≤ ±2 像素，AWGN σ=20。

| 帧数 | 单帧带噪 | 单帧降噪 | 直接平均 | **对齐融合** | 理想配准 |
|---|---|---|---|---|---|
| 4 | 21.34 | 24.60 | 23.89 | **25.57**（+0.97） | 26.90 |
| 8 | 19.02 | 22.71 | 23.25 | **23.00**（+0.29） | 27.17 |

「理想配准」用的是已知位移，所以它与对齐融合的差就是**配准误差的代价**：
4 帧 1.3 dB，8 帧 4.2 dB。

### 老实说

- 这组合成数据上，融合只比单帧降噪高 **0.3~1.0 dB**，8 帧时还不再提升——残余错位把平均糊掉了。
- 如果配准是准的，同样的帧能拿到 **+2.5~4.5 dB**。
  **瓶颈是配准精度，不是融合思路本身。**
- 真实手持连拍的运动比这组测试更小更平滑，效果应当介于两列之间，但**这一点没有实测**。

## 用法

```python
import numpy as np
from PIL import Image
import npss_burst as B

frames = np.stack([np.asarray(Image.open(f).convert("RGB"), np.float32) / 255.0
                   for f in sorted(paths)])
out, info = B.fuse(frames)
Image.fromarray((out * 255 + 0.5).astype(np.uint8)).save("clean.png")
```

命令行：

```bash
python3 demo.py --video clip.mp4 --time 3.0
python3 demo.py --frames burst/
python3 benchmark.py --frames 4 8 16 --sigma 20 50
```

## 它不是什么

- **不是多帧超分**：融合后的分辨率等于输入帧的分辨率。
- **不是视频降噪**：只从一小段连拍里出一张静帧，不做整段视频的时域滤波。
- **不是免配准的**：帧间位移超过几个像素，仿射模型就失效，融合会退化到单帧效果。

## 许可

双许可：学术/非商业免费（[LICENSE-ACADEMIC.md](LICENSE-ACADEMIC.md)），
商业使用需付费授权（[LICENSE-COMMERCIAL.md](LICENSE-COMMERCIAL.md)）。联系 `2459551532@qq.com`。
