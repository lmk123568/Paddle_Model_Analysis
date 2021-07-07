

![acuowkoa](C:\Users\Mike\Desktop\ppma\source\acuowkoa.png)

# 📦 Paddle Model Analysis

[![](https://img.shields.io/badge/Paddle-2.0-blue)](https://www.paddlepaddle.org.cn/)[![Documentation Status](https://img.shields.io/badge/Tutorial-最新-brightgreen.svg)](https://paddlepaddle.org.cn/documentation/docs/zh/guides/index_cn.html)[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)![](https://img.shields.io/badge/version-0.1-yellow)

这是基于飞桨开发的工具包，以极简主义为特色，用于对分类任务模型进行快速分析![qaeewagy](C:\Users\Mike\Desktop\ppma\source\qaeewagy.png)

目前所支持的功能有：![oqrhsqot](C:\Users\Mike\Desktop\ppma\source\oqrhsqot.gif)

- [x] ImageNet 上快速验证模型
- [x] 测试图片 Top5 类别
- [x] 测试模型 Param、Thoughtout
- [x] CAM (Class Activation Mapping)
- [x] TTA (Test Time Augmention)
- [ ] 计划中:clipboard: ...

## 安装

```bash
pip install ppma -i https://pypi.python.org/simple
```

## 快速开始

> Note：推荐去 AI Studio 在线免费运行项目 [PPMA 快速指南](https://aistudio.baidu.com/aistudio/projectdetail/2143665)

* ImageNet 上快速验证模型

当训练了新的模型后，或者复现了某个模型，我们需要在 ImageNet 数据集上验证性能，先准备数据集结构如下

```bash
data/ILSVRC2012
		├─ ILSVRC2012_val_00000001.JPEG
		├─ ILSVRC2012_val_00000002.JPEG
		├─ ILSVRC2012_val_00000003.JPEG
		├─ ...
		├─ ILSVRC2012_val_00050000.JPEG
		└─ val.txt   # target
```

准备好数据集后，运行以下代码

```python
import ppma
import paddle

model = paddle.vision.models.resnet50(pretrained=True)	# 可以替换自己的模型
data_path = "data/ILSVRC2012"	                        # 数据路径

ppma.imagenet.val(model, data_path)
```

* 测试图片 Top5 类别

```python
import ppma
import paddle

img_path = 'test.jpg'                                    # 图片路径
model = paddle.vision.models.resnet50(pretrained=True)   # 可以替换自己的模型

ppma.imagenet.test_img(model, img_path)
```

* 测试模型 Param、Thoughtout

```python
import ppma
import paddle

model = paddle.vision.models.resnet50()   # 可以替换自己的模型

# Params -- depend model
param = ppma.tools.param(model)
print('Params：{:,}'.format(param))

# Thoughtout -- depend model and resolution
ppma.tools.throughput(model, image_size=224)
```

* CAM (Class Activation Mapping)

```python
import paddle
import matplotlib.pyplot as plt
from ppma import cam

img_path = 'img1.jpg'                                      # 图片路径
model = paddle.vision.models.resnet18(pretrained=True)     # 模型定义
target_layer = model.layer4[-1]                            # 提取模型某层的激活图
cam_extractor = cam.GradCAMPlusPlus(model, target_layer)   # 支持 GradCAM、XGradCAM、GradCAM++

# 提取激活图
activation_map = cam_extractor(img_path, label=None)   
plt.imshow(activation_map)
plt.axis('off')
plt.show()

# 与原图融合
cam_image = cam.overlay(img_path, activation_map)   
plt.imshow(cam_image)
plt.axis('off')
plt.show()
```

* TTA (Test Time Augmention)

```python
import paddle
import ppma
import ppma.tta as tta

model = paddle.vision.resnet18(pretrained=True)
model_tta = tta.ClassTTA(model, tta.aliases.hflip_transform())   # 生成 TTA 模型

ppma.imagenet.val(model_tta, "data/ILSVRC2012")
```

## 设计的哲学

在阅读使用 Fastai、Keras、sklearn 等简洁的包，拥有大量的使用体验后，总结对于一个工具是否简洁高效要看以下两点

* 命名的艺术

过度简单的命名容易与用户自定义变量命名冲突，复杂的命名不容易用户理解记忆，所以需要进行良好的 Trade-off，既能保证命名易懂好理解又不复杂，又能保证命名不会出现在常见的用户自定义命名里

```python
# 针对 ImageNet 数据集进行验证的函数
# 当前方案
ppma.imagenet.val(model, img_path)     # 简洁优雅

# 曾经方案
ppma.ILSVRC2012.val(model, img_path)         # ILSVRC2012 太长不方便记忆
ppma.imagenet2012.validate(model, img_path)  # 变量命名有些冗余，可以缩短而不影响理解
...
```

* 结构的设计

```python
# 本项目设计参考 Box 思想，用户只需要准备需要的放入函数里一键运行即可
#
#  model \
#  img   － Box － Result
#  ...   /
#    
```

