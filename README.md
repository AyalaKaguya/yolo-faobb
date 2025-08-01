# YOLO-FAOBB

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7.1+-red.svg)](https://pytorch.org/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-8.3.168+-green.svg)](https://github.com/ultralytics/ultralytics)

YOLO-FAOBB（Full-Angle Oriented Bounding Box）是一个基于 YOLO11-OBB 的两阶段全角度旋转边界框检测模型，旨在解决传统OBB模型在角度预测上的限制，实现0°-360°的全角度预测。

## 项目特色

- **全角度预测**：突破传统OBB模型0°-90°-180°的角度限制，实现360°全角度检测
- **两阶段架构**：结合OBB检测和四向分类，提升角度预测精度
- **数据增强**：提供丰富的数据增强策略，提升模型泛化能力
- **易于使用**：提供完整的数据处理和模型训练工具链

## 技术架构

### 核心思想

本项目采用两阶段设计：

1. **第一阶段**：使用YOLO11-OBB模型进行旋转边界框检测
2. **第二阶段**：对检测到的边界框进行截取，分为四个方向（上、下、左、右），通过四向分类确定准确的方向角度

### 工作流程

```
输入图像 → YOLO11-OBB检测 → 边界框截取 → 四向分类 → 置信度融合 → 最终预测
```

## 项目结构

```
yolo-faobb/
├── dataset.py                          # 数据集处理和数据增强
├── how_to_patch_yolo.ipynb            # YOLO框架修改指南（可选）
├── pyproject.toml                      # 项目配置
├── README.md                           # 项目文档
└── datasets/
    ├── screw_long_obb_dataset.yaml     # 数据集配置
    └── screw_long_obb_dataset/         # 示例数据集
        ├── classes.txt                 # 类别定义
        ├── images/                     # 图像文件
        └── labels/                     # 标签文件
```

## 🚀 快速开始

### 环境要求

- Python >= 3.10
- PyTorch >= 2.7.1
- Ultralytics >= 8.3.168

除此之外还有numpy、matplotlib、pillow、opencv-python、tqdm这几个依赖库

### 安装

请先自行安装合适的[PyTorch](https://pytorch.org/)

```bash
# 克隆项目
git clone https://github.com/AyalaKaguya/yolo-faobb.git
cd yolo-faobb

# 安装依赖
pip install -e .
```

### 数据准备

本项目支持YOLO格式的OBB数据集。数据集结构如下：

```
dataset/
├── classes.txt          # 类别名称，每行一个类别
├── images/              # 图像文件夹
│   ├── train/           # 训练图像
│   └── val/             # 验证图像
└── labels/              # 标签文件夹
    ├── train/           # 训练标签
    └── val/             # 验证标签
```

标签格式为：`class_id x1 y1 x2 y2 x3 y3 x4 y4`（归一化坐标）

### 使用示例

#### 1. 加载OBB数据集

```python
from dataset import YOLOv8_OBB_Dataset

# 创建数据集实例
dataset = YOLOv8_OBB_Dataset(
    'datasets/screw_long_obb_dataset',
    pixel_coordinate=True,
    xywhr_format=True,
)

# 批量加载训练和验证集
datasets = YOLOv8_OBB_Dataset.load_yolo_obb('datasets/screw_long_obb_dataset')
train_dataset = datasets['train']
val_dataset = datasets['val']
```

#### 2. 生成四向分类数据集

```python
from dataset import PieceClassificationAugmenter

# 基础配置
augmenter = PieceClassificationAugmenter()
augmenter.build(dataset, 'output/classification_dataset')

# 带数据增强的配置
augmenter = PieceClassificationAugmenter(
    random_offset=0.08,      # 随机偏移
    random_rotate=5,         # 随机旋转角度
    random_twist=True,       # 随机90度旋转
    margin=20,               # 边距扩展
    padding=20,              # 填充大小
    target_size=224          # 目标图像尺寸
)
augmenter.build(dataset, 'output/augmented_dataset', file_prefix='aug_')
```

## API 参考

### YOLOv8_OBB_Dataset

OBB数据集加载器，支持多种数据格式和坐标系统。

**参数：**
- `path`: 数据集路径
- `path_prefix`: 路径前缀（用于区分train/val）
- `pixel_coordinate`: 是否返回像素坐标
- `xywhr_format`: 是否转换为xywhr格式

### PieceClassificationAugmenter

零件分类数据增强器，用于生成四向分类训练数据。

**主要方法：**
- `__init__()`: 初始化增强参数
- `build()`: 构建分类数据集
- `get_piece_images()`: 提取四向图像
- `make_square()`: 生成正方形图像

**增强参数：**
- `margin`: 扩展边距（默认20像素）
- `random_offset`: 随机偏移比例（默认0.0）
- `random_rotate`: 随机旋转角度范围（默认0.0）
- `padding`: 填充大小（默认20像素）
- `target_size`: 目标图像尺寸（默认224）
- `fill_color`: 填充颜色（默认灰色）
- `random_twist`: 是否随机90度旋转（默认False）

## 🔧 高级用法

### 数据增强策略

项目提供多种数据增强配置：

```python
# 轻微增强
light_augmenter = PieceClassificationAugmenter(
    random_offset=0.04,
    random_rotate=10,
    random_twist=False
)

# 强增强
strong_augmenter = PieceClassificationAugmenter(
    random_offset=0.1,
    random_rotate=15,
    random_twist=True,
    margin=10,
    padding=10
)

# 无边距增强（适合紧凑物体）
tight_augmenter = PieceClassificationAugmenter(
    random_offset=0.06,
    random_rotate=5,
    margin=0,
    padding=0,
    random_twist=True
)
```

### 自定义数据处理

```python
# 获取单个样本
image, boxes, image_id = dataset[0]

# 处理单个边界框
for class_id, annotation in boxes:
    # annotation 格式：[cx, cy, w, h, angle] (if xywhr_format=True)
    # 或者 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]] (if xywhr_format=False)
    piece_images = augmenter.get_piece_images(image, annotation)
    # piece_images: [right, down, left, up] 四个方向的图像
```

## 🛠️ 开发指南

### 扩展支持新的数据格式

继承 `YOLOv8_OBB_Dataset` 类并重写相应方法：

```python
class CustomOBBDataset(YOLOv8_OBB_Dataset):
    def __getitem__(self, idx):
        # 自定义数据加载逻辑
        pass
```

### 添加新的增强策略

继承 `PieceClassificationAugmenter` 类：

```python
class AdvancedAugmenter(PieceClassificationAugmenter):
    def get_piece_images(self, image, ann):
        # 添加自定义增强逻辑
        return super().get_piece_images(image, ann)
```

## 贡献

欢迎提交 Issue 和 Pull Request！

## 许可证

本项目采用 [AGPL-3.0 License](LICENSE)。

## 联系方式

- 项目地址：[https://github.com/AyalaKaguya/yolo-faobb](https://github.com/AyalaKaguya/yolo-faobb)
- Issue 反馈：[https://github.com/AyalaKaguya/yolo-faobb/issues](https://github.com/AyalaKaguya/yolo-faobb/issues)

---

*本项目旨在推进旋转目标检测技术的发展，如果对您的研究有帮助，请考虑给项目点个星⭐*
