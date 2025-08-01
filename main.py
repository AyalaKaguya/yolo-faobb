import numpy as np
from model import FullAngleOBB
from ultralytics import YOLO
from dataset import YOLOv8_OBB_Dataset, PieceClassificationAugmenter
from visualize import test_position_weights, visualize_direction_classification

# 加载数据集
dataset = YOLOv8_OBB_Dataset(
    'datasets/screw_long_obb_dataset',
    pixel_coordinate=True,
    xywhr_format=True,
)

# ============================================================
# 生成四向分类数据集
# ============================================================

piece_dataset_path = './datasets/screw_long_piece_classification'

# 基础配置
PieceClassificationAugmenter().build(
    dataset, piece_dataset_path)
# 配置1: 轻微偏移和旋转，不随机旋转
PieceClassificationAugmenter(
    random_offset=0.04,
    random_rotate=10,
    random_twist=False
).build(dataset, piece_dataset_path, file_prefix='rd0_')
# 配置2: 较大偏移和旋转，开启随机旋转
PieceClassificationAugmenter(
    random_offset=0.08,
    random_rotate=5,
    random_twist=True
).build(dataset, piece_dataset_path, file_prefix='rd1_')
# 配置3: 中等偏移，减少边距和填充
PieceClassificationAugmenter(
    random_offset=0.06,
    random_rotate=5,
    padding=10,
    margin=10,
    random_twist=True
).build(dataset, piece_dataset_path, file_prefix='rd2_')
# 配置4: 无边距和填充，轻微旋转
PieceClassificationAugmenter(
    padding=0,
    margin=0,
    random_rotate=2,
    random_twist=True
).build(dataset, piece_dataset_path, file_prefix='rd3_')
# 配置5: 最大偏移，最小旋转
PieceClassificationAugmenter(
    random_offset=0.1,
    random_rotate=1,
    random_twist=True
).build(dataset, piece_dataset_path, file_prefix='rd4_')
# 你可以根据需要添加更多配置

# ============================================================
# 训练OBB和四向分类模型
# ============================================================


model_cls = YOLO("yolo11n-cls.pt")
model_obb = YOLO("yolo11n-obb.pt")

# 注意不要修改piece_dataset下分类的名称，如果乱了重映射会很麻烦
model_cls.train(
    data=piece_dataset_path,
    imgsz=640,
)

# 注意将ultralytics的数据集目录临时改到项目下，不然这一步会报错
model_obb.train(
    data="./datasets/screw_long_obb_dataset.yaml",
    imgsz=640,
)

# ============================================================
# 预测、可视化与评价
# ============================================================

# 使用训练好的模型进行预测，支持输入模型路径
model = FullAngleOBB(model_cls, model_obb)

model.position_weights = np.array([4.0, 0.25, 4.0, 0.25]) # 强调左右对结果的影响更大
model.padding = 5 # 四向分类模型前端分割器填充正方形时留出的内边距
model.margin = 0  # 四向分类模型前端分割器分割图像时额外的边距大小

img_path = "./datasets/screw_long_obb_dataset/images/a0afc5a8-screws_001.png"

result = model.predict(img_path, obb_conf=0.7)

visualize_direction_classification(img_path, result).show() # 查看两个阶段模型的各自的置信度和效果

test_position_weights(model, result['cls_predictions'][0], result['cls_confidences'][0]).show()  # 测试不同位置权重对角度匹配结果的影响

# 评价函数暂时不开源