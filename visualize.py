import cv2
from matplotlib import pyplot as plt
import numpy as np
from dataset import YOLOv8_OBB_Dataset
from model import FullAngleOBB


def visualize_direction_classification(image_path, result, figsize=(12, 6)):
    """
    可视化指定目标的四个方向切片分类结果

    Args:
        image_path: 图像路径
        result: predict方法的返回结果
        target_idx: 要可视化的目标索引（从0开始）
    """

    # 读取图像
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # type: ignore

    # 创建图像显示区域
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # 绘制原始OBB结果
    axes[0].imshow(image)
    axes[0].set_title('原始OBB检测结果', fontsize=14)
    axes[0].axis('off')

    # 绘制修正后的OBB结果
    axes[1].imshow(image)
    axes[1].set_title('角度修正后的OBB结果', fontsize=14)
    axes[1].axis('off')

    colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan']

    for i, (orig_box, corr_box, a_conf, obb_conf) in enumerate(zip(result['original_obb_boxes'], result['obb_boxes'], result['andle_confidences'], result['obb_confidences'])):
        color = colors[i % len(colors)]

        # 转换为四个顶点坐标
        orig_pts = YOLOv8_OBB_Dataset.xywhr2xyxyxyxy(orig_box)
        corr_pts = YOLOv8_OBB_Dataset.xywhr2xyxyxyxy(corr_box)

        # 绘制原始框
        orig_pts_closed = np.vstack([orig_pts, orig_pts[0]])  # 闭合多边形
        axes[0].plot(orig_pts_closed[:, 0], orig_pts_closed[:, 1],
                     color=color, linewidth=2, label=f'目标 {i+1}')
        axes[0].fill(orig_pts_closed[:, 0], orig_pts_closed[:, 1],
                     color=color, alpha=0.2)

        # 绘制修正后的框
        corr_pts_closed = np.vstack([corr_pts, corr_pts[0]])  # 闭合多边形
        axes[1].plot(corr_pts_closed[:, 0], corr_pts_closed[:, 1],
                     color=color, linewidth=2, label=f'目标 {i+1}')
        axes[1].fill(corr_pts_closed[:, 0], corr_pts_closed[:, 1],
                     color=color, alpha=0.2)

        # 添加角度标注
        cx, cy = orig_box[0], orig_box[1]
        orig_angle = np.degrees(orig_box[4])
        corr_angle = np.degrees(corr_box[4])

        # 原始角度标注
        axes[0].annotate(f'{orig_angle:.1f}°@{obb_conf:.2f}',
                         xy=(cx, cy), xytext=(cx+20, cy-20),
                         fontsize=10, color=color, weight='bold',
                         bbox=dict(boxstyle="round,pad=0.3",
                                   facecolor='white', alpha=0.7),
                         arrowprops=dict(arrowstyle='->', color=color))

        # 修正后角度标注
        axes[1].annotate(f'{corr_angle:.1f}°@{a_conf:.2f}',
                         xy=(cx, cy), xytext=(cx+20, cy-20),
                         fontsize=10, color=color, weight='bold',
                         bbox=dict(boxstyle="round,pad=0.3",
                                   facecolor='white', alpha=0.7),
                         arrowprops=dict(arrowstyle='->', color=color))

        # 添加方向箭头（指向物体的头部方向）
        arrow_length = 60
        color = 'white'

        # 原始方向箭头
        orig_dx = arrow_length * np.cos(orig_box[4])
        orig_dy = arrow_length * np.sin(orig_box[4])
        axes[0].arrow(cx, cy, orig_dx, orig_dy,
                      head_width=8, head_length=8, fc=color, ec=color, alpha=0.8)

        # 修正后方向箭头
        corr_dx = arrow_length * np.cos(corr_box[4])
        corr_dy = arrow_length * np.sin(corr_box[4])
        axes[1].arrow(cx, cy, corr_dx, corr_dy,
                      head_width=8, head_length=8, fc=color, ec=color, alpha=0.8)

    # 添加图例
    axes[0].legend(loc='upper right')
    axes[1].legend(loc='upper right')

    plt.tight_layout()

    return fig

def test_position_weights(model: FullAngleOBB, input_vector: np.ndarray, input_vector_conf: np.ndarray):
    """测试不同位置权重对角度匹配结果的影响"""
    # 不同的位置权重配置
    weight_configs = [
        {
            "name": "均等权重",
            "weights": np.array([1.0, 1.0, 1.0, 1.0]),
            "description": "所有位置权重相等"
        },
        {
            "name": "强调上下",
            "weights": np.array([0.5, 2.0, 0.5, 2.0]),
            "description": "上下方向权重更高"
        },
        {
            "name": "强调左右",
            "weights": np.array([2.0, 0.5, 2.0, 0.5]),
            "description": "左右方向权重更高"
        },
        {
            "name": "只看左侧",
            "weights": np.array([0.1, 0.1, 3.0, 0.1]),
            "description": "主要依赖左侧分类"
        },
        {
            "name": "高置信度优先",
            "weights": input_vector_conf,  # 使用置信度作为权重
            "description": "置信度高的位置权重更大"
        },
    ]

    angle_names = ["0°", "90°", "180°", "270°"]

    results = []

    for config in weight_configs:
        # 调用匹配算法
        model.position_weights = config['weights']
        angle_correction, correction_confidence = model.match_angle_matrix(
            input_vector, input_vector_conf
        )

        angle_degrees = np.degrees(angle_correction)
        angle_idx = int(angle_degrees // 90) % 4

        results.append({
            'name': config['name'],
            'weights': config['weights'],
            'angle': angle_degrees + 360 if angle_degrees < 0 else angle_degrees,
            'confidence': correction_confidence
        })

    # 可视化对比
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 图1: 角度选择对比
    config_names = [r['name'] for r in results]
    angles = [r['angle'] for r in results]
    confidences = [r['confidence'] for r in results]

    colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold', 'plum']
    bars1 = ax1.bar(config_names, angles, color=colors, alpha=0.7)
    ax1.set_ylabel('角度修正 (度)')
    ax1.set_title('不同位置权重下的角度选择')
    ax1.set_ylim(0, 360)

    # 添加数值标签
    for bar, angle in zip(bars1, angles):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 5,
                 f'{angle:.0f}°', ha='center', va='bottom')

    # 添加角度线
    for angle_deg, angle_name in enumerate([0, 90, 180, 270]):
        ax1.axhline(y=angle_deg*90, color='gray', linestyle='--', alpha=0.5)
        ax1.text(len(config_names)-0.5, angle_deg*90, angle_names[angle_deg],
                 ha='right', va='center', fontsize=9, alpha=0.7)

    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')

    # 图2: 置信度对比
    bars2 = ax2.bar(config_names, confidences, color=colors, alpha=0.7)
    ax2.set_ylabel('修正置信度')
    ax2.set_title('不同位置权重下的置信度')
    ax2.set_ylim(0, 1)

    # 添加数值标签
    for bar, conf in zip(bars2, confidences):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{conf:.3f}', ha='center', va='bottom')

    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')

    fig.tight_layout()
    return fig
    
    