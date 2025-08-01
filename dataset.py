from pathlib import Path
import random

import cv2
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm


class YOLOv8_OBB_Dataset(Dataset):
    """
    YOLOv8 旋转检测框数据集

    Args:
        path (str or Path): 数据集路径，包含 'images' 和 'labels' 子目录
        path_perfix (str): 图像和标签文件的前缀路径，用于区分训练和验证集
        pixel_coordinate (bool): 是否使用像素坐标
        int_pixel_coordinate (bool): 是否将像素坐标转换为整数
        require_pliiow_image (bool): 是否返回Pillow图像对象
        xywhr_format (bool): 是否使用xywhr格式（中心点坐标、宽度、高度、旋转角度）
    """

    def __init__(
        self,
        path: Path | str,
        path_perfix: str = '',
        pixel_coordinate=False,
        int_pixel_coordinate=True,
        require_pliiow_image=False,
        xywhr_format=False,
    ) -> None:
        path = path if isinstance(path, Path) else Path(path)
        # 读取分类
        with open(path.joinpath('classes.txt'), 'r', encoding='utf-8') as f:
            self.classes = [line.strip() for line in f.readlines()]

        self.image_files = list(
            path
            .joinpath('images')
            .joinpath(path_perfix)
            .glob('*.png')
        )
        self.label_files = list(
            path
            .joinpath('labels')
            .joinpath(path_perfix)
            .glob('*.txt')
        )

        self.path = path
        self.path_perfix = path_perfix
        self.pixel_coordinate = pixel_coordinate
        self.int_pixel_coordinate = int_pixel_coordinate
        self.require_pliiow_image = require_pliiow_image
        self.xywhr_format = xywhr_format
        if len(list(self.image_files)) != len(list(self.label_files)):
            raise ValueError("图像和标签文件数量不匹配")

    def __len__(self):
        return len(list(self.image_files))

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        label_file = self.label_files[idx]
        id_image = image_file.stem  # 获取图像文件名（不带扩展名）

        image = cv2.imread(str(image_file), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"无法读取图像文件: {image_file}")

        with open(label_file, 'r', encoding='utf-8') as f:
            labels = [line.strip().split() for line in f.readlines()]

        img_w, img_h = image.shape[1], image.shape[0]

        # 转换标签格式
        boxes = []
        for label in labels:
            class_id = int(label[0])
            x1, y1, x2, y2, x3, y3, x4, y4 = map(float, label[1:])
            ann = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
            if self.pixel_coordinate:
                ann = ann * np.array([img_w, img_h])
            if self.int_pixel_coordinate:
                ann = ann.astype(int)
            if self.xywhr_format:
                ann = self.xyxyxyxy2xywhr(ann)
            boxes.append((class_id, ann))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换为RGB格式
        if self.require_pliiow_image:
            image = Image.fromarray(image)
        return image, boxes, id_image

    def __str__(self):
        return f"YOLOv8_OBB_DS_Part(path={self.path}, path_perfix={self.path_perfix}, len={len(self)})"

    def __repr__(self):
        return self.__str__()

    @staticmethod
    def xyxyxyxy2xywhr(x: np.ndarray) -> np.ndarray:
        pts = x.reshape(-1, 2)
        cx = np.mean(pts[:, 0])
        cy = np.mean(pts[:, 1])
        pt1, pt2, pt3, pt4 = pts
        w = np.sqrt(np.sum((pt2 - pt3) ** 2))
        h = np.sqrt(np.sum((pt4 - pt3) ** 2))
        angle = np.arctan2((pt1-pt4)[1], (pt1-pt4)[0])
        return np.asarray([cx, cy, w, h, angle])

    @staticmethod
    def xywhr2xyxyxyxy(x: np.ndarray) -> np.ndarray:
        ctr = x[:2]
        w, h, angle = x[2], x[3], x[4]
        cos_value, sin_value = np.cos(angle), np.sin(angle)
        vec1 = np.array([w / 2 * cos_value, w / 2 * sin_value])
        vec2 = np.array([-h / 2 * sin_value, h / 2 * cos_value])
        pt1 = ctr + vec1 + vec2
        pt2 = ctr + vec1 - vec2
        pt3 = ctr - vec1 - vec2
        pt4 = ctr - vec1 + vec2
        return np.array([pt1, pt2, pt3, pt4])

    @classmethod
    def load_yolo_obb(cls, path: str, **kwargs):
        """加载YOLO格式的OBB数据集"""
        train_dataset = cls(path, path_perfix='train', **kwargs)
        val_dataset = cls(path, path_perfix='val', **kwargs)
        # test_dataset = cls(path, path_perfix='test', **kwargs)
        return dict(
            train=train_dataset,
            val=val_dataset,
            # test=test_dataset,
            classes=train_dataset.classes
        )


class PieceClassificationAugmenter:
    """
    零件分类数据增强器

    用于从OBB数据集中提取零件图像并生成分类数据集
    """

    def __init__(
        self,
        margin: int = 20,
        random_offset: float = 0.0,
        random_rotate: float = 0.0,
        padding: int = 20,
        target_size: int = 224,
        fill_color: tuple = (128, 128, 128),
        random_twist: bool = False
    ):
        """
        初始化数据增强器配置

        Args:
            margin (int): 扩展边距，默认为20像素，用于在切割时增加边距以避免边缘效应
            random_offset (float): 随机偏移量，默认为0.0，用于在切割时增加随机偏移，数值表示偏移占长宽的比例
            random_rotate (float): 随机旋转角度，默认为0.0，用于在切割时增加随机旋转，数值表示角度的范围（-random_rotate到random_rotate）
            padding (int): 填充大小，默认为20像素，用于将切割后的图像填充为正方形
            target_size (int): 目标图像大小，默认为224，用于将切割后的图像填充为正方形
            fill_color (tuple): 填充颜色，默认为灰色 (128, 128, 128)
            random_twist (bool): 是否随机旋转图像，默认为False，旋转步进量为90°
        """
        self.margin = margin
        self.random_offset = random_offset
        self.random_rotate = random_rotate
        self.padding = padding
        self.target_size = target_size
        self.fill_color = fill_color
        self.random_twist = random_twist

    def make_square(self, img: cv2.typing.MatLike) -> cv2.typing.MatLike:
        """
        将图像填充为正方形

        Args:
            img: 输入图像
        Returns:
            正方形图像
        """
        h, w = img.shape[:2]
        # 计算可用空间（减去内边距）
        available_size = self.target_size - self.padding * 2
        # 计算缩放比例，保持长宽比不变
        scale = min(available_size / w, available_size / h)
        new_w, new_h = int(w * scale), int(h * scale)
        # 缩放图像
        img_resized = cv2.resize(
            img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        # 创建正方形画布
        square_img = np.full((self.target_size, self.target_size, 3),
                             self.fill_color, dtype=np.uint8)
        # 计算居中位置
        y_offset = (self.target_size - new_h) // 2
        x_offset = (self.target_size - new_w) // 2
        # 将缩放后的图像放置在画布中心
        square_img[y_offset:y_offset+new_h,
                   x_offset:x_offset+new_w] = img_resized

        return square_img

    def get_piece_images(self, image: np.ndarray, ann: np.ndarray) -> list[np.ndarray]:
        """
        从图像中提取旋转检测框正方向的零件图像，用于二阶段分类模型的训练

        Args:
            image (np.ndarray): 输入图像
            ann (np.ndarray): 旋转检测框的xywhr格式参数
        Returns:
            list[np.ndarray]: 包含四个方向图像的列表，顺序为右、下、左、上，代表0°、90°、180°、270°四个方向的图像
        """
        cx, cy, w, h, angle = ann

        # 随机偏移
        if self.random_offset > 0:
            ox = (np.random.random() - 0.5) * 2 * w * self.random_offset
            oy = (np.random.random() - 0.5) * 2 * h * self.random_offset
            cx += ox
            cy += oy

        # 随机旋转
        if self.random_rotate > 0:
            angle_offset = (np.random.random() - 0.5) * \
                2 * np.radians(self.random_rotate)
            angle += angle_offset

        w += self.margin * 2
        h += self.margin * 2
        pts = YOLOv8_OBB_Dataset.xywhr2xyxyxyxy(
            np.array([cx, cy, w, h, angle])
        )

        M = cv2.getPerspectiveTransform(
            pts.astype(np.float32),
            np.array([[w, 0], [w, h], [0, h], [0, 0]]).astype(np.float32)
        )
        piece_image = cv2.warpPerspective(image, M, (int(w), int(h)))

        down_part = piece_image[int(h/2):, :]
        up_part = piece_image[:int(h/2), :]
        left_part = piece_image[:, :int(w/2)]
        right_part = piece_image[:, int(w/2):]

        return [
            self.make_square(right_part),
            self.make_square(down_part),
            self.make_square(left_part),
            self.make_square(up_part)
        ]

    def build(
        self,
        obb_dataset: Dataset,
        save_path: str | Path,
        file_prefix: str = ''
    ):
        """
        构建零件分类数据集

        Args:
            obb_dataset (Dataset): OBB数据集
            save_path (str or Path): 保存路径
            file_prefix (str): 文件名前缀
        """
        save_path = Path(save_path) if isinstance(
            save_path, str) else save_path
        if not save_path.exists():
            save_path.mkdir(parents=True, exist_ok=True)

        save_path.joinpath("down").mkdir(parents=True, exist_ok=True)
        save_path.joinpath("up").mkdir(parents=True, exist_ok=True)
        save_path.joinpath("left").mkdir(parents=True, exist_ok=True)
        save_path.joinpath("right").mkdir(parents=True, exist_ok=True)

        # 从OBB数据集中提取每一个旋转检测框，并将其中图像上正的切下来
        for image, boxes, id_image in tqdm(obb_dataset, desc="提取零件图像"):  # type: ignore
            image: cv2.Mat
            for id_label, (class_id, ann) in enumerate(boxes):
                right_part, down_part, left_part, up_part = self.get_piece_images(
                    image, ann)

                # 保存图像
                file_name = f"{id_image}_{class_id}_{id_label}"
                if file_prefix.strip():
                    file_name = file_prefix + file_name

                # 随机旋转90°、180°或270°
                twist_angle = lambda: 0
                if self.random_twist:
                    twist_angle = lambda: random.choice([0, 90, 180, 270])

                Image.fromarray(down_part).rotate(twist_angle()).save(
                    save_path / "down" / f"{file_name}_down.png")
                Image.fromarray(up_part).rotate(twist_angle()).save(
                    save_path / "up" / f"{file_name}_up.png")
                Image.fromarray(left_part).rotate(twist_angle()).save(
                    save_path / "left" / f"{file_name}_left.png")
                Image.fromarray(right_part).rotate(twist_angle()).save(
                    save_path / "right" / f"{file_name}_right.png")

