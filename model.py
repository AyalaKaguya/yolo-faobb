from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
from dataset import YOLOv8_OBB_Dataset

class FullAngleOBB():
    def __init__(
        self,
        cls_ckpt: YOLO | str | Path = "yolo11n-cls.pt",
        obb_ckpt: YOLO | str | Path = "yolo11n-obb.pt",
    ):
        if isinstance(cls_ckpt, (str, Path)):
            self.model_cls = YOLO(cls_ckpt)
        elif isinstance(cls_ckpt, YOLO):
            self.model_cls = cls_ckpt
        else:
            raise TypeError(
                "cls_ckpt must be a YOLO model or a path to the model file")

        if isinstance(obb_ckpt, (str, Path)):
            self.model_obb = YOLO(obb_ckpt)
        elif isinstance(obb_ckpt, YOLO):
            self.model_obb = obb_ckpt
        else:
            raise TypeError(
                "obb_ckpt must be a YOLO model or a path to the model file")

        self.__angle_matrix = np.array([
            [2, 0, 1, 3],
            [3, 2, 0, 1],
            [1, 3, 2, 0],
            [0, 1, 3, 2]
        ])
        self.__position_weights = np.array([1.0, 1.0, 1.0, 1.0])
        self.__margin = 20
        self.__padding = 20

    @property
    def margin(self) -> int:
        """
        获取分类模型前端分割器分割图像时额外的边距大小
        Returns:
            int: 边距大小
        """
        return self.__margin

    @margin.setter
    def margin(self, value: int):
        """
        设置分类模型前端分割器分割图像时额外的边距大小
        Args:
            value (int): 边距大小
        """
        if value < 0:
            raise ValueError("margin must be a non-negative integer")
        self.__margin = value

    @property
    def padding(self) -> int:
        """
        获取分类模型前端分割器填充正方形时留出的内边距
        Returns:
            int: 填充大小
        """
        return self.__padding

    @padding.setter
    def padding(self, value: int):
        """
        设置分类模型前端分割器填充正方形时留出的内边距
        Args:
            value (int): 填充大小
        """
        if value < 0:
            raise ValueError("padding must be a non-negative integer")
        self.__padding = value

    @property
    def cls_classes(self) -> dict[int, str]:
        """
        获取分类名称列表
        Returns:
            list[str]: 分类名称列表
        """
        return self.model_cls.names

    @property
    def obb_classes(self) -> dict[int, str]:
        """
        获取OBB检测框类别名称列表
        Returns:
            list[str]: OBB检测框类别名称列表
        """
        return self.model_obb.names

    def make_square(self, img: cv2.typing.MatLike, target_size=224, fill_color=(128, 128, 128)) -> cv2.typing.MatLike:
        """
        将图像填充为正方形
        Args:
            img: 输入图像
            target_size: 目标尺寸
            fill_color: 填充颜色 (灰色)
        Returns:
            正方形图像
        """
        h, w = img.shape[:2]
        # 计算可用空间（减去内边距）
        available_size = target_size - self.padding * 2
        # 计算缩放比例，保持长宽比不变
        scale = min(available_size / w, available_size / h)
        new_w, new_h = int(w * scale), int(h * scale)
        # 缩放图像
        img_resized = cv2.resize(
            img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        # 创建正方形画布
        square_img = np.full((target_size, target_size, 3),
                             fill_color, dtype=np.uint8)
        # 计算居中位置
        y_offset = (target_size - new_h) // 2
        x_offset = (target_size - new_w) // 2
        # 将缩放后的图像放置在画布中心
        square_img[y_offset:y_offset+new_h,
                   x_offset:x_offset+new_w] = img_resized

        return square_img

    def get_piece_image(self, image: np.ndarray, ann: np.ndarray) -> list[np.ndarray]:
        """
        从图像中提取旋转检测框正方向的零件图像

        Args:
            image (np.ndarray): 输入图像
            ann (np.ndarray): 旋转检测框的xywhr格式参数
        Returns:
            list[np.ndarray]: 包含四个方向图像的列表，顺序为右、下、左、上，代表0°、90°、180°、270°四个方向的图像
        """
        cx, cy, w, h, angle = ann
        w += self.margin * 2
        h += self.margin * 2
        pts = YOLOv8_OBB_Dataset.xywhr2xyxyxyxy(
            np.array([cx, cy, w, h, angle])
        )

        M = cv2.getPerspectiveTransform(
            pts.astype(np.float32),
            np.array([[w, h], [w, 0], [0, 0], [0, h],]).astype(np.float32)
        )
        piece_image = cv2.warpPerspective(image, M, (int(w), int(h)))

        right_part = piece_image[:, int(w/2):]
        down_part = piece_image[int(h/2):, :]
        left_part = piece_image[:, :int(w/2)]
        up_part = piece_image[:int(h/2), :]

        return [
            self.make_square(right_part),
            self.make_square(down_part),
            self.make_square(left_part),
            self.make_square(up_part)
        ]

    @property
    def angle_matrix(self) -> np.ndarray:
        """
        角度矩阵

        该属性返回一个二维数组，表示每个旋转角度下的期望分类结果。
        行索引：0=0°, 1=90°, 2=180°, 3=270°
        列索引：0=右方向, 1=下方向, 2=左方向, 3=上方向
        数值：期望的分类结果 (0=down, 1=left, 2=right, 3=up)
        """
        return self.__angle_matrix

    @angle_matrix.setter
    def angle_matrix(self, value: np.ndarray):
        """
        设置角度矩阵

        该方法允许用户自定义角度矩阵，以适应不同的分类需求。
        Args:
            value (np.ndarray): 新的角度矩阵，必须是一个形状为 (4, 4) 的二维数组
        """
        if value.shape != (4, 4):
            raise ValueError("angle_matrix must be a 4x4 matrix")
        self.__angle_matrix = value

    @property
    def position_weights(self) -> np.ndarray:
        """
        位置权重向量

        该属性返回一个一维数组，表示每个方向分类结果的权重。
        顺序为 [右, 下, 左, 上]，默认值为 [1.0, 1.0, 1.0, 1.0]
        """
        return self.__position_weights

    @position_weights.setter
    def position_weights(self, value: np.ndarray):
        """
        设置位置权重向量

        该方法允许用户自定义位置权重，以适应不同的分类需求。
        Args:
            value (np.ndarray): 新的权重向量，必须是一个长度为 4 的一维数组
        """
        if value.shape != (4,):
            raise ValueError("position_weights must be a 1D array of length 4")
        self.__position_weights = value

    def match_angle_matrix(
        self,
        input_vector: np.ndarray,
        input_vector_conf: np.ndarray,
        match_reward: float = 1.0,
        mismatch_penalty: float = -0.5
    ) -> tuple[float, float]:
        """
        基于置信度加权的角度匹配算法

        该方法根据输入的方向分类结果向量和置信度向量，计算出最佳的角度修正值和对应的置信度。
        Args:
            input_vector (np.ndarray): 输入的方向分类结果向量 [右,下,左,上] 对应的类别索引
            input_vector_conf (np.ndarray): 输入的方向分类置信度向量 [右,下,左,上] 对应的置信度
            position_weights (np.ndarray, optional): 位置权重向量 [右,下,左,上]，默认为 [1.0, 1.0, 1.0, 1.0]
            match_reward (float): 匹配奖励值，默认为 1.0
            mismatch_penalty (float): 不匹配惩罚值，默认为 -0
        Returns:
            float: 最终角度修正值（弧度）
            float: 修正值的置信度
        """
        assert len(input_vector) == 4, "input_vector must have 4 elements"
        assert len(
            input_vector_conf) == 4, "input_vector_conf must have 4 elements"

        # 向量化计算所有角度模式的匹配情况
        matches_matrix = (input_vector[np.newaxis, :]
                          == self.angle_matrix)  # shape: (4, 4)

        # 创建奖励/惩罚矩阵
        reward_penalty_matrix = np.where(
            matches_matrix, match_reward, mismatch_penalty)  # shape: (4, 4)

        # 向量化计算加权分数
        confidence_weights = input_vector_conf * \
            self.position_weights  # shape: (4,)
        weighted_scores = np.sum(
            # shape: (4,)
            confidence_weights[np.newaxis, :] * reward_penalty_matrix, axis=1)

        # 选择加权分数最高的角度
        best_angle_idx = np.argmax(weighted_scores)
        best_matches = matches_matrix[best_angle_idx]  # shape: (4,)

        # 向量化计算最终置信度
        if np.any(best_matches):
            # 使用匹配位置的加权平均置信度
            matching_mask = best_matches
            matching_confidences = input_vector_conf[matching_mask]
            matching_weights = self.position_weights[matching_mask]
            final_confidence = np.average(
                matching_confidences, weights=matching_weights)
        else:
            # 如果没有匹配位置，使用所有位置的加权平均
            final_confidence = np.average(
                input_vector_conf, weights=self.position_weights)

        # 将角度索引转换为[-pi, pi]的弧度值
        angle_correction = np.radians(best_angle_idx * 90)
        if angle_correction > np.pi:
            angle_correction -= 2 * np.pi

        return angle_correction, float(final_confidence)

    def predict(
        self,
        image: np.ndarray | str | Path,
        obb_conf: float = 0.6,
        obb_iou: float = 0.6,
    ):
        """
        对输入图像进行预测，返回分类结果和旋转检测框，不支持批量预测，但是在四向分类阶段支持
        Args:
            image (np.ndarray | str | Path): 输入图像，可以是图像路径或图像数组
        Returns:
            dict: 包含OBB检测结果和每个检测框四个方向分类结果的字典
        """
        if isinstance(image, (str, Path)):
            image = cv2.imread(str(image)) # type: ignore
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # type: ignore

        # 第一阶段：OBB检测
        results_obb = self.model_obb.predict(
            image,
            verbose=False,
            conf=obb_conf,
            iou=obb_iou
        )

        if len(results_obb[0].obb) == 0:  # type: ignore
            return {
                "obb_boxes": [],
                "piece_classifications": [],
            }

        # 获取OBB检测结果
        obb_boxes = results_obb[0].obb.xywhr.cpu().numpy()   # type: ignore
        obb_confs = results_obb[0].obb.conf.cpu().numpy()   # type: ignore
        obb_cls = results_obb[0].obb.cls.cpu().numpy()   # type: ignore

        # 第二阶段：对每个检测框进行四个方向的分类
        piece_images = [img for box in obb_boxes
                        for img in self.get_piece_image(image, box)] # type: ignore

        results_cls = self.model_cls.predict(piece_images, verbose=False)

        # 每四个结果一组，使用角度匹配算法确定最佳角度修正
        angle_corrections = []
        andle_confidences = []
        cls_predictions = []
        cls_confidences = []
        cls_pred_matrices = []
        cls_conf_matrices = []

        for i in range(0, len(piece_images), 4):
            direction_results = results_cls[i:i+4]
            # 提取分类结果
            predictions = np.array([result.probs.top1   # type: ignore
                                    for result in direction_results])
            confidences = np.array([result.probs.top1conf.cpu().numpy()  # type: ignore
                                    for result in direction_results])
            # 角度匹配
            angle_correction, andle_confidence = self.match_angle_matrix(
                np.array(predictions), np.array(confidences)
            )
            # 保存结果
            angle_corrections.append(angle_correction)
            andle_confidences.append(andle_confidence)
            cls_predictions.append(predictions)
            cls_confidences.append(confidences)

            cls_pred_matrix = np.array([result.probs.top5  # type: ignore
                                        for result in direction_results])
            cls_conf_matrix = np.array([result.probs.top5conf.cpu().numpy()  # type: ignore
                                        for result in direction_results])
            cls_pred_matrices.append(cls_pred_matrix)
            cls_conf_matrices.append(cls_conf_matrix)

        # 修正OBB角度和宽高
        corrected_boxes = obb_boxes.copy()
        for i, correction in enumerate(angle_corrections):
            angle = corrected_boxes[i][4] + correction
            angle = angle if angle < np.pi else angle - 2 * np.pi
            corrected_boxes[i][4] = angle  # 更新角度
            correction_degrees = np.degrees(correction) % 360
            if abs(correction_degrees - 90) < 45 or abs(correction_degrees - 270) < 45:
                corrected_boxes[i][2], corrected_boxes[i][3] = corrected_boxes[i][3], corrected_boxes[i][2]

        return {
            "obb_boxes": corrected_boxes,
            "obb_classes": obb_cls,
            "obb_confidences": obb_confs,
            "original_obb_boxes": obb_boxes,
            "angle": angle_corrections,
            "andle_confidences": andle_confidences,
            "cls_predictions": cls_predictions,
            "cls_confidences": cls_confidences,
            "cls_conf_matrices": cls_conf_matrices,
            "cls_pred_matrices": cls_pred_matrices,
        }