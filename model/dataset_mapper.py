import copy
import logging
import numpy as np
import torch
from detectron2.structures import Instances
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T

__all__ = ["SparseInstDatasetMapper"]

def build_transform_gen(cfg, is_train):
    augmentation = []
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        sample_style = "choice"

    if is_train and cfg.INPUT.RANDOM_FLIP != "none":
        augmentation.append(
            T.RandomFlip(
                horizontal=cfg.INPUT.RANDOM_FLIP == "horizontal",
                vertical=cfg.INPUT.RANDOM_FLIP == "vertical",
            )
        )

    if is_train:
        augmentation.append(
            T.ResizeShortestEdge(min_size, max_size, sample_style)
        )

    return augmentation


class SparseInstDatasetMapper:
    def __init__(self, cfg, is_train: bool = True):
        augs = build_transform_gen(cfg, is_train)
        self.default_aug = T.AugmentationList(augs)

        if cfg.INPUT.CROP.ENABLED and is_train:
            crop_gen = [
                T.ResizeShortestEdge([400, 500, 600], sample_style='choice'),
                T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE)
            ]
            recompute_boxes = cfg.MODEL.MASK_ON
            augs = augs[:-1] + crop_gen + augs[-1:]
            self.crop_aug = T.AugmentationList(augs)
        else:
            self.crop_aug = None
            recompute_boxes = False

        self.is_train = is_train
        self.image_format = cfg.INPUT.FORMAT
        self.use_instance_mask = cfg.MODEL.MASK_ON
        self.instance_mask_format = cfg.INPUT.MASK_FORMAT
        self.recompute_boxes = recompute_boxes

        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[DatasetMapper] Augmentations used in {mode}: {augs}")

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        logger = logging.getLogger(__name__)

        try:
            # 读取图片
            image = utils.read_image(dataset_dict["file_name"], format=self.image_format)

            # 获取真实的图片尺寸
            actual_height, actual_width = image.shape[:2]
            dataset_dict["height"], dataset_dict["width"] = actual_height, actual_width

            # 检查图片尺寸是否匹配
            utils.check_image_size(dataset_dict, image)

            # 处理 `sem_seg_file_name`
            sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name"), "L").squeeze(
                2) if "sem_seg_file_name" in dataset_dict else None

            # 数据增强
            aug_input = T.AugInput(image, sem_seg=sem_seg_gt)
            transforms = self.crop_aug(aug_input) if self.crop_aug and np.random.rand() > 0.5 else self.default_aug(
                aug_input)
            image, sem_seg_gt = aug_input.image, aug_input.sem_seg

            # 确保 dataset_dict["height"] 和 dataset_dict["width"] 是最新的
            dataset_dict["height"], dataset_dict["width"] = image.shape[:2]

            # 处理 image 数据
            dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

            if sem_seg_gt is not None:
                dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))

            if not self.is_train:
                dataset_dict.pop("annotations", None)
                dataset_dict.pop("sem_seg_file_name", None)
                return dataset_dict

            # 处理 annotations
            if "annotations" in dataset_dict:
                annos = [
                    utils.transform_instance_annotations(obj, transforms, image.shape[:2])
                    for obj in dataset_dict.pop("annotations")
                    if obj.get("iscrowd", 0) == 0
                ]

                # 检查并确保每个 annotation 都有 category_id
                for obj in annos:
                    if "category_id" not in obj:
                        logger.warning(f"Annotation missing 'category_id', assigning default value 0.")
                        obj["category_id"] = 0  # 默认类别 ID

                # 生成 instances
                instances = utils.annotations_to_instances(
                    annos, image.shape[:2], mask_format=self.instance_mask_format
                )

                # 确保 instances 包含 gt_classes
                if not instances.has("gt_classes"):
                    logger.warning(f"Instances missing 'gt_classes', adding from annotations.")
                    gt_classes = [obj.get("category_id", 0) for obj in annos]
                    instances.gt_classes = torch.tensor(gt_classes, dtype=torch.int64)

                # 确保 instances 包含 gt_boxes（如果需要）
                if self.recompute_boxes and instances.has("gt_masks"):
                    instances.gt_boxes = instances.gt_masks.get_bounding_boxes()

                # 过滤空实例
                instances = utils.filter_empty_instances(instances)
                if len(instances) == 0:
                    logger.warning(f"No valid instances found, creating empty Instances.")
                    instances = Instances(image.shape[:2])
                    instances.gt_classes = torch.tensor([], dtype=torch.int64)  # 添加空的 gt_classes

                dataset_dict["instances"] = instances
            else:
                logger.warning(f"No annotations found, creating empty Instances.")
                instances = Instances(image.shape[:2])
                instances.gt_classes = torch.tensor([], dtype=torch.int64)  # 添加空的 gt_classes
                dataset_dict["instances"] = instances

            return dataset_dict

        except Exception as e:
            logger.error(
                f"严重错误: 无法处理文件 {dataset_dict.get('file_name', 'unknown')}, 错误类型: {type(e).__name__}, 详情: {str(e)}")
            return None  # 返回None使DataLoader自动跳过
