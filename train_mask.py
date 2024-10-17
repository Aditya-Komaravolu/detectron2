import torch, detectron2
TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)
print("detectron2:", detectron2.__version__)



# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
# import numpy as np
import os
# import json, cv2, random
# from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.data import transforms as T
# from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg, CfgNode
# from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)


from detectron2.engine import DefaultTrainer

# if your dataset is in COCO format, this cell can be replaced by the following three lines:
from detectron2.data.datasets import register_coco_instances
register_coco_instances("train", {}, "/home/aditya/snaglist_dataset_d4_may4/annotations/train.json", "/home/aditya/snaglist_dataset_d4_may4/train/")
register_coco_instances("val", {}, "/home/aditya/snaglist_dataset_d4_may4/annotations/valid.json", "/home/aditya/snaglist_dataset_d4_may4/valid")



train_meta = MetadataCatalog.get("train")
train_meta.thing_classes = ['cement_slurry', 'honeycomb']

def build_evaluator(cfg, dataset_name, output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each builtin dataset.
    For your own dataset, you can simply create an evaluator manually in your
    script and do not have to worry about the hacky if-else logic here.
    """
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    evaluator_list = []
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
        evaluator_list.append(
            SemSegEvaluator(
                dataset_name,
                distributed=True,
                output_dir=output_folder,
            )
        )
    if evaluator_type in ["coco", "coco_panoptic_seg"]:
        evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
    if evaluator_type == "coco_panoptic_seg":
        evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
    if evaluator_type == "cityscapes_instance":
        return CityscapesInstanceEvaluator(dataset_name)
    if evaluator_type == "cityscapes_sem_seg":
        return CityscapesSemSegEvaluator(dataset_name)
    elif evaluator_type == "pascal_voc":
        return PascalVOCDetectionEvaluator(dataset_name)
    elif evaluator_type == "lvis":
        return LVISEvaluator(dataset_name, output_dir=output_folder)
    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(
                dataset_name, evaluator_type
            )
        )
    elif len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)

class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can write your
    own training loop. You can use "tools/plain_train_net.py" as an example.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return build_evaluator(cfg, dataset_name, output_folder)


# ARCHITECTURE = "mask_rcnn_X_101_32x8d_FPN_3x"
CONFIG_FILE_PATH = "/home/aditya/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"
OUTPUT_DIR_PATH = "/home/aditya/detectron2_mask_rcnn_X_101_32x8d_FPN_3x_training_d4_may8"
  


cfg = get_cfg()
cfg.merge_from_file(CONFIG_FILE_PATH)
cfg.DATASETS.TRAIN = ("train",)
cfg.DATASETS.TEST = ("val",)
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = "/home/aditya/detectron2/model_final_2d9806.pkl"  # Let training initialize from model zoo

cfg.SOLVER.IMS_PER_BATCH = 32  # This is the real "batch size" commonly known to deep learning people
cfg.INPUT.MASK_FORMAT='bitmask'
cfg.SOLVER.BASE_LR = 1e-4  # pick a good LR
cfg.SOLVER.MAX_ITER = 200000   # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = (2000,10000, 20000)
cfg.SOLVER.GAMMA = 0.1  # Reduce learning rate by a factor of 10 at each step
# cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)

cfg.SOLVER.CHECKPOINT_PERIOD = 1000
cfg.OUTPUT_DIR = OUTPUT_DIR_PATH
cfg.TEST.EVAL_PERIOD=1000
cfg.DATALOADER.AUGMENTATIONS = [
    # T.Resize((1024, 1024)),
    # T.RandomBrightness(0.5, 1.5),
    # T.RandomContrast(0.5, 1.5),
    # T.RandomRotation(angle=[-10, 10], expand=True),
    # T.RandomCrop("absolute", (640, 640))
]     
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.


os.makedirs(OUTPUT_DIR_PATH, exist_ok=True)
cfg_yaml = CfgNode(cfg).dump()
config_path = os.path.join(cfg.OUTPUT_DIR, "config.yaml")
with open(config_path, "w") as f:
    f.write(cfg_yaml)


trainer = Trainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()
