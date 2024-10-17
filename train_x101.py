from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer, HookBase
from detectron2.config import get_cfg, CfgNode
from detectron2 import model_zoo
from detectron2.evaluation import COCOEvaluator
from detectron2.utils.events import get_event_storage

import torch
import os


register_coco_instances("snaglist_train", {}, "/home/aditya/snaglist_dataset_apr9/annotations/train.json", "/home/aditya/snaglist_dataset_apr9/train")
register_coco_instances("snaglist_val", {}, "/home/aditya/snaglist_dataset_apr9/annotations/valid.json", "/home/aditya/snaglist_dataset_apr9/valid")


cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("snaglist_train",)
cfg.DATASETS.TEST = ("snaglist_val",)
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 12  # This is the real "batch size" commonly known to deep learning people
cfg.SOLVER.BASE_LR = 5e-4  # pick a good LR
cfg.SOLVER.MAX_ITER = 170000    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.SOLVER.LOGGING_PERIOD = 100  # Log training metrics every 20 iterations
cfg.SOLVER.CHECKPOINT_PERIOD = 1000
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

cfg.OUTPUT_DIR = "/home/aditya/faster_rcnn_X_101_32x8d_FPN_3x-training-new"




os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
# Convert config to a YAML string and save it
cfg_yaml = CfgNode(cfg).dump()
config_path = os.path.join(cfg.OUTPUT_DIR, "config.yaml")
with open(config_path, "w") as f:
    f.write(cfg_yaml)

    

class CustomEvalHook(HookBase):
    def __init__(self, eval_period, evaluator):
        self._period = eval_period
        self._evaluator = evaluator

    def after_step(self):
        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        if is_final or (self._period > 0 and next_iter % self._period == 0):
            self._do_eval()

    def _do_eval(self):
        results = self.trainer.test(self.trainer.cfg, self.trainer.model, evaluators=[self._evaluator])
        if torch.cuda.is_initialized():
            torch.cuda.synchronize()
        print("Evaluation results:", results)  # Add this line to inspect the results structure
        self._log_eval_results(results)



    def _log_eval_results(self, results):
        print("Logging results to TensorBoard")
        storage = get_event_storage()
        for key, metrics in results.items():
            for metric, value in metrics.items():
                if not torch.isnan(torch.tensor(value)):
                    storage.put_scalar(f"{key}/{metric}", value)




class CustomTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)
    
    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.append(CustomEvalHook(eval_period=1000, evaluator=self.build_evaluator(self.cfg, "snaglist_val")))
        return hooks




# Use CustomTraine
trainer = CustomTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()