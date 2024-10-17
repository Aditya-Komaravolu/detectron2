from .mask_rcnn_R_101_FPN_100ep_LSJ import (
    dataloader,
    lr_multiplier,
    model,
    optimizer,
    train,
)

train.max_iter *= 4  # 100ep -> 400ep

train.init_checkpoint = "/home/aditya/detectron2/model_final_f96b26.pkl"
optimizer.lr = 1e-5
# optimizer.betas = (0.9, 0.999)
optimizer.weight_decay = 2.5e-4
# optimizer.params.lr_factor_func = lambda module_name: 0.1 if "backbone" in module_name else 1

model.roi_heads.num_classes = 2


train.output_dir = "/home/aditya/detectron2_mask_rcnn_R_101_FPN_400ep_LSJ_training_d4_may8"


# run evaluation every 5000 iters
train.eval_period = 1000

# log training infomation every 20 iters
train.log_period = 100

# save checkpoint every 5000 iters
train.checkpointer.period = 1000


# set training devices
train.device = "cuda"


lr_multiplier.scheduler.milestones = [
    milestone * 4 for milestone in lr_multiplier.scheduler.milestones
]
lr_multiplier.scheduler.num_updates = train.max_iter

dataloader.train.num_workers = 8

dataloader.train.total_batch_size = 8
