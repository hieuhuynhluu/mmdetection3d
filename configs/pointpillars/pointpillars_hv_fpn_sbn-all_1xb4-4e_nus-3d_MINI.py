# pointpillars_hv_fpn_sbn-all_1xb4-4e_nus-3d_custom.py

_base_ = './pointpillars_hv_fpn_sbn-all_8xb4-2x_nus-3d.py'

# 1. Customize training schedule to run for 4 epochs (for quick testing)
max_epochs = 4
train_cfg = dict(
    max_epochs=max_epochs,
    val_interval=1
)

# 2. Adjust LR schedule to match the new max_epochs
param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[3], # Decay the LR once near the end
        gamma=0.1)
]

data_root = 'data/nuscenes/'

# 2. Update the training and validation paths to use the new data_root
# This block ensures the correct info files are used for training

train_dataloader = dict(
    batch_size=4, # Your custom batch size
    dataset=dict(
        data_root=data_root,
        
        # The training info file
        ann_file='nuscenes_infos_train.pkl',
    )
)

# This block ensures the correct info file is used for validation

val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file='nuscenes_infos_val.pkl'
    )
)

test_dataloader = dict(
    batch_size=4,
    dataset=dict(
        data_root=data_root,
        ann_file='nuscenes_infos_val.pkl'   # ← Must match evaluator split
    )
)
