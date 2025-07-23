_base_ = ["../_base_/default_runtime.py"]

# misc custom setting
batch_size = 1  # bs: total bs in all gpus
num_worker = 24
mix_prob = 0.8
empty_cache = False
enable_amp = True

# model settings
model = dict(
    type="DefaultSegmentorV2",
    num_classes=4,
    backbone_out_channels=64,
    backbone=dict(
        type="PT-v3m1",
        in_channels=3,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(1024, 1024, 1024, 1024),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=True,
        pre_norm=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        cls_mode=False,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D"),
    ),
    criteria=[
        dict(type="CrossEntropyLoss", weight=[0.01,0.45,0.45,0.09], loss_weight=1.0, ignore_index=-1),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
    ],
)

# scheduler settings
epoch = 7
eval_epoch=1
optimizer = dict(type="AdamW", lr=0.006, weight_decay=0.05)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.006, 0.0006],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)
param_dicts = [dict(keyword="block", lr=0.0006)]

# dataset settings
dataset_type = "QCSFDataset"
data_root = "/home/ostocker/travaux/data/ALSlike_full"
info=dict(intensity_max=125, dataset_usage=1)


data = dict(
    num_classes=4,
    ignore_index=4,
    names=[
        "ground",
        "abies balsamea",
        "picea mariana",
        "low vegetation",
    ],
    train=dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
        info=info,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="train",
                keys=("coord", "strength", "segment"),
                return_grid_coord=True,
            ),
            dict(type="CenterShift", apply_z=False),
            
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment"),
                feat_keys=("coord"),
            ),
        ],
        test_mode=False,
    ),
    val=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        info=info,
        transform=[
            dict(type="CenterShift", apply_z=False),
            dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="train",
                keys=("coord", "strength", "segment"),
                return_grid_coord=True,
            ),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment"),
                feat_keys=("coord"),
            ),
        ],
        test_mode=False,
    ),
    test=dict(
        type=dataset_type,
        split="test",
        data_root=data_root,
        info=info,
        transform=[
            dict(type="CenterShift", apply_z=True),
        ],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="test",
                keys=("coord", "strength", "segment"),
                return_grid_coord=True,
            ),
            crop=None,
            post_transform=[
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "grid_coord","index"),
                    feat_keys=("coord"),
                ),
            ],
            aug_transform=[[dict(type="Dummy")]],
        ),
    ),
)
