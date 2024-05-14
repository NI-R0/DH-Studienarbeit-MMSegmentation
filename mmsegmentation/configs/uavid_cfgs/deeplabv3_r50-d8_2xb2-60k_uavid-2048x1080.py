_base_ = [
    '../_base_/models/deeplabv3_r50-d8.py', # Path to the model
    '../_base_/datasets/uavid-2xb2-2048x1080.py', # Path to your dataset config file
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_60k.py' # Specify according to your needs
]
crop_size = (1080,2048)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(num_classes=8),
    auxiliary_head=dict(num_classes=8))