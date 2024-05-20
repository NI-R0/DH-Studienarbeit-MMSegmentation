from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset

@DATASETS.register_module()
class UDDDataset(BaseSegDataset):
    METAINFO = dict(
    # All segmentation classes of the dataset
        # classes = (
        #     'Vegetation',
        #     'Building',
        #     'Road',
        #     'Vehicle',
        #     'Other'
        # ),
        # # One colour (RGB) per segmentation class. Used to visualize the prediction result.
        # palette=[
        #     [107,142,35],
        #     [102,102,156],
        #     [128,64,128],
        #     [0,0,142],
        #     [0,0,0]
        # ]
        classes=(
            'Building',
            'Road',
            'Static Car',
            'Tree',
            'Low Vegetation',
            'Human',
            'Moving Car',
            'Background'
        ),
        # One colour (RGB) per segmentation class. Used to visualize the prediction result.
        palette=[
            [102,102,156],
            [128,64,128],
            [0,0,142],
            [107,142,35],
            [107,142,35],
            [0,0,0],
            [0,0,142],
            [0,0,0]
        ]
    )
    def __init__(self,
    img_suffix='.JPG',
    seg_map_suffix='_labelTrainIds.png',
    **kwargs) -> None:
        super().__init__(img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)