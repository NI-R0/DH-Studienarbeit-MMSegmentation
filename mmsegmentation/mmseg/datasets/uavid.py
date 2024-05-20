from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset

@DATASETS.register_module()
class UavidDataset(BaseSegDataset):
    METAINFO = dict(
    # All segmentation classes of the dataset
        classes = ('Building','Road','Static Car','Tree','Low vegetation',
            'Human','Moving Car','Background Clutter'),
        # One colour (RGB) per segmentation class. Used to visualize the prediction result.
        palette=[[128, 0, 0],[128, 64, 128],[192, 0, 192],[0, 128, 0],
            [128, 128, 0],[64, 64, 0],[64, 0, 128],[0, 0, 0]])
    
    def __init__(self,
    img_suffix='.png',
    seg_map_suffix='_labelTrainIds.png',
    **kwargs) -> None:
        super().__init__(img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)



