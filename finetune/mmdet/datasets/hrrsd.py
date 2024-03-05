from .builder import DATASETS
from .coco import CocoDataset


@DATASETS.register_module()
class HRRSDDataset(CocoDataset):

    CLASSES =  ( "bridge", "airplane", "ground track field", "vehicle", "parking lot", "T junction", 
                "baseball diamond", "tennis court", "basketball court", "ship", "crossroad", "harbor", "storage tank")