from .builder import DATASETS
from .coco import CocoDataset


@DATASETS.register_module()
class NWPUDataset(CocoDataset):

    CLASSES =  ("airplane", "ship", "storage_tank", "baseball_diamond", "tennis_court",  
           "basketball_court", "ground_track_field", "harbor", "bridge", "vehicle")
