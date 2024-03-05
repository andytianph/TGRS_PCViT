from .builder import DATASETS
from .coco import CocoDataset


@DATASETS.register_module()
class DIORDataset(CocoDataset):

    CLASSES =  ("airplane", "airport", "baseball_field", "basketball_court", "bridge",  
            "chimney", "dam", "expressway_service_area", "expressway_toll_station", "golf_field",
            "ground_track_field", "harbor", "overpass", "ship", "stadium", "storage_tank", "tennis_court",
            "train_station", "vehicle", "wind_mill")
