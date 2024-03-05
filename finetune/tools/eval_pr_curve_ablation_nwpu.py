# =========================================================
# @purpose: plot PR curve by COCO API and mmdet API
# @date：   2020/12
# @version: v1.0
# @author： Xu Huasheng
# @github： https://github.com/xuhuasheng/mmdetection_plot_pr_curve
# =========================================================

import os
import mmcv
import numpy as np
import matplotlib.pyplot as plt

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from mmcv import Config
from mmdet.datasets import build_dataset

EXPERIMENT = "a_TGRS"
DATA = "nwpu_split1"

# MODEL_NAME = "ViTDet_PCViT_Base_FPN_nwpuH_xP"
# OUT_DIR = "ViTDet_ConvViTAE_Base_FPN_xP"

MODEL_NAME1 = "ViTDet_PCViT_Base_FPN_nwpuH_xP"
OUT_DIR1 = "ViTDet_ConvViTAE_Base_FPN_xP"
CONFIG_FILE1 = f"../configs/{EXPERIMENT}/{DATA}/{MODEL_NAME1}.py"
RESULT_FILE1 = f"../work_dirs/{DATA}/{OUT_DIR1}/best.pkl"

MODEL_NAME2 = "ViTDet_PCViT_Base_FPN_nwpuH_FastConvPre"
OUT_DIR2 = "ViTDet_ConvViTAE_Base_FPN_FastConvPre_49epo"
CONFIG_FILE2 = f"../configs/{EXPERIMENT}/{DATA}/{MODEL_NAME2}.py"
RESULT_FILE2 = f"../work_dirs/{DATA}/{OUT_DIR2}/best.pkl"

MODEL_NAME3 = "ViTDet_PCViTknn_Base_FPN_nwpuH_FastConvPre"
OUT_DIR3 = "ViTDet_ConvViTAE_knn_Base_FPN_FastConvPre_49epo"
CONFIG_FILE3 = f"../configs/{EXPERIMENT}/{DATA}/{MODEL_NAME3}.py"
RESULT_FILE3 = f"../work_dirs/{DATA}/{OUT_DIR3}/best.pkl"

MODEL_NAME4 = "ViTDet_PCViTknn_Base_FRPN_nwpuH_FastConvPre"
OUT_DIR4 = "ViTDet_PCViT_knn_Base_FRPN_6x_fuse"
CONFIG_FILE4 = f"../configs/{EXPERIMENT}/{DATA}/{MODEL_NAME4}.py"
RESULT_FILE4 = f"../work_dirs/{DATA}/{OUT_DIR4}/best.pkl"

CONFIG_FILE = [CONFIG_FILE1, CONFIG_FILE2, CONFIG_FILE3, CONFIG_FILE4]
RESULT_FILE = [RESULT_FILE1, RESULT_FILE2, RESULT_FILE3, RESULT_FILE4]

# CONFIG_FILE = f"../configs/{EXPERIMENT}/{DATA}/{MODEL_NAME1}.py"
# RESULT_FILE = f"../work_dirs/{DATA}/{OUT_DIR1}/best.pkl"

def plot_pr_curve(config_file, result_file, metric="bbox"):
    """plot precison-recall curve based on testing results of pkl file.

        Args:
            config_file (list[list | tuple]): config file path.
            result_file (str): pkl file of testing results path.
            metric (str): Metrics to be evaluated. Options are
                'bbox', 'segm'.
    """
    
    cfg = Config.fromfile(config_file[0])
    # turn on test mode of dataset
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True

    # build dataset
    dataset = build_dataset(cfg.data.test)

    # load result file in pkl format
    pkl_results1 = mmcv.load(result_file[0])
    # convert pkl file (list[list | tuple | ndarray]) to json
    json_results1, _ = dataset.format_results(pkl_results1)
    # initialize COCO instance
    coco = COCO(annotation_file=cfg.data.test.ann_file)
    coco_gt = coco
    coco_dt = coco_gt.loadRes(json_results1[metric]) 
    # initialize COCOeval instance
    coco_eval = COCOeval(coco_gt, coco_dt, metric)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    # extract eval data
    precisions = coco_eval.eval["precision"]
    '''
    precisions[T, R, K, A, M]
    T: iou thresholds [0.5 : 0.05 : 0.95], idx from 0 to 9
    R: recall thresholds [0 : 0.01 : 1], idx from 0 to 100
    K: category, idx from 0 to ...
    A: area range, (all, small, medium, large), idx from 0 to 3
    M: max dets, (1, 10, 100), idx from 0 to 2
    '''
    pr_array_mean1 = \
    (precisions[0, :, 0, 0, 2] + precisions[0, :, 1, 0, 2] + precisions[0, :, 2, 0, 2] + \
    precisions[0, :, 3, 0, 2] + precisions[0, :, 4, 0, 2] + precisions[0, :, 5, 0, 2] + \
    precisions[0, :, 6, 0, 2] + precisions[0, :, 7, 0, 2] + precisions[0, :, 8, 0, 2] + \
    precisions[0, :, 9, 0, 2]) / 10


    # load result file in pkl format
    pkl_results2 = mmcv.load(result_file[1])
    # convert pkl file (list[list | tuple | ndarray]) to json
    json_results2, _ = dataset.format_results(pkl_results2)
    # initialize COCO instance
    coco = COCO(annotation_file=cfg.data.test.ann_file)
    coco_gt = coco
    coco_dt = coco_gt.loadRes(json_results2[metric]) 
    # initialize COCOeval instance
    coco_eval = COCOeval(coco_gt, coco_dt, metric)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    # extract eval data
    precisions = coco_eval.eval["precision"]
    '''
    precisions[T, R, K, A, M]
    T: iou thresholds [0.5 : 0.05 : 0.95], idx from 0 to 9
    R: recall thresholds [0 : 0.01 : 1], idx from 0 to 100
    K: category, idx from 0 to ...
    A: area range, (all, small, medium, large), idx from 0 to 3
    M: max dets, (1, 10, 100), idx from 0 to 2
    '''
    pr_array_mean2 = \
    (precisions[0, :, 0, 0, 2] + precisions[0, :, 1, 0, 2] + precisions[0, :, 2, 0, 2] + \
    precisions[0, :, 3, 0, 2] + precisions[0, :, 4, 0, 2] + precisions[0, :, 5, 0, 2] + \
    precisions[0, :, 6, 0, 2] + precisions[0, :, 7, 0, 2] + precisions[0, :, 8, 0, 2] + \
    precisions[0, :, 9, 0, 2]) / 10


    # load result file in pkl format
    pkl_results3 = mmcv.load(result_file[2])
    # convert pkl file (list[list | tuple | ndarray]) to json
    json_results3, _ = dataset.format_results(pkl_results3)
    # initialize COCO instance
    coco = COCO(annotation_file=cfg.data.test.ann_file)
    coco_gt = coco
    coco_dt = coco_gt.loadRes(json_results3[metric]) 
    # initialize COCOeval instance
    coco_eval = COCOeval(coco_gt, coco_dt, metric)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    # extract eval data
    precisions = coco_eval.eval["precision"]
    '''
    precisions[T, R, K, A, M]
    T: iou thresholds [0.5 : 0.05 : 0.95], idx from 0 to 9
    R: recall thresholds [0 : 0.01 : 1], idx from 0 to 100
    K: category, idx from 0 to ...
    A: area range, (all, small, medium, large), idx from 0 to 3
    M: max dets, (1, 10, 100), idx from 0 to 2
    '''
    pr_array_mean3 = \
    (precisions[0, :, 0, 0, 2] + precisions[0, :, 1, 0, 2] + precisions[0, :, 2, 0, 2] + \
    precisions[0, :, 3, 0, 2] + precisions[0, :, 4, 0, 2] + precisions[0, :, 5, 0, 2] + \
    precisions[0, :, 6, 0, 2] + precisions[0, :, 7, 0, 2] + precisions[0, :, 8, 0, 2] + \
    precisions[0, :, 9, 0, 2]) / 10


    # load result file in pkl format
    pkl_results4 = mmcv.load(result_file[3])
    # convert pkl file (list[list | tuple | ndarray]) to json
    json_results4, _ = dataset.format_results(pkl_results4)
    # initialize COCO instance
    coco = COCO(annotation_file=cfg.data.test.ann_file)
    coco_gt = coco
    coco_dt = coco_gt.loadRes(json_results4[metric]) 
    # initialize COCOeval instance
    coco_eval = COCOeval(coco_gt, coco_dt, metric)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    # extract eval data
    precisions = coco_eval.eval["precision"]
    '''
    precisions[T, R, K, A, M]
    T: iou thresholds [0.5 : 0.05 : 0.95], idx from 0 to 9
    R: recall thresholds [0 : 0.01 : 1], idx from 0 to 100
    K: category, idx from 0 to ...
    A: area range, (all, small, medium, large), idx from 0 to 3
    M: max dets, (1, 10, 100), idx from 0 to 2
    '''
    pr_array_mean4 = \
    (precisions[0, :, 0, 0, 2] + precisions[0, :, 1, 0, 2] + precisions[0, :, 2, 0, 2] + \
    precisions[0, :, 3, 0, 2] + precisions[0, :, 4, 0, 2] + precisions[0, :, 5, 0, 2] + \
    precisions[0, :, 6, 0, 2] + precisions[0, :, 7, 0, 2] + precisions[0, :, 8, 0, 2] + \
    precisions[0, :, 9, 0, 2]) / 10


    # pr_array1 = precisions[0, :, 0, 0, 2]
    # pr_array2 = precisions[1, :, 0, 0, 2] 
    # pr_array3 = precisions[2, :, 0, 0, 2] 
    # pr_array4 = precisions[3, :, 0, 0, 2] 
    # pr_array5 = precisions[4, :, 0, 0, 2] 
    # pr_array6 = precisions[5, :, 0, 0, 2] 
    # pr_array7 = precisions[6, :, 0, 0, 2] 
    # pr_array8 = precisions[7, :, 0, 0, 2] 
    # pr_array9 = precisions[8, :, 0, 0, 2] 
    # pr_array10 = precisions[9, :, 0, 0, 2] 

    x = np.arange(0.0, 1.01, 0.01)
    # plot PR curve
    plt.plot(x, pr_array_mean1, label="PCViT")
    plt.plot(x, pr_array_mean2, label="+MPDJ")
    plt.plot(x, pr_array_mean3, label="+KNN")
    plt.plot(x, pr_array_mean4, label="+FRPN")
    # plt.plot(x, pr_array1, label="iou=0.5")
    # plt.plot(x, pr_array2, label="iou=0.55")
    # plt.plot(x, pr_array3, label="iou=0.6")
    # plt.plot(x, pr_array4, label="iou=0.65")
    # plt.plot(x, pr_array5, label="iou=0.7")
    # plt.plot(x, pr_array6, label="iou=0.75")
    # plt.plot(x, pr_array7, label="iou=0.8")
    # plt.plot(x, pr_array8, label="iou=0.85")
    # plt.plot(x, pr_array9, label="iou=0.9")
    # plt.plot(x, pr_array10, label="iou=0.95")

    plt.xlabel("recall")
    plt.ylabel("precison")
    plt.xlim(0, 1.0)
    plt.ylim(0, 1.01)
    plt.grid(True)
    plt.legend(loc="lower left")
    plt.show()

if __name__ == "__main__":
    plot_pr_curve(config_file=CONFIG_FILE, result_file=RESULT_FILE, metric="bbox")

    


    
