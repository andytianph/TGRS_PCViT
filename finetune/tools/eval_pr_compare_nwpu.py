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
config = {
            "font.family": 'serif',
            "font.size": 14,# 相当于小四大小
            "font.serif": ["Times New Roman"] + plt.rcParams["font.serif"],
            'axes.unicode_minus': False # 处理负号，即-号
         }
plt.rcParams.update(config)

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from mmcv import Config
from mmdet.datasets import build_dataset


NWPU_CATEGORIES = ["Airplane", "Ship", "Storage tank", "Baseball diamond", "Tennis court",  
           "Basketball court", "Ground track field", "Harbor", "Bridge", "Vehicle"]

EXPERIMENT = "a_TGRS"
DATA = "nwpu_split1"

MODEL_NAME1 = "faster_rcnn_r50_fpn"
OUT_DIR1 = "faster_rcnn_r50_fpn_6x"
CONFIG_FILE1 = f"configs/{EXPERIMENT}/{DATA}/{MODEL_NAME1}.py"
RESULT_FILE1 = f"work_dirs/{DATA}/{OUT_DIR1}/best.pkl"

MODEL_NAME2 = "mask_rcnn_r50_fpn"
OUT_DIR2 = "mask_rcnn_r50_fpn_6x"
CONFIG_FILE2 = f"configs/{EXPERIMENT}/{DATA}/{MODEL_NAME2}.py"
RESULT_FILE2 = f"work_dirs/{DATA}/{OUT_DIR2}/best.pkl"

MODEL_NAME3 = "oriented_rcnn_r50_fpn"
OUT_DIR3 = "oriented_rcnn_r50_fpn_6x_faster"
# CONFIG_FILE4 = f"configs/{EXPERIMENT}/{DATA}/{MODEL_NAME4}.py"
CONFIG_FILE3 = f"work_dirs/{DATA}/{OUT_DIR3}/{MODEL_NAME3}.py"
RESULT_FILE3 = f"work_dirs/{DATA}/{OUT_DIR3}/best.pkl"

MODEL_NAME4 = "redet_re50_refpn_6x_faster"
OUT_DIR4 = "redet_re50_refpn_6x_faster_regclsagnosticT_border0"
# CONFIG_FILE5 = f"configs/{EXPERIMENT}/{DATA}/{MODEL_NAME5}.py"
CONFIG_FILE4 = f"work_dirs/{DATA}/{OUT_DIR4}/{MODEL_NAME4}.py"
RESULT_FILE4 = f"work_dirs/{DATA}/{OUT_DIR4}/best.pkl"

MODEL_NAME5 = "mask_rcnn_swin-t-p4-w7_fpn"
OUT_DIR5 = "mask_rcnn_swin-t-p4-w7_fpn"
CONFIG_FILE5 = f"configs/{EXPERIMENT}/{DATA}/{MODEL_NAME5}.py"
RESULT_FILE5 = f"work_dirs/{DATA}/{OUT_DIR5}/best.pkl"

MODEL_NAME6 = "ViTDet_ViT_Base_FPN"
OUT_DIR6 = "ViTDet_ViT_Base_FPN_6x"
CONFIG_FILE6 = f"configs/{EXPERIMENT}/{DATA}/{MODEL_NAME6}.py"
RESULT_FILE6 = f"work_dirs/{DATA}/{OUT_DIR6}/best.pkl"

MODEL_NAME7 = "ViTDet_ViTAE_Base_FPN"
OUT_DIR7 = "ViTDet_ViTAE_Base_FPN_6x"
CONFIG_FILE7 = f"configs/{EXPERIMENT}/{DATA}/{MODEL_NAME7}.py"
RESULT_FILE7 = f"work_dirs/{DATA}/{OUT_DIR7}/best.pkl"

MODEL_NAME8 = "ViTDet_PCViTknn_Base_FRPN_nwpuH_FastConvPre"
OUT_DIR8 = "ViTDet_PCViT_knn_Base_FRPN_6x_fuse"
CONFIG_FILE8 = f"configs/{EXPERIMENT}/{DATA}/{MODEL_NAME8}.py"
RESULT_FILE8 = f"work_dirs/{DATA}/{OUT_DIR8}/best.pkl"



# CONFIG_FILE = [CONFIG_FILE1, CONFIG_FILE2, CONFIG_FILE3, CONFIG_FILE4, CONFIG_FILE5, CONFIG_FILE6]
# RESULT_FILE = [RESULT_FILE1, RESULT_FILE2, RESULT_FILE3, RESULT_FILE4, RESULT_FILE5, RESULT_FILE6]

CONFIG_FILE = [CONFIG_FILE1, CONFIG_FILE2, CONFIG_FILE3, CONFIG_FILE4, CONFIG_FILE5, CONFIG_FILE6, CONFIG_FILE7, CONFIG_FILE8]
RESULT_FILE = [RESULT_FILE1, RESULT_FILE2, RESULT_FILE3, RESULT_FILE4, RESULT_FILE5, RESULT_FILE6, RESULT_FILE7, RESULT_FILE8]

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

    ''' CONFIG_FILE1 '''
    # load result file in pkl format
    pkl_results1 = mmcv.load(result_file[0])
    # convert pkl file (list[list | tuple | ndarray]) to json
    json_results1, _ = dataset.format_results(pkl_results1)
    # initialize COCO instance
    coco1 = COCO(annotation_file=cfg.data.test.ann_file)
    coco_gt1 = coco1
    coco_dt1 = coco_gt1.loadRes(json_results1[metric]) 
    # initialize COCOeval instance
    coco_eval1 = COCOeval(coco_gt1, coco_dt1, metric)
    coco_eval1.evaluate()
    coco_eval1.accumulate()
    coco_eval1.summarize()
    # extract eval data
    precisions1 = coco_eval1.eval["precision"]

    ''' CONFIG_FILE2 '''
    # load result file in pkl format
    pkl_results2 = mmcv.load(result_file[1])
    # convert pkl file (list[list | tuple | ndarray]) to json
    json_results2, _ = dataset.format_results(pkl_results2)
    # initialize COCO instance
    coco2 = COCO(annotation_file=cfg.data.test.ann_file)
    coco_gt2 = coco2
    coco_dt2 = coco_gt2.loadRes(json_results2[metric]) 
    # initialize COCOeval instance
    coco_eval2 = COCOeval(coco_gt2, coco_dt2, metric)
    coco_eval2.evaluate()
    coco_eval2.accumulate()
    coco_eval2.summarize()
    # extract eval data
    precisions2 = coco_eval2.eval["precision"]

    ''' CONFIG_FILE3 '''
    # load result file in pkl format
    pkl_results3 = mmcv.load(result_file[2])
    # convert pkl file (list[list | tuple | ndarray]) to json
    json_results3, _ = dataset.format_results(pkl_results3)
    # initialize COCO instance
    coco3 = COCO(annotation_file=cfg.data.test.ann_file)
    coco_gt3 = coco3
    coco_dt3 = coco_gt3.loadRes(json_results3[metric]) 
    # initialize COCOeval instance
    coco_eval3 = COCOeval(coco_gt3, coco_dt3, metric)
    coco_eval3.evaluate()
    coco_eval3.accumulate()
    coco_eval3.summarize()
    # extract eval data
    precisions3 = coco_eval3.eval["precision"]

    ''' CONFIG_FILE4 '''
    # load result file in pkl format
    pkl_results4 = mmcv.load(result_file[3])
    # convert pkl file (list[list | tuple | ndarray]) to json
    json_results4, _ = dataset.format_results(pkl_results4)
    # initialize COCO instance
    coco4 = COCO(annotation_file=cfg.data.test.ann_file)
    coco_gt4 = coco4
    coco_dt4 = coco_gt4.loadRes(json_results4[metric])
    # initialize COCOeval instance
    coco_eval4 = COCOeval(coco_gt4, coco_dt4, metric)
    coco_eval4.evaluate()
    coco_eval4.accumulate()
    coco_eval4.summarize()
    # extract eval data
    precisions4 = coco_eval4.eval["precision"]

    ''' CONFIG_FILE5 '''
    # load result file in pkl format
    pkl_results5 = mmcv.load(result_file[4])
    # convert pkl file (list[list | tuple | ndarray]) to json
    json_results5, _ = dataset.format_results(pkl_results5)
    # initialize COCO instance
    coco5 = COCO(annotation_file=cfg.data.test.ann_file)
    coco_gt5 = coco5
    coco_dt5 = coco_gt5.loadRes(json_results5[metric]) 
    # initialize COCOeval instance
    coco_eval5 = COCOeval(coco_gt5, coco_dt5, metric)
    coco_eval5.evaluate()
    coco_eval5.accumulate()
    coco_eval5.summarize()
    # extract eval data
    precisions5 = coco_eval5.eval["precision"]

    ''' CONFIG_FILE6 '''
    # load result file in pkl format
    pkl_results6 = mmcv.load(result_file[5])
    # convert pkl file (list[list | tuple | ndarray]) to json
    json_results6, _ = dataset.format_results(pkl_results6)
    # initialize COCO instance
    coco6 = COCO(annotation_file=cfg.data.test.ann_file)
    coco_gt6 = coco6
    coco_dt6 = coco_gt6.loadRes(json_results6[metric]) 
    # initialize COCOeval instance
    coco_eval6 = COCOeval(coco_gt6, coco_dt6, metric)
    coco_eval6.evaluate()
    coco_eval6.accumulate()
    coco_eval6.summarize()
    # extract eval data
    precisions6 = coco_eval6.eval["precision"]

    ''' CONFIG_FILE7 '''
    # load result file in pkl format
    pkl_results7 = mmcv.load(result_file[6])
    # convert pkl file (list[list | tuple | ndarray]) to json
    json_results7, _ = dataset.format_results(pkl_results7)
    # initialize COCO instance
    coco7 = COCO(annotation_file=cfg.data.test.ann_file)
    coco_gt7 = coco7
    coco_dt7 = coco_gt7.loadRes(json_results7[metric]) 
    # initialize COCOeval instance
    coco_eval7 = COCOeval(coco_gt7, coco_dt7, metric)
    coco_eval7.evaluate()
    coco_eval7.accumulate()
    coco_eval7.summarize()
    # extract eval data
    precisions7 = coco_eval7.eval["precision"]

    ''' CONFIG_FILE8 '''
    # load result file in pkl format
    pkl_results8 = mmcv.load(result_file[7])
    # convert pkl file (list[list | tuple | ndarray]) to json
    json_results8, _ = dataset.format_results(pkl_results8)
    # initialize COCO instance
    coco8 = COCO(annotation_file=cfg.data.test.ann_file)
    coco_gt8 = coco8
    coco_dt8 = coco_gt8.loadRes(json_results8[metric]) 
    # initialize COCOeval instance
    coco_eval8 = COCOeval(coco_gt8, coco_dt8, metric)
    coco_eval8.evaluate()
    coco_eval8.accumulate()
    coco_eval8.summarize()
    # extract eval data
    precisions8 = coco_eval8.eval["precision"]

    '''
    precisions[T, R, K, A, M]
    T: iou thresholds [0.5 : 0.05 : 0.95], idx from 0 to 9
    R: recall thresholds [0 : 0.01 : 1], idx from 0 to 100
    K: category, idx from 0 to ...
    A: area range, (all, small, medium, large), idx from 0 to 3
    M: max dets, (1, 10, 100), idx from 0 to 2
    '''


    for i in range(10):
        pr_array1 = precisions1[0, :, i, 0, 2]
        pr_array2 = precisions2[0, :, i, 0, 2]
        pr_array3 = precisions3[0, :, i, 0, 2]
        pr_array4 = precisions4[0, :, i, 0, 2]
        pr_array5 = precisions5[0, :, i, 0, 2]
        pr_array6 = precisions6[0, :, i, 0, 2]
        pr_array7 = precisions7[0, :, i, 0, 2]
        pr_array8 = precisions8[0, :, i, 0, 2]

        x = np.arange(0.0, 1.01, 0.01)
        fig = plt.figure()
        # plot PR curve
        plt.plot(x, pr_array1, label="Faster R-CNN", color='m')
        plt.plot(x, pr_array2, label="Mask R-CNN", color='k')
        plt.plot(x, pr_array3, label="Oriented R-CNN", color='deepskyblue')
        plt.plot(x, pr_array4, label="ReDet", color='g')
        plt.plot(x, pr_array5, label="Swin-B", color='c')
        plt.plot(x, pr_array6, label="ViT-B", color='m')
        plt.plot(x, pr_array7, label="ViTAE-B", color='y')
        plt.plot(x, pr_array8, label="PCViT", color='r')

        plt.title(NWPU_CATEGORIES[i])
        plt.xlabel("Recall")
        plt.ylabel("Precison")
        plt.xlim(0, 1.0)
        plt.ylim(0, 1.01)
        plt.grid(True)
        plt.legend(loc="lower left")
        # plt.show()
        plt.savefig("/home/data/jiyuqing/.tph/tph_vitdet/work_dirs/nwpu_split1/compare_pr_update/" + NWPU_CATEGORIES[i] + ".png", dpi=600)
        plt.close()



    pr_array_mean1 = \
    (precisions1[0, :, 0, 0, 2] + precisions1[0, :, 1, 0, 2] + precisions1[0, :, 2, 0, 2] + \
    precisions1[0, :, 3, 0, 2] + precisions1[0, :, 4, 0, 2] + precisions1[0, :, 5, 0, 2] + \
    precisions1[0, :, 6, 0, 2] + precisions1[0, :, 7, 0, 2] + precisions1[0, :, 8, 0, 2] + \
    precisions1[0, :, 9, 0, 2]) / 10

    pr_array_mean2 = \
    (precisions2[0, :, 0, 0, 2] + precisions2[0, :, 1, 0, 2] + precisions2[0, :, 2, 0, 2] + \
    precisions2[0, :, 3, 0, 2] + precisions2[0, :, 4, 0, 2] + precisions2[0, :, 5, 0, 2] + \
    precisions2[0, :, 6, 0, 2] + precisions2[0, :, 7, 0, 2] + precisions2[0, :, 8, 0, 2] + \
    precisions2[0, :, 9, 0, 2]) / 10

    pr_array_mean3 = \
    (precisions3[0, :, 0, 0, 2] + precisions3[0, :, 1, 0, 2] + precisions3[0, :, 2, 0, 2] + \
    precisions3[0, :, 3, 0, 2] + precisions3[0, :, 4, 0, 2] + precisions3[0, :, 5, 0, 2] + \
    precisions3[0, :, 6, 0, 2] + precisions3[0, :, 7, 0, 2] + precisions3[0, :, 8, 0, 2] + \
    precisions3[0, :, 9, 0, 2]) / 10

    pr_array_mean4 = \
    (precisions4[0, :, 0, 0, 2] + precisions4[0, :, 1, 0, 2] + precisions4[0, :, 2, 0, 2] + \
    precisions4[0, :, 3, 0, 2] + precisions4[0, :, 4, 0, 2] + precisions4[0, :, 5, 0, 2] + \
    precisions4[0, :, 6, 0, 2] + precisions4[0, :, 7, 0, 2] + precisions4[0, :, 8, 0, 2] + \
    precisions4[0, :, 9, 0, 2]) / 10

    pr_array_mean5 = \
    (precisions5[0, :, 0, 0, 2] + precisions5[0, :, 1, 0, 2] + precisions5[0, :, 2, 0, 2] + \
    precisions5[0, :, 3, 0, 2] + precisions5[0, :, 4, 0, 2] + precisions5[0, :, 5, 0, 2] + \
    precisions5[0, :, 6, 0, 2] + precisions5[0, :, 7, 0, 2] + precisions5[0, :, 8, 0, 2] + \
    precisions5[0, :, 9, 0, 2]) / 10

    pr_array_mean6 = \
    (precisions6[0, :, 0, 0, 2] + precisions6[0, :, 1, 0, 2] + precisions6[0, :, 2, 0, 2] + \
    precisions6[0, :, 3, 0, 2] + precisions6[0, :, 4, 0, 2] + precisions6[0, :, 5, 0, 2] + \
    precisions6[0, :, 6, 0, 2] + precisions6[0, :, 7, 0, 2] + precisions6[0, :, 8, 0, 2] + \
    precisions6[0, :, 9, 0, 2]) / 10

    pr_array_mean7 = \
    (precisions7[0, :, 0, 0, 2] + precisions7[0, :, 1, 0, 2] + precisions7[0, :, 2, 0, 2] + \
    precisions7[0, :, 3, 0, 2] + precisions7[0, :, 4, 0, 2] + precisions7[0, :, 5, 0, 2] + \
    precisions7[0, :, 6, 0, 2] + precisions7[0, :, 7, 0, 2] + precisions7[0, :, 8, 0, 2] + \
    precisions7[0, :, 9, 0, 2]) / 10

    pr_array_mean8 = \
    (precisions8[0, :, 0, 0, 2] + precisions8[0, :, 1, 0, 2] + precisions8[0, :, 2, 0, 2] + \
    precisions8[0, :, 3, 0, 2] + precisions8[0, :, 4, 0, 2] + precisions8[0, :, 5, 0, 2] + \
    precisions8[0, :, 6, 0, 2] + precisions8[0, :, 7, 0, 2] + precisions8[0, :, 8, 0, 2] + \
    precisions8[0, :, 9, 0, 2]) / 10

    x = np.arange(0.0, 1.01, 0.01)
    fig = plt.figure()
    # plot PR curve
    plt.plot(x, pr_array_mean1, label="Faster R-CNN", color='m')
    plt.plot(x, pr_array_mean2, label="Mask R-CNN", color='k')
    plt.plot(x, pr_array_mean3, label="Oriented R-CNN", color='deepskyblue')
    plt.plot(x, pr_array_mean4, label="ReDet", color='g')
    plt.plot(x, pr_array_mean5, label="Swin-B", color='c')
    plt.plot(x, pr_array_mean6, label="ViT-B", color='m')
    plt.plot(x, pr_array_mean7, label="ViTAE-B", color='y')
    plt.plot(x, pr_array_mean8, label="PCViT", color='r')

    # plt.title(NWPU_CATEGORIES[i])
    plt.xlabel("Recall")
    plt.ylabel("Precison")
    plt.xlim(0, 1.0)
    plt.ylim(0, 1.01)
    plt.grid(True)
    plt.legend(loc="lower left")
    # plt.show()
    plt.savefig("/home/data/jiyuqing/.tph/tph_vitdet/work_dirs/nwpu_split1/nwpu_compare_update.png", dpi=600)
    plt.close()



if __name__ == "__main__":
    plot_pr_curve(config_file=CONFIG_FILE, result_file=RESULT_FILE, metric="bbox")

