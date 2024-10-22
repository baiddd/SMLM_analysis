import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize
from scipy.misc import bytescale
from scipy.ndimage.filters import gaussian_filter
from skimage import exposure, io

from utils.msssim import MultiScaleSSIM


def im_norm(im, pmin=2, pmax=99.8, eps=1e-20):
    dtype = np.float32
    mi = np.percentile(im, pmin, keepdims=True).astype(dtype, copy=False)
    ma = np.percentile(im, pmax, keepdims=True).astype(dtype, copy=False)
    im_norm = (im - mi) / (ma - mi + eps)
    print(mi[0], ma[0])
    return im_norm


def align_image(a, b, preprocess=False):
    import imreg_dft as ird

    if preprocess:
        b = exposure.equalize_hist(b)
        # print(b.shape)
        b = scipy.ndimage.filters.gaussian_filter(b, sigma=(6, 6))
        b = scipy.misc.imresize(b, a.shape)
    ts = ird.translation(b, a)
    tvec = ts["tvec"].round(4)
    # the Transformed IMaGe.
    a = ird.transform_img(a, tvec=tvec)
    if preprocess:
        a = scipy.misc.imresize(a, b.shape)
    return a


def df_generator_3(
    DirBase_1,
    DirBase_2,
    DirBase_3,
    key,
    crop_cord=(0, 0),
    chopsize=2560 // 2,
    FrameNb=300,
):
    DIR4Workspace_1 = os.listdir(DirBase_1)
    DIR4Workspace_2 = os.listdir(DirBase_2)
    DIR4Workspace_3 = os.listdir(DirBase_3)

    CellName = []
    for name in DIR4Workspace_2:
        # print(name.split('.')[0])
        if key in name:
            CellName.append(name.split(".")[0])
    CellName = np.unique(CellName)
    x1, y1 = crop_cord
    x2, y2 = (chopsize, chopsize)
    df_3 = pd.DataFrame(
        columns=["CellName", "FrameRange", "SSIM_sparse", "SSIM_both_raw", "ErrorMap"]
    )

    df_2 = pd.DataFrame(
        columns=["CellName", "FrameRange", "SSIM_sparse", "SSIM_both_raw", "ErrorMap"]
    )
    df_1 = pd.DataFrame(
        columns=["CellName", "FrameRange", "SSIM_sparse", "SSIM_both_raw", "ErrorMap"]
    )
    for name in CellName:
        List4Image_2 = [n for n in DIR4Workspace_2 if "{}.".format(name) in n]
        FrameNbInit = List4Image_2[0].split("(")[1]
        FrameNbInit = FrameNbInit.split(",")[0]
        FrameNbInit = int(FrameNbInit)
        List4CertainRange_2 = [
            n
            for n in List4Image_2
            if "({}, {})".format(FrameNbInit, FrameNbInit + FrameNb) in n
        ]
        # NameOfInput = [n for n in List4CertainRange_Mixed if 'real_A' in n]
        NameOfOutput = [n for n in List4CertainRange_2 if "real_B_" in n]
        # NameOfWF = [n for n in List4CertainRange_Mixed if '_lr_inputs' in n]
        NameOfReco = [n for n in List4CertainRange_2 if "reco" in n]
        NameOIn = [n for n in List4CertainRange_2 if "real_A_" in n]
        NameOfErroMap = [n for n in List4CertainRange_2 if "squirrel_error_map" in n]

        histout = io.imread(
            os.path.join(DirBase_3, "".join([str(elem) for elem in NameOfOutput]))
        )
        histin = io.imread(
            os.path.join(DirBase_3, "".join([str(elem) for elem in NameOIn]))
        )

        histout_crop = histout[x1 : x1 + x2, y1 : y1 + y2]
        histin_crop = histin[x1 : x1 + x2, y1 : y1 + y2]
        SSIM_sparse = MultiScaleSSIM((histin_crop), (histout_crop), max_val=1)

        reco_1 = io.imread(
            os.path.join(DirBase_1, "".join([str(elem) for elem in NameOfReco]))
        )
        # reco_1 = align_image(reco_1,histout)
        reco_1_crop = reco_1[x1 : x1 + x2, y1 : y1 + y2]
        ErroMap_1 = io.imread(
            os.path.join(DirBase_1, "".join([str(elem) for elem in NameOfErroMap]))
        )

        reco_2 = io.imread(
            os.path.join(DirBase_2, "".join([str(elem) for elem in NameOfReco]))
        )
        # reco_2 = align_image(reco_2,histout)
        reco_2_crop = reco_2[x1 : x1 + x2, y1 : y1 + y2]
        ErroMap_2 = io.imread(
            os.path.join(DirBase_2, "".join([str(elem) for elem in NameOfErroMap]))
        )

        reco_3 = io.imread(
            os.path.join(DirBase_3, "".join([str(elem) for elem in NameOfReco]))
        )
        # reco_2 = align_image(reco_2,histout)
        reco_3_crop = reco_3[x1 : x1 + x2, y1 : y1 + y2]
        ErroMap_3 = io.imread(
            os.path.join(DirBase_3, "".join([str(elem) for elem in NameOfErroMap]))
        )
        print(
            reco_1_crop.min(), reco_1_crop.max(), histout_crop.min(), histout_crop.max()
        )
        ssim_both_raw_1 = MultiScaleSSIM((reco_1_crop), (histout_crop), max_val=1)

        IntDen_ErrMap_1 = ErroMap_1.sum()

        ssim_both_raw_2 = MultiScaleSSIM((reco_2_crop), (histout_crop), max_val=1)
        IntDen_ErrMap_2 = ErroMap_2.sum()

        ssim_both_raw_3 = MultiScaleSSIM((reco_3_crop), (histout_crop), max_val=1)
        IntDen_ErrMap_3 = ErroMap_3.sum()

        df_1 = df_1.append(
            {
                "CellName": "{}".format(name),
                "FrameRange": "[{},{}]".format(FrameNbInit, FrameNbInit + FrameNb),
                "SSIM_both_raw": ssim_both_raw_1,
                #'SSIM_both_norm' : ssim_both_norm_1,
                "SSIM_sparse": SSIM_sparse,
                "ErrorMap": IntDen_ErrMap_1,
            },
            ignore_index=True,
        )

        df_2 = df_2.append(
            {
                "CellName": "{}".format(name),
                "FrameRange": "[{},{}]".format(FrameNbInit, FrameNbInit + FrameNb),
                "SSIM_both_raw": ssim_both_raw_2,
                #'SSIM_both_norm' : ssim_both_norm_2,
                "SSIM_sparse": SSIM_sparse,
                "ErrorMap": IntDen_ErrMap_2,
            },
            ignore_index=True,
        )
        df_3 = df_3.append(
            {
                "CellName": "{}".format(name),
                "FrameRange": "[{},{}]".format(FrameNbInit, FrameNbInit + FrameNb),
                "SSIM_both_raw": ssim_both_raw_3,
                #'SSIM_both_norm' : ssim_both_norm_2,
                "SSIM_sparse": SSIM_sparse,
                "ErrorMap": IntDen_ErrMap_3,
            },
            ignore_index=True,
        )

        print(name)
    return df_1, df_2, df_3


# plt.savefig("./scatter+box")


def df_generator(
    DirBase_1, DirBase_2, keyword, crop_cord=(0, 0), chopsize=2560 // 2, FrameNb=300
):
    DIR4Workspace_1 = os.listdir(DirBase_1)
    DIR4Workspace_2 = os.listdir(DirBase_2)

    CellName = []
    for name in DIR4Workspace_2:
        # print(name.split('.')[0])
        if keyword in name:
            CellName.append(name.split(".")[0])
    CellName = np.unique(CellName)
    x1, y1 = crop_cord
    x2, y2 = (chopsize, chopsize)
    df_2 = pd.DataFrame(
        columns=["CellName", "FrameRange", "SSIM_both_raw", "SSIM_sparse", "ErrorMap"]
    )
    df_1 = pd.DataFrame(
        columns=["CellName", "FrameRange", "SSIM_both_raw", "SSIM_sparse", "ErrorMap"]
    )
    for name in CellName:
        List4Image_2 = [n for n in DIR4Workspace_2 if "{}.".format(name) in n]
        FrameNbInit = List4Image_2[0].split("(")[1]
        FrameNbInit = FrameNbInit.split(",")[0]
        FrameNbInit = int(FrameNbInit)
        List4CertainRange_2 = [
            n
            for n in List4Image_2
            if "({}, {})".format(FrameNbInit, FrameNbInit + FrameNb) in n
        ]
        # NameOfInput = [n for n in List4CertainRange_Mixed if 'real_A' in n]
        NameOfOutput = [n for n in List4CertainRange_2 if "real_B_" in n]
        # NameOfOutput = [n for n in List4CertainRange_2 if 'reco' in n]
        # NameOfWF = [n for n in List4CertainRange_Mixed if '_lr_inputs' in n]
        NameOfReco = [n for n in List4CertainRange_2 if "reco" in n]
        NameOIn = [n for n in List4CertainRange_2 if "real_A_" in n]
        NameOfErroMap = [n for n in List4CertainRange_2 if "squirrel_error_map" in n]
        # NameOfVar = [n for n in List4CertainRange_2 if 'epistemic_uncertainty_' in n]

        histout = io.imread(
            os.path.join(DirBase_2, "".join([str(elem) for elem in NameOfOutput]))
        )
        histin = io.imread(
            os.path.join(DirBase_2, "".join([str(elem) for elem in NameOIn]))
        )

        histout_crop = histout[x1 : x1 + x2, y1 : y1 + y2]
        histin_crop = histin[x1 : x1 + x2, y1 : y1 + y2]
        SSIM_sparse = MultiScaleSSIM((histin_crop), (histout_crop), max_val=1)
        reco_1 = io.imread(
            os.path.join(DirBase_1, "".join([str(elem) for elem in NameOfReco]))
        )
        # reco_1 = align_image(reco_1,histout)
        reco_1_crop = reco_1[x1 : x1 + x2, y1 : y1 + y2]
        ErroMap_1 = io.imread(
            os.path.join(DirBase_1, "".join([str(elem) for elem in NameOfErroMap]))
        )

        reco_2 = io.imread(
            os.path.join(DirBase_2, "".join([str(elem) for elem in NameOfReco]))
        )
        # reco_2 = align_image(reco_2,histout)
        reco_2_crop = reco_2[x1 : x1 + x2, y1 : y1 + y2]
        ErroMap_2 = io.imread(
            os.path.join(DirBase_2, "".join([str(elem) for elem in NameOfErroMap]))
        )

        # Var_dropout1 = io.imread(os.path.join(DirBase_1,''.join([str(elem) for elem in NameOfVar])))
        # Var_dropout_crop1=Var_dropout1[x1:x1+x2,y1:y1+y2]
        # Var_dropout2 = io.imread(os.path.join(DirBase_2,''.join([str(elem) for elem in NameOfVar])))
        # Var_dropout_crop2=Var_dropout2[x1:x1+x2,y1:y1+y2]

        print(
            reco_1_crop.min(), reco_1_crop.max(), histout_crop.min(), histout_crop.max()
        )
        ssim_both_raw_1 = MultiScaleSSIM((reco_1_crop), (histout_crop), max_val=1)

        gt_temp = histout_crop
        IntDen_ErrMap_1 = ErroMap_1.sum()

        ssim_both_raw_2 = MultiScaleSSIM((reco_2_crop), (histout_crop), max_val=1)
        IntDen_ErrMap_2 = ErroMap_2.sum()

        df_1 = df_1.append(
            {
                "CellName": "{}".format(name),
                "FrameRange": "[{},{}]".format(FrameNbInit, FrameNbInit + FrameNb),
                "SSIM_both_raw": ssim_both_raw_1,
                "SSIM_sparse": SSIM_sparse,
                "ErrorMap": IntDen_ErrMap_1,
            },
            ignore_index=True,
        )

        df_2 = df_2.append(
            {
                "CellName": "{}".format(name),
                "FrameRange": "[{},{}]".format(FrameNbInit, FrameNbInit + FrameNb),
                "SSIM_both_raw": ssim_both_raw_2,
                "SSIM_sparse": SSIM_sparse,
                "ErrorMap": IntDen_ErrMap_2,
            },
            ignore_index=True,
        )
        print(name)
    return df_1, df_2
