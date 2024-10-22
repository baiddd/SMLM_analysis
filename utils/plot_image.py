import fnmatch
import os
import sys

import imreg_dft as ird
import matplotlib
import matplotlib.font_manager as fm
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import scipy
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, zoomed_inset_axes
from PIL import Image
from scipy.misc import bytescale
from skimage import exposure
from skimage.measure import profile_line

from utils.msssim import MultiScaleSSIM

pixel_size = 20


def zoom(image, lim, zoom=3, loc=2, loc1=2, loc2=4, f=1, **kwargs):
    """zoom in the image at the location of lim
    Args:
        image: 2D image
        lim: (x1, y1, x2, y2)
        zoom: zoom rate
        loc: location of the zoomed image
        loc1: location of the inset axes
        loc2: location of the inset axes
        f: factor to adjust the zoomed image
        kwargs: other parameters for imshow

    """
    # --------
    ax = plt.gca()
    axins = zoomed_inset_axes(ax, zoom, loc=loc)  # zoom = 6
    print(np.shape(image))
    axins.imshow(image, interpolation="nearest", origin="lower", **kwargs)
    x1, y1, x2, y2 = lim
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)

    plt.xticks(visible=False)
    plt.yticks(visible=False)

    # draw a bbox of the region of the inset axes in the parent axes and
    # connecting lines between the bbox and the inset axes area
    mark_inset(ax, axins, loc1=loc1, loc2=loc2, fc="none", ec="0.8", lw=2)


def zoom_line(image, lim, line, c="y", zoom=3, loc=2, loc1=2, loc2=4, **kwargs):
    """zoom in the image with line at the location of lim

    Args:
        image: 2D image
        lim: (x1, y1, x2, y2)
        line: (x0, y0, x1, y1)
        c: color of the line
        zoom: zoom rate
        loc: location of the zoomed image
        loc1: location of the inset axes
        loc2: location of the inset axes
        kwargs: other parameters for imshow

    """
    # --------
    ax = plt.gca()
    axins = zoomed_inset_axes(ax, zoom, loc=loc)  # zoom = 6
    x0, y0, x1, y1 = line
    axins.imshow(image, interpolation="nearest", origin="lower", **kwargs)
    plt.plot([x0, x1], [y0, y1], "{}-".format(c), linewidth=2 * zoom, alpha=1)
    plt.plot([x0, x1], [y0, y1], "{}-".format(c), linewidth=1 * zoom)
    plt.arrow(x0, y0, (x1 - x0), (y1 - y0), fc="y", ec="y", head_width=zoom)
    # sub region of the original image
    x1, y1, x2, y2 = lim
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)

    plt.xticks(visible=False)
    plt.yticks(visible=False)

    # draw a bbox of the region of the inset axes in the parent axes and
    # connecting lines between the bbox and the inset axes area
    mark_inset(ax, axins, loc1=loc1, loc2=loc2, fc="none", ec="0.8", lw=2)


def scalebar(scale_factor=500 / pixel_size, down_scale=1):
    """
    Add scale bar to the image
    Args:
        scale_factor: scale factor
        down_scale: down scale factor
    """

    ax = plt.gca()
    fontprops = fm.FontProperties(size=30)
    asb = AnchoredSizeBar(
        ax.transData,
        scale_factor,
        "",  # r"500nm",
        loc=3,
        color="white",
        size_vertical=2.0 / down_scale,
        pad=0.1,
        borderpad=0.8,
        sep=5,
        fontproperties=fontprops,
        frameon=False,
    )
    ax.add_artist(asb)


def align_image(a, b, preprocess=False):
    """
    Align image a to image b
    Args:
        a: image to be aligned
        b: reference image
        preprocess: whether to preprocess the image
    Returns:
        aligned image
    """
    if preprocess:
        b = exposure.equalize_hist(b)
        # print(b.shape)
        b = scipy.ndimage.filters.gaussian_filter(b, sigma=(4, 4))
        b = scipy.misc.imresize(b, a.shape)
    ts = ird.translation(b, a)
    tvec = ts["tvec"].round(4)
    # the Transformed IMaGe.
    # a = scipy.ndimage.filters.gaussian_filter(a, sigma=(4, 4))
    a = ird.transform_img(a, tvec=tvec)
    if preprocess:
        a = scipy.misc.imresize(a, b.shape)
    return a


def plot_image1(
    data,
    frame_num=300,
    model_number=2,
    zoom_lim=(0, 0, 40, 40),
    line=(215, 200, 255, 240),  # x0,y0,x1,y1
    add_title=True,
    save_fig=False,
    show_zoom=True,
    savedir="./output",
    name="ANNA-PALM_model",
    title_c="Baseline",
    title_d="Retrained model",
):
    """
    Plot the image
    Args:

        data: tuple of images
        frame_num: number of frames
        model_number: number of models
        zoom_lim: zoom limit
        line: line to zoom
        add_title: whether to add title
        save_fig: whether to save the figure
        show_zoom: whether to show zoomed image
        savedir: directory to save the figure

    """
    print("ploting")
    from datetime import date

    today = str(date.today()).replace("-", "_")
    if model_number == 3:
        (wf, histin, gt, Merged_1, Merged_2, Merged_3, Reco_1, Reco_2, Reco_3) = data
    else:
        (wf, histin, gt, Merged_1, Merged_2, Reco_1, Reco_2) = data

    zoom_rate = 4  # 10  #4
    font_size = 45
    loc1 = 2  # 4
    loc2 = 3  # 2
    loc = 2  # 1
    fontsize = 45
    zoomsize = 50

    x0, y0, x1, y1 = line
    plt.figure(figsize=(40, 40))
    gs = gridspec.GridSpec(2, 2)
    gs.update(wspace=0.02, hspace=0.02)  # set the spacing between axes.
    matplotlib.rc("axes", edgecolor="w", linewidth=5)

    # -------------input-target---------------
    FigSize = 10
    # plt.subplot(gs[0])
    plt.figure(figsize=(FigSize, FigSize))
    plt.axis("off")
    plt.imshow(wf, interpolation="nearest", origin="lower", cmap="gray", vmin=0)
    if add_title:
        plt.title("a. wide-field", fontsize=fontsize)
    scalebar(scale_factor=1000 / pixel_size / 4, down_scale=2)
    if show_zoom:
        zoom(
            wf,
            list(np.array(zoom_lim) // 4),
            zoom_rate,
            loc=loc,
            loc1=loc1,
            loc2=loc2,
            f=4,
            cmap="gray",
            vmin=0,
        )
    if save_fig:
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        plt.savefig(
            os.path.join(savedir, f"WF_{name}_{zoom_lim}_{today}.svg"),
            transparent=False,
        )

    vmax_in = 3

    # plt.subplot(gs[1])
    plt.figure(figsize=(FigSize, FigSize))
    #
    plt.axis("off")
    scalebar(scale_factor=1000 / pixel_size, down_scale=0.5)
    plt.imshow(histin, origin="lower", cmap="hot", vmin=0, vmax=vmax_in)
    if add_title:
        plt.title("b. PALM(k={})".format(frame_num), fontsize=fontsize)
    if show_zoom:
        zoom(
            histin,
            zoom_lim,
            zoom_rate,
            loc=loc,
            loc1=loc1,
            loc2=loc2,
            cmap="hot",
            vmin=0,
            vmax=vmax_in,
        )
    if save_fig:
        plt.savefig(
            os.path.join(savedir, f"histin_{name}_{zoom_lim}_{today}.svg"),
            transparent=False,
        )
    print("b vmax:", vmax_in * 255)

    # -------------merged---------------

    # ax2 = plt.subplot(gs[2])
    plt.figure(figsize=(FigSize, FigSize))
    plt.axis("off")
    plt.imshow(Merged_1, origin="lower")
    plt.plot([x0, x1], [y0, y1], "y-", linewidth=5, alpha=0.3)
    plt.arrow(x0, y0, (x1 - x0), (y1 - y0), fc="y", ec="y", head_width=4)
    if add_title:
        plt.title("c. {}".format(title_c), fontsize=fontsize)
    # scalebar()
    scalebar(scale_factor=1000 / pixel_size, down_scale=0.5)
    if show_zoom:
        zoom_line(
            Merged_1,
            lim=zoom_lim,
            line=(x0, y0, x1, y1),
            zoom=zoom_rate,
            loc=loc,
            loc1=loc1,
            loc2=loc2,
            cmap="hot",
            vmin=0,
            vmax=1,
            c="y",
        )
    if save_fig:
        plt.savefig(
            os.path.join(savedir, f"baseline_{name}_{zoom_lim}_{today}.svg"),
            transparent=False,
        )

    # ax3 = plt.subplot(gs[3])
    plt.figure(figsize=(FigSize, FigSize))
    plt.axis("off")
    # scalebar()
    scalebar(scale_factor=1000 / pixel_size, down_scale=0.5)
    plt.imshow(Merged_2, origin="lower")
    plt.plot([x0, x1], [y0, y1], "y-", linewidth=5, alpha=0.3)
    plt.arrow(x0, y0, (x1 - x0), (y1 - y0), fc="y", ec="y", head_width=4)
    if add_title:
        plt.title("d. {}".format(title_d), fontsize=fontsize)
    if show_zoom:
        # zoom(Merged_d, zoom_lim, zoom_rate,loc=loc, loc1=loc1, loc2=loc2)
        zoom_line(
            Merged_2,
            lim=zoom_lim,
            line=(x0, y0, x1, y1),
            zoom=zoom_rate,
            loc=loc,
            loc1=loc1,
            loc2=loc2,
            cmap="hot",
            vmin=0,
            vmax=1,
            c="y",
        )
    if save_fig:
        plt.savefig(
            os.path.join(savedir, f"reco1_{name}_{zoom_lim}_{today}.svg"),
            transparent=False,
        )

    plt.figure(figsize=(FigSize, FigSize))
    plt.axis("off")
    # scalebar()
    scalebar(scale_factor=1000 / pixel_size, down_scale=0.5)
    plt.imshow(bytescale(gt), origin="lower", vmin=0, vmax=50, cmap="hot")
    plt.plot([x0, x1], [y0, y1], "y-", linewidth=5, alpha=0.3)
    plt.arrow(x0, y0, (x1 - x0), (y1 - y0), fc="y", ec="y", head_width=4)
    if add_title:
        plt.title("d. {}".format(title_d), fontsize=fontsize)
    if show_zoom:
        # zoom(Merged_d, zoom_lim, zoom_rate,loc=loc, loc1=loc1, loc2=loc2)
        zoom_line(
            bytescale(gt),
            lim=zoom_lim,
            line=(x0, y0, x1, y1),
            zoom=zoom_rate,
            loc=loc,
            loc1=loc1,
            loc2=loc2,
            cmap="hot",
            vmin=0,
            vmax=50,
            c="y",
        )
    if save_fig:
        plt.savefig(
            os.path.join(savedir, f"gt_{name}_{zoom_lim}_{today}.svg"),
            transparent=False,
        )

    if model_number == 3:
        # ax3 = plt.subplot(gs[3])
        plt.figure(figsize=(FigSize, FigSize))
        plt.axis("off")
        # scalebar()
        scalebar(scale_factor=1000 / pixel_size, down_scale=0.5)
        plt.imshow(Merged_3, origin="lower")
        plt.plot([x0, x1], [y0, y1], "y-", linewidth=5, alpha=0.3)
        plt.arrow(x0, y0, (x1 - x0), (y1 - y0), fc="y", ec="y", head_width=4)
        if add_title:
            plt.title("d. {}".format(title_d), fontsize=fontsize)
        if show_zoom:
            # zoom(Merged_d, zoom_lim, zoom_rate,loc=loc, loc1=loc1, loc2=loc2)
            zoom_line(
                Merged_3,
                lim=zoom_lim,
                line=(x0, y0, x1, y1),
                zoom=zoom_rate,
                loc=loc,
                loc1=loc1,
                loc2=loc2,
                cmap="hot",
                vmin=0,
                vmax=1,
                c="y",
            )
        if save_fig:
            plt.savefig(
                os.path.join(savedir, f"reco2_{name}_{zoom_lim}_{today}.svg"),
                transparent=False,
            )


def outputs_shareloc(
    DirBase_1,
    DirBase_2,
    DirBase_3=None,
    key="",
    crop_cord=(0, 0),
    zoom_cord=(0, 0),
    line=(215, 200, 255, 240),
    fact=10,
    zoomsize=50,
    chopsize=2560 // 2,
    FrameNb=300,
    name_ind=0,
    add_title=True,
    save_fig=False,
    show_zoom=True,
    norm=True,
    align_images=False,
    savedir="fig2",
    title_c="Merged,trained on IMOD",
    title_d="Merged,trained on 3Labs",
):
    """
    Compute the SSIM and plot the images


    Args:

        DirBase_1: directory of the first model
        DirBase_2: directory of the second model
        DirBase_3: directory of the third model
        key: key word to search the file
        crop_cord: crop coordinates
        zoom_cord: zoom coordinates
        line: line to zoom
        fact: factor to adjust the zoomed image
        zoomsize: zoom size
        chopsize: crop size
        FrameNb: number of frames
        name_ind: index of the name
        add_title: whether to add title
        save_fig: whether to save the figure
        show_zoom: whether to show zoomed image
        norm: whether to normalize the image
        align_images: whether to align the images
        savedir: directory to save the figure
        title_c: title of the first model
        title_d: title of the second model

    """
    DIR4Workspace = os.listdir(DirBase_1)
    CellName = []
    for name in DIR4Workspace:
        if key in name:
            CellName.append(name.split(".csv")[0])
    CellName = np.unique(CellName)
    name = CellName[name_ind]
    print(name)
    List4Image_d = []
    List4Image_d = [n for n in DIR4Workspace if "{}.".format(name) in n]

    FrameNbInit = List4Image_d[0].split("(")[1]
    FrameNbInit = FrameNbInit.split(",")[0]
    FrameNbInit = int(FrameNbInit)
    List4CertainRange_d = []
    List4CertainRange_d = [
        n
        for n in List4Image_d
        if "({}, {})".format(FrameNbInit, FrameNbInit + FrameNb) in n
    ]

    NameOfInput = [n for n in List4CertainRange_d if "real_A" in n]
    NameOfGt = [n for n in List4CertainRange_d if "_real_B_b0_" in n]
    NameOfWF = [n for n in List4CertainRange_d if "lr_inputs_b0" in n]
    NameOfReco = [n for n in List4CertainRange_d if "reco" in n]
    NameOfErroMap = [n for n in List4CertainRange_d if "squirrel_error_map" in n]

    histin = np.array(
        Image.open(
            os.path.join(DirBase_1, "".join([str(elem) for elem in NameOfInput]))
        )
    )
    wf = np.array(
        Image.open(os.path.join(DirBase_1, "".join([str(elem) for elem in NameOfWF])))
    )
    gt = np.array(
        Image.open(os.path.join(DirBase_1, "".join([str(elem) for elem in NameOfGt])))
    )
    Reco_1 = np.array(
        Image.open(os.path.join(DirBase_1, "".join([str(elem) for elem in NameOfReco])))
    )
    Reco_2 = np.array(
        Image.open(os.path.join(DirBase_2, "".join([str(elem) for elem in NameOfReco])))
    )
    if not DirBase_3 is None:
        Reco_3 = np.array(
            Image.open(
                os.path.join(DirBase_3, "".join([str(elem) for elem in NameOfReco]))
            )
        )
        ErroMap_3 = np.array(
            Image.open(
                os.path.join(DirBase_3, "".join([str(elem) for elem in NameOfErroMap]))
            )
        )
        Reco_3 = Reco_3[crop_y0 : crop_y0 + chopsize, crop_x0 : crop_x0 + chopsize]

    ErroMap_1 = np.array(
        Image.open(
            os.path.join(DirBase_1, "".join([str(elem) for elem in NameOfErroMap]))
        )
    )
    ErroMap_2 = np.array(
        Image.open(
            os.path.join(DirBase_2, "".join([str(elem) for elem in NameOfErroMap]))
        )
    )

    crop_x0, crop_y0 = crop_cord
    wf = wf[
        (crop_y0 // 4) : ((crop_y0 + chopsize) // 4),
        (crop_x0 // 4) : ((crop_x0 + chopsize) // 4),
    ]
    histin = histin[crop_y0 : crop_y0 + chopsize, crop_x0 : crop_x0 + chopsize]
    gt = gt[crop_y0 : crop_y0 + chopsize, crop_x0 : crop_x0 + chopsize]

    Reco_1 = Reco_1[crop_y0 : crop_y0 + chopsize, crop_x0 : crop_x0 + chopsize]
    Reco_2 = Reco_2[crop_y0 : crop_y0 + chopsize, crop_x0 : crop_x0 + chopsize]

    if align_images:
        Reco_1 = align_image(Reco_1, gt)
        Reco_2 = align_image(Reco_2, gt)
        if not DirBase_3 is None:
            Reco_3 = align_image(Reco_3, gt)

    print(
        "input localization number:",
        histin.sum(),
        "total localization numbver:",
        gt.sum() * 255,
    )

    if norm:
        Merged_1 = np.clip(
            np.stack(
                [(Reco_1 / Reco_1.sum()), (gt / (gt.sum())), (Reco_1 / Reco_1.sum())],
                axis=2,
            )
            * (gt.sum())
            * fact,
            0,
            1,
        )
        Merged_2 = np.clip(
            np.stack(
                [(Reco_2 / Reco_2.sum()), (gt / (gt.sum())), (Reco_2 / Reco_2.sum())],
                axis=2,
            )
            * (gt.sum())
            * fact,
            0,
            1,
        )
        if not DirBase_3 is None:
            Merged_3 = np.clip(
                np.stack(
                    [
                        (Reco_3 / Reco_3.sum()),
                        (gt / (gt.sum())),
                        (Reco_3 / Reco_3.sum()),
                    ],
                    axis=2,
                )
                * (gt.sum())
                * fact,
                0,
                1,
            )
    else:
        Merged_1 = np.clip(np.stack([Reco_1, gt, Reco_1], axis=2) * fact, 0, 1)
        Merged_2 = np.clip(np.stack([Reco_2, gt, Reco_2], axis=2) * fact, 0, 1)
        if not DirBase_3 is None:
            Merged_3 = np.clip(np.stack([Reco_3, gt, Reco_3], axis=2) * fact, 0, 1)

    Merged_1 = Merged_1[crop_y0 : crop_y0 + chopsize, crop_x0 : crop_x0 + chopsize]
    Merged_2 = Merged_2[crop_y0 : crop_y0 + chopsize, crop_x0 : crop_x0 + chopsize]
    if not DirBase_3 is None:
        Merged_3 = Merged_3[crop_y0 : crop_y0 + chopsize, crop_x0 : crop_x0 + chopsize]

    print("LocNb_in", (np.asarray(histin).sum()))
    ssim_raw_in = MultiScaleSSIM((histin), bytescale(gt), max_val=histin.max())
    ssim_raw_1 = MultiScaleSSIM(bytescale(Reco_1), bytescale(gt), max_val=255)
    ssim_raw_2 = MultiScaleSSIM(bytescale(Reco_2), bytescale(gt), max_val=255)
    if not DirBase_3 is None:
        ssim_raw_3 = MultiScaleSSIM(bytescale(Reco_3), bytescale(gt), max_val=255)

    print(
        "ssim_in : ",
        ssim_raw_in,
        "ssim_raw_1 : ",
        ssim_raw_1,
        "ssim_raw_2 : ",
        ssim_raw_2,
    )
    if not DirBase_3 is None:
        print("ssim_raw_3 : ", ssim_raw_3)

    zoom_x0, zoom_y0 = zoom_cord
    if not DirBase_3 is None:
        data = wf, histin, gt, Merged_1, Merged_2, Merged_3, Reco_1, Reco_2, Reco_3
    else:
        data = wf, histin, gt, Merged_1, Merged_2, Reco_1, Reco_2
    if DirBase_3 is None:
        model_number = 2
    else:
        model_number = 3
    plot_image1(
        data=data,
        zoom_lim=(zoom_x0, zoom_y0, zoom_x0 + zoomsize, zoom_y0 + zoomsize),
        line=line,
        model_number=model_number,
        show_zoom=show_zoom,
        savedir=savedir,
        title_c=title_c,
        title_d=title_d,
        add_title=add_title,
        save_fig=save_fig,
        name=name,
    )

    plt.figure(figsize=(10, 10))
    plt.axis("off")
    plt.imshow(
        gt,
        interpolation="nearest",
        origin="lower",
        cmap="hot",
        vmax=np.array(gt).max() * 0.2,
    )
    plt.title("Ground Truth : {}".format(name), fontsize=50)
    zoom(
        gt,
        lim=(zoom_x0, zoom_y0, zoom_x0 + zoomsize, zoom_y0 + zoomsize),
        zoom=4,
        loc=1,
        loc1=4,
        loc2=2,
        cmap="hot",
        vmax=np.array(gt).max() * 0.2,
    )

    return data


def draw_profile(data, line, save_fig, savedir="./outputs", title=""):
    """
    Draw the profile line
    Args:
        data: tuple of images
        line: line to draw
        save_fig: whether to save the figure
        savedir: directory to save the figure
        title: title of the figure
    """
    wf, histin, gt, Merged_1, Merged_2, Merged_3, Reco_1, Reco_2, Reco_3 = data
    fig, ax = plt.subplots(figsize=(20, 4))
    plt.subplot(121)
    zoomsize = 50
    x0, y0, x1, y1 = line

    profile_gt = profile_line(gt, (y0, x0), (y1, x1), linewidth=5)
    profile_1 = profile_line(
        (Reco_1 / Reco_1.sum()) * gt.sum(), (y0, x0), (y1, x1), linewidth=5
    )
    profile_2 = profile_line(
        (Reco_2 / Reco_2.sum()) * gt.sum(), (y0, x0), (y1, x1), linewidth=5
    )
    profile_3 = profile_line(
        (Reco_3 / Reco_3.sum()) * gt.sum(), (y0, x0), (y1, x1), linewidth=5
    )

    matplotlib.rc("axes", edgecolor="k", linewidth=4)
    lw = 4
    plt.plot(profile_gt, color="green", lw=lw)
    plt.plot(profile_1, color="m", ls="--", lw=lw)
    plt.plot(profile_2, color="m", ls="-.", lw=lw)
    plt.plot(profile_3, color="m", lw=lw)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_visible(True)

    plt.axis("on")
    from datetime import date

    today = str(date.today()).replace("-", "_")
    if save_fig:
        plt.savefig(
            os.path.join(
                savedir, f"profilline_{x0}_{y0}_{x1}_{y1}_{today}_{title}.svg"
            ),
            transparent=False,
        )
