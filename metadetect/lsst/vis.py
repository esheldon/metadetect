import numpy as np
from matplotlib import pyplot as mplt
from .util import copy_mbexp

DEFAULT_STRETCH = 1.25
DEFAULT_Q = 7.5
SIZE = 16
COLOR = 'red'
EDGECOLOR = 'white'


def show_exp(exp, mess=None, use_mpl=False, sources=None):
    """
    show the image in ds9

    Parameters
    ----------
    exp: Exposure
        The image to show
    mess: str, optional
        A message to use as title to plot
    """
    import matplotlib.pyplot as mplt
    import lsst.afw.display as afw_display

    if use_mpl:
        fig, axs = mplt.subplots(ncols=3)
        image = exp.image.array

        noise = np.sqrt(np.median(exp.variance.array))

        noise = np.sqrt(np.median(exp.variance.array))
        minval = 0.1 * noise
        axs[0].imshow(np.log(image.clip(min=minval)))

        axs[1].imshow(exp.variance.array)
        axs[1].set_title('variance')

        axs[2].imshow(exp.mask.array)
        axs[2].set_title('mask')

        if sources is not None:
            from pprint import pprint
            bbox = exp.getBBox()
            pprint(exp.mask.getMaskPlaneDict())
            axs[0].scatter(
                sources['base_SdssCentroid_x'] - bbox.beginX,
                sources['base_SdssCentroid_y'] - bbox.beginY,
                color='red',
            )

        if mess is not None:
            fig.suptitle(mess)

        mplt.show()

    else:

        display = afw_display.getDisplay(backend='ds9')
        display.mtv(exp)
        display.scale('log', 'minmax')

        prompt = 'hit enter'
        if mess is not None:
            prompt = '%s: (%s)' % (mess, prompt)

        input(prompt)


def show_mbexp(
    mbexp, stretch=DEFAULT_STRETCH, q=DEFAULT_Q, mess=None, ax=None,
    sources=None,
    show=True,
):
    """
    visialize a MultibandExposure

    Parameters
    ----------
    mbexp: lsst.afw.image.MultibandExposure
        MutibandExposure to visualize.  Currently must be at least
        three bands.
    stretch: float, optional
        The stretch parameter for
        astropy.visualization.lupton_rgb. import AsinhMapping
    q: float, optional
        The Q parameter for
        astropy.visualization.lupton_rgb. import AsinhMapping
    mess: str, optional
        A message to use as title to plot
    ax: matplotlib plot axis
        If sent use this axis for plot rather than creating a
        new one
    show: bool, optional
        If set to True, show on the screen.

    Returns
    -------
    The axis used for plotting
    """

    image = mbexp.image.array

    if ax is None:
        fig, ax = mplt.subplots()

    if image.shape[0] >= 3:
        import scarlet
        from astropy.visualization.lupton_rgb import AsinhMapping

        timage = image[:3, :, :].clip(min=0)

        asinh = AsinhMapping(
            minimum=0,
            stretch=stretch,
            Q=q,
        )

        img_rgb = scarlet.display.img_to_rgb(timage, norm=asinh)

        ax.imshow(img_rgb)

    else:
        noise = np.sqrt(np.median(mbexp.variance.array))
        if image.shape[0] == 1:
            timage = image[0]
        else:
            timage = image.sum(axis=0)

        noise = np.sqrt(np.median(mbexp.variance.array))
        minval = 0.1 * noise
        ax.imshow(np.log(timage.clip(min=minval)))

    if sources is not None:
        x, y = _extract_xy(mbexp, sources)
        ax.scatter(
            x, y,
            s=SIZE, color=COLOR, edgecolor=EDGECOLOR,
        )

    if mess is not None:
        ax.set_title(mess)

    if show:
        mplt.show()

    return ax


def _extract_xy(mbexp, sources):
    if isinstance(sources, np.ndarray):
        y = sources['row'] - sources['row0']
        x = sources['col'] - sources['col0']
    else:
        bbox = mbexp.getBBox()
        x0 = bbox.getBeginX()
        y0 = bbox.getBeginY()
        x = []
        y = []
        for source in sources[mbexp.bands[0]]:
            peak = source.getFootprint().getPeaks()[0]
            cen = peak.getCentroid()
            x.append(cen.getX() - x0)
            y.append(cen.getY() - y0)

    return x, y


def compare_mbexp(
    mbexp, model, sources=None, stretch=DEFAULT_STRETCH, q=DEFAULT_Q,
):
    """
    compare two MultibandExposure with residuals

    Parameters
    ----------
    mbexp: lsst.afw.image.MultibandExposure
        MutibandExposure to visualize.  Currently must be at least
        three bands.
    model: lsst.afw.image.MultibandExposure
        MutibandExposure to visualize.  Currently must be at least
        three bands.
    stretch: float, optional
        The stretch parameter for
        astropy.visualization.lupton_rgb. import AsinhMapping
    q: float, optional
        The Q parameter for
        astropy.visualization.lupton_rgb. import AsinhMapping
    """

    if isinstance(model, list):
        mold = model
        model = copy_mbexp(mbexp)
        for iband, band in enumerate(model.bands):
            model[band].image.array[:, :] = mold[iband]

    fig, axs = mplt.subplots(nrows=2, ncols=2, figsize=(12, 12))
    show_mbexp(
        mbexp, stretch=stretch, q=q, mess='data', ax=axs[0, 0], show=False,
        sources=sources,
    )
    show_mbexp(
        model, stretch=stretch, q=q, mess='model', ax=axs[0, 1], show=False,
        sources=sources,
    )
    axs[1, 1].axis('off')

    c = copy_mbexp(mbexp)
    c.image.array -= model.image.array
    show_mbexp(
        c, stretch=stretch, q=q, mess='data - model', ax=axs[1, 0], show=False,
        sources=sources,
    )
    mplt.show()


def show_three_mbexp(
    mbexp_list,
    labels=None,
    stretch=DEFAULT_STRETCH, q=DEFAULT_Q, mess=None, ax=None,
):
    """
    show three different MultibandExposure

    Parameters
    ----------
    mbexp_list: [lsst.afw.image.MultibandExposure]
        List of MutibandExposure to visualize.  Currently must be at least
        three bands.
    labels: list, optional
        Listof labels for a legend
    stretch: float, optional
        The stretch parameter for
        astropy.visualization.lupton_rgb. import AsinhMapping
    q: float, optional
        The Q parameter for
        astropy.visualization.lupton_rgb. import AsinhMapping
    ax: matplotlib plot axis
        If sent use this axis for plot rather than creating a
        new one
    mess: str, optional
        A message to use as title to plot
    """

    assert len(mbexp_list) == 3
    if labels is None:
        labels = ['image 1', 'image 2', 'image 3']
    assert len(labels) == 3

    fig, axs = mplt.subplots(nrows=2, ncols=2, figsize=(12, 12))
    show_mbexp(
        mbexp_list[0], stretch=stretch, q=q, ax=axs[0, 0], show=False,
        mess=labels[0],
    )
    show_mbexp(
        mbexp_list[1], stretch=stretch, q=q, ax=axs[0, 1], show=False,
        mess=labels[1],
    )
    show_mbexp(
        mbexp_list[2], stretch=stretch, q=q, ax=axs[1, 0], show=False,
        mess=labels[2],
    )
    axs[1, 1].axis('off')

    mplt.show()


def show_mbexp_demo(mbexp):
    """
    Show a few stretches, q values. Useful for choosing the stretch
    you would like

    Parameters
    ----------
    mbexp: lsst.afw.image.MultibandExposure
        MutibandExposure to visualize.  Currently must be at least
        three bands.
    """
    from astropy.visualization.lupton_rgb import AsinhMapping
    import scarlet

    image = mbexp.image.array

    # TODO repeat bands as needed
    assert image.shape[0] >= 3, 'need at least 3 exposure for color'

    # stretches = [0.1, 0.5, 1.0]
    # qvals = [10, 20, 30]  # noqa
    stretches = [0.75, 1, 1.25]
    qvals = [5, 7.5, 10]  # noqa

    fig, axs = mplt.subplots(len(stretches), len(qvals), figsize=(12, 12))
    for ist, stretch in enumerate(stretches):
        for iq, q in enumerate(Qvals):  # noqa
            asinh = AsinhMapping(
                minimum=0,
                stretch=stretch, Q=q,
            )
            img_rgb = scarlet.display.img_to_rgb(image, norm=asinh)
            axs[ist, iq].imshow(img_rgb)
            axs[ist, iq].set_title(f'Stretch {stretch}, Q {q}')

    mplt.show()


def show_multi_mbobs(mbobs):
    """
    Show images from a ngmix.MultiBandObsList

    Parameters
    ----------
    mbobs: ngmix.MultiBandObsList
        The data
    """
    import matplotlib.pyplot as mplt

    nband = len(mbobs)

    types = ['image', 'weight', 'bmask']
    ntypes = len(types)
    fig, axs = mplt.subplots(nrows=ntypes, ncols=nband)

    for iband, obslist in enumerate(mbobs):

        obs = obslist[0]

        for itype, type in enumerate(types):

            im = getattr(obs, type)

            axs[iband, itype].imshow(im)
            axs[iband, itype].set_title(f'band {iband} {type}')

    mplt.show()


def show_multi_mbexp(mbexp, sources=None):
    """
    Show images from a ngmix.MultiBandObsList

    Parameters
    ----------
    mbexp: lsst.afw.image.MultibandExposure
        The data
    """
    import matplotlib.pyplot as mplt

    nband = len(mbexp)

    ncols = 3
    fig, axs = mplt.subplots(nrows=nband, ncols=ncols, squeeze=False)

    if sources is not None:
        x, y = _extract_xy(mbexp, sources)

    for iband, band in enumerate(mbexp.bands):
        exp = mbexp[band]

        noise = np.sqrt(np.median(exp.variance.array))
        minval = 0.1 * noise
        axs[iband, 0].imshow(np.log(exp.image.array.clip(min=minval)))
        axs[iband, 0].set_title(f'band {iband} image')

        axs[iband, 1].imshow(exp.variance.array)
        axs[iband, 1].set_title(f'band {iband} var')

        axs[iband, 2].imshow(exp.mask.array)
        axs[iband, 2].set_title(f'band {iband} mask')

        if sources is not None:
            for col in range(ncols):
                axs[iband, col].scatter(
                    x, y,
                    s=SIZE, color=COLOR, edgecolor=EDGECOLOR,
                )

    mplt.show()


def show_image_and_mask(mbexp, band=0):
    """
    Show images from a ngmix.MultiBandObsList

    Parameters
    ----------
    mbexp: lsst.afw.image.MultibandExposure
        The data
    """
    import matplotlib.pyplot as mplt

    fig, axs = mplt.subplots(ncols=2)
    axs[0].axis('off')
    axs[1].axis('off')

    bname = mbexp.bands[band]
    exp = mbexp[bname]

    noise = np.sqrt(np.median(exp.variance.array))
    minval = 0.1 * noise
    axs[0].imshow(np.log(exp.image.array.clip(min=minval)))
    axs[1].imshow(exp.mask.array)
    mplt.show()
