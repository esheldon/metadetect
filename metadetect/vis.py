from matplotlib import pyplot as mplt
from .util import copy_mbexp

DEFAULT_STRETCH = 1.25
DEFAULT_Q = 7.5


def show_exp(exp, mess=None):
    """
    show the image in ds9

    Parameters
    ----------
    exp: Exposure
        The image to show
    mess: str, optional
        A message to use as title to plot
    """
    import lsst.afw.display as afw_display

    display = afw_display.getDisplay(backend='ds9')
    display.mtv(exp)
    display.scale('log', 'minmax')

    prompt = 'hit enter'
    if mess is not None:
        prompt = '%s: (%s)' % (mess, prompt)

    input(prompt)


def show_mbexp(
    mbexp, stretch=DEFAULT_STRETCH, q=DEFAULT_Q, mess=None, ax=None,
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
    from astropy.visualization.lupton_rgb import AsinhMapping
    import scarlet

    image = mbexp.image.array

    # TODO repeat bands as needed
    assert image.shape[0] >= 3, 'need at least 3 exposure for color'

    asinh = AsinhMapping(
        # minimum=image.min(),
        minimum=0,
        stretch=stretch,
        Q=q,
    )

    img_rgb = scarlet.display.img_to_rgb(image, norm=asinh)

    if ax is None:
        fig, ax = mplt.subplots()

    if mess is not None:
        ax.set_title(mess)

    ax.imshow(img_rgb)
    if show:
        mplt.show()

    return ax


def compare_mbexp(mbexp, model, stretch=DEFAULT_STRETCH, q=DEFAULT_Q):
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
    fig, axs = mplt.subplots(nrows=2, ncols=2, figsize=(12, 12))
    show_mbexp(
        mbexp, stretch=stretch, q=q, mess='data', ax=axs[0, 0], show=False,
    )
    show_mbexp(
        model, stretch=stretch, q=q, mess='model', ax=axs[0, 1], show=False,
    )
    axs[1, 1].axis('off')

    c = copy_mbexp(mbexp)
    c.image.array -= model.image.array
    show_mbexp(
        c, stretch=stretch, q=q, mess='data - model', ax=axs[1, 0], show=False,
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
