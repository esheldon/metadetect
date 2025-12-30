import numpy as np
import galsim
import ngmix
from ngmix.metacal.metacal import _get_gauss_target_psf, _get_ellip_dilation
from tqdm import tqdm


SCALE = 0.2


def fit_gauss(im, noise, rng):
    fitter = ngmix.admom.AdmomFitter(rng=rng)
    guesser = ngmix.guessers.GMixPSFGuesser(
        rng=rng,
        ngauss=1,
        guess_from_moms=True,
    )
    runner = ngmix.runners.PSFRunner(fitter=fitter, guesser=guesser, ntry=4)

    cen = (np.array(im.shape) - 1.0) / 2.0
    jac = ngmix.DiagonalJacobian(
        row=cen[0],
        col=cen[1],
        scale=SCALE,
    )
    obs = ngmix.Observation(
        image=im,
        weight=im * 0 + 1.0 / noise ** 2,
        jacobian=jac,
    )
    res = runner.go(obs)
    return res


def get_psf(e1, e2, dim=49, scale=SCALE):
    psf = galsim.Moffat(
        fwhm=0.8, beta=2.5,
    # psf = galsim.Gaussian(
    #     fwhm=0.8,
    ).shear(
        e1=e1,
        e2=e2,
    )
    gsim = psf.drawImage(
        nx=dim,
        ny=dim,
        dtype=np.float64,
        scale=scale,
    )

    return psf, gsim


def get_e1e2(rng):
    while True:
        e1, e2 = rng.normal(scale=0.1 / np.sqrt(2), size=2)
        e = np.sqrt(e1 ** 2 + e2 ** 2)
        if e < 0.9:
            break
    return e1, e2


def get_Tvals(rng, noise, ntrial):
    s2n_vals = np.zeros(ntrial)
    Tvals = np.zeros(ntrial)
    Tvals_fitgauss = np.zeros(ntrial)

    for i in range(ntrial):

        e1, e2 = get_e1e2(rng)

        psf, gsim = get_psf(
            e1=e1,
            e2=e2,
        )
        gsim.array[:, :] += rng.normal(
            scale=noise,
            size=gsim.array.shape,
        )

        gsim_int = galsim.InterpolatedImage(
            gsim, x_interpolant='lanczos15',
        )
        gauss_target_psf = _get_gauss_target_psf(
            gsim_int,
            flux=1.0,
        )
        Tvals[i] = 2 * gauss_target_psf.sigma ** 2

        res = fit_gauss(
            im=gsim.array,
            noise=noise,
            rng=rng,
        )
        dilation = _get_ellip_dilation(
            e1=res['e'][0], e2=res['e'][1], T=res['T'],
        )
        T_dilated = res['T'] * dilation

        Tvals_fitgauss[i] = T_dilated
        s2n_vals[i] = res['s2n']

    return Tvals, Tvals_fitgauss, s2n_vals


def plot_fwhms(
    s2n,
    fwhms,
    fwhm_errs,
    fwhms_fitgauss,
    fwhm_fitgauss_errs,
):
    from matplotlib import pyplot as mplt

    fig, ax = mplt.subplots(figsize=(10, 6))
    ax.set(
        xlabel='S/N',
        ylabel='fwhm [arcsec]',
        xlim=(0.5 * s2n.min(), 1.5 * s2n.max()),
    )
    ax.set_xscale('log')

    ax.errorbar(
        s2n,
        fwhms,
        fwhm_errs,
        label='gauss',
    )
    ax.errorbar(
        s2n,
        fwhms_fitgauss,
        fwhm_fitgauss_errs,
        label='fitgauss',
    )
    ax.text(75, 0.92, "Moffat PSF")

    ax.legend()
    output = 'fwhm-vs-s2n.pdf'
    print('writing:', output)
    fig.savefig(output)


def main():
    ntrial = 100
    rng = np.random.RandomState()

    noises = np.logspace(
        # np.log10(0.0001),
        np.log10(0.00001),
        np.log10(0.001),
        10,
    )

    s2ns = np.zeros(noises.size)
    fwhms = np.zeros(noises.size)
    fwhm_errs = np.zeros(noises.size)
    fwhms_fitgauss = np.zeros(noises.size)
    fwhm_fitgauss_errs = np.zeros(noises.size)

    for i, noise in enumerate(tqdm(noises, ascii=True, ncols=70)):
        Tvals, Tvals_fitgauss, s2n_vals = get_Tvals(
            rng=rng,
            noise=noise,
            ntrial=ntrial,
        )

        s2ns[i] = s2n_vals.mean()

        fwhm_vals = ngmix.moments.T_to_fwhm(Tvals)
        fwhm_fitgauss_vals = ngmix.moments.T_to_fwhm(Tvals_fitgauss)

        fwhms[i] = fwhm_vals.mean()
        fwhm_errs[i] = fwhm_vals.std() / np.sqrt(fwhm_vals.size)

        fwhms_fitgauss[i] = fwhm_fitgauss_vals.mean()
        fwhm_fitgauss_errs[i] = (
            fwhm_fitgauss_vals.std() / np.sqrt(fwhm_fitgauss_vals.size)
        )

    plot_fwhms(
        s2n=s2ns,
        fwhms=fwhms,
        fwhm_errs=fwhm_errs,
        fwhms_fitgauss=fwhms_fitgauss,
        fwhm_fitgauss_errs=fwhm_fitgauss_errs,
    )


main()
