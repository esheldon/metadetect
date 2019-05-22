import numpy as np
import galsim


def shear_positions(rows, cols, shear_str, obs, step=0.01):
    """
    unshear the input row and column positions

    Parameters
    ----------
    rows: array
        array of row values in sheared coordinates
    cols: array
        array of col values in sheared coordinates
    shear_str: string
        'noshear', '1p', '1m', '2p', '2m'
    jac: ngmix.Jacobian
        Describes the wcs
    step: float
        shear step for metacal, default 0.01

    Returns
    -------
    rows_sheared, cols_sheared:
        rows and cols in the sheared coordinates
    """

    if shear_str == 'noshear':
        return rows, cols

    shear = get_galsim_shear(shear_str, step)

    # this is the matrix that does shearing in u, v
    a = shear.getMatrix()

    # this is the jacobian that deals with the WCS
    jac = obs.jacobian

    # we need the canonical image center in (u, v) for undoing the
    # shearing

    dims = obs.image.shape
    row_cen = (dims[0] - 1) / 2
    col_cen = (dims[1] - 1) / 2

    v_cen, u_cen = jac.get_vu(row=row_cen, col=col_cen)

    # apply WCS to get to world coords
    v, u = jac.get_vu(row=rows, col=cols)

    # unshear (subtract and then add the canonical center in u, v)
    u = np.atleast_1d(u) - u_cen
    v = np.atleast_1d(v) - v_cen

    pos = np.vstack((u, v))

    out = np.dot(a, pos)
    assert out.shape[1] == u.shape[0]
    u_sheared = out[0, :] + u_cen
    v_sheared = out[1, :] + v_cen

    # undo the WCS to get to image coords
    rows_sheared, cols_sheared = jac.get_rowcol(v=v_sheared, u=u_sheared)

    return rows_sheared, cols_sheared


def unshear_positions(rows, cols, shear_str, obs, step=0.01):
    """
    unshear the input row and column positions

    Parameters
    ----------
    rows: array
        array of row values in sheared coordinates
    cols: array
        array of col values in sheared coordinates
    shear_str: string
        'noshear', '1p', '1m', '2p', '2m'
    jac: ngmix.Jacobian
        Describes the wcs
    step: float
        shear step for metacal, default 0.01

    Returns
    -------
    rows_unsheared, cols_unsheared:
        rows and cols in the unsheared coordinates
    """

    if shear_str == 'noshear':
        return rows, cols

    shear = get_galsim_shear(shear_str, step)

    # this is the matrix that undoes shearing in u, v
    ainv = np.linalg.inv(shear.getMatrix())

    # this is the jacobian that deals with the WCS
    jac = obs.jacobian

    # we need the canonical image center in (u, v) for undoing the
    # shearing

    dims = obs.image.shape
    row_cen = (dims[0] - 1) / 2
    col_cen = (dims[1] - 1) / 2

    v_cen, u_cen = jac.get_vu(row=row_cen, col=col_cen)

    # apply WCS to get to world coords
    v, u = jac.get_vu(row=rows, col=cols)

    # unshear (subtract and then add the canonical center in u, v)
    u = np.atleast_1d(u) - u_cen
    v = np.atleast_1d(v) - v_cen

    pos = np.vstack((u, v))

    out = np.dot(ainv, pos)
    assert out.shape[1] == u.shape[0]
    u_unsheared = out[0] + u_cen
    v_unsheared = out[1] + v_cen

    # undo the WCS to get to image coords
    rows_unsheared, cols_unsheared = \
        jac.get_rowcol(v=v_unsheared, u=u_unsheared)

    return rows_unsheared, cols_unsheared


def get_galsim_shear(shear_str, step):
    """
    convert shear string to galsim Shear object
    """

    if shear_str == '1p':
        shear = galsim.Shear(g1=+step, g2=0.00)
    elif shear_str == '1m':
        shear = galsim.Shear(g1=-step, g2=0.00)
    elif shear_str == '2p':
        shear = galsim.Shear(g1=0.00, g2=+step)
    elif shear_str == '2m':
        shear = galsim.Shear(g1=0.00, g2=-step)
    else:
        raise ValueError('can only convert 1p,1m,2p,2m to galsim Shear')

    return shear
