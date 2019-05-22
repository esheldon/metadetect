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

    def _shearpos(row, col):
        # apply WCS to get to world coords
        v, u = jac.get_vu(row=row, col=col)

        # unshear (subtract and then add the canonical center in u, v)
        u = np.atleast_1d(u) - u_cen
        v = np.atleast_1d(v) - v_cen
        out = np.dot(a, np.vstack([u, v]))
        assert out.shape[1] == u.shape[0]
        u_sheared = out[0] + u_cen
        v_sheared = out[1] + v_cen

        # undo the WCS to get to image coords
        row_sheared, col_sheared = jac.get_rowcol(v=v_sheared, u=u_sheared)
        return row_sheared[0], col_sheared[0]

    rows_sheared = np.zeros(rows.size)
    cols_sheared = np.zeros(rows.size)

    for i in range(rows.size):
        rows_sheared[i], cols_sheared[i] = _shearpos(rows[i], cols[i])

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

    def _unshearpos(row, col):
        # apply WCS to get to world coords
        v, u = jac.get_vu(row=row, col=col)

        # unshear (subtract and then add the canonical center in u, v)
        u = np.atleast_1d(u) - u_cen
        v = np.atleast_1d(v) - v_cen
        out = np.dot(ainv, np.vstack([u, v]))
        assert out.shape[1] == u.shape[0]
        u_unsheared = out[0] + u_cen
        v_unsheared = out[1] + v_cen

        # undo the WCS to get to image coords
        row_unsheared, col_unsheared = jac.get_rowcol(v=v_unsheared, u=u_unsheared)

        return row_unsheared[0], col_unsheared[0]

    rows_unsheared = np.zeros(rows.size)
    cols_unsheared = np.zeros(rows.size)

    for i in range(rows.size):
        rows_unsheared[i], cols_unsheared[i] = _unshearpos(rows[i], cols[i])

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
