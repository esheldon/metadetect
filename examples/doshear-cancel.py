import numpy as np
import argparse
import fitsio


def get_summed(data):
    sdata = data[0].copy()

    for n in sdata.dtype.names:
        sdata[n] = data[n].sum(axis=0)

    return sdata


def sub1(data, subdata):
    odata = data.copy()

    for n in data.dtype.names:
        odata[n] -= subdata[n]

    return odata


def get_m1(data):
    g1_1p = data['g_ns_sum_1p'][0]/data['n_ns_1p']
    g1_1p_1p = data['g_1p_sum_1p'][0]/data['n_1p_1p']
    g1_1m_1p = data['g_1m_sum_1p'][0]/data['n_1m_1p']

    R11_1p = (g1_1p_1p - g1_1m_1p)/0.02  # noqa

    s1_1p = g1_1p/R11_1p

    g1_1m = data['g_ns_sum_1m'][0]/data['n_ns_1m']
    g1_1p_1m = data['g_1p_sum_1m'][0]/data['n_1p_1m']
    g1_1m_1m = data['g_1m_sum_1m'][0]/data['n_1m_1m']

    R11_1m = (g1_1p_1m - g1_1m_1m)/0.02  # noqa

    s1_1m = g1_1m/R11_1m
    print(s1_1p, s1_1m)

    m1 = (s1_1p - s1_1m)/0.04 - 1
    return m1


def jackknife_m(data):
    sdata = get_summed(data)
    m1 = get_m1(sdata)

    m1vals = np.zeros(data.size)

    for i in range(m1vals.size):
        subdata = sub1(sdata, data[i])
        tm1 = get_m1(subdata)
        m1vals[i] = tm1

    nchunks = m1vals.size
    fac = (nchunks-1)/float(nchunks)
    m1cov = fac*((m1 - m1vals)**2).sum()
    m1err = np.sqrt(m1cov)
    return m1, m1err


def get_sums(data, stype):
    w, = np.where(
        (data['flags'] == 0) &
        (data['shear_type'] == stype)
    )
    g_sum = data['wmom_g'][w].sum(axis=0)
    return g_sum, w.size


def read_flist(args):
    nf = len(args.flist)

    dt = [
        ('g_ns_sum_1p', ('f8', 2)),
        ('g_1p_sum_1p', ('f8', 2)),
        ('g_1m_sum_1p', ('f8', 2)),

        ('n_ns_1p', 'i8'),
        ('n_1p_1p', 'i8'),
        ('n_1m_1p', 'i8'),

        ('g_ns_sum_1m', ('f8', 2)),
        ('g_1p_sum_1m', ('f8', 2)),
        ('g_1m_sum_1m', ('f8', 2)),

        ('n_ns_1m', 'i8'),
        ('n_1p_1m', 'i8'),
        ('n_1m_1m', 'i8'),
    ]

    data = np.zeros(nf, dtype=dt)

    for i, f in enumerate(args.flist):
        print('%d/%d %s' % (i+1, nf, f))

        d = data[i]

        data_1p = fitsio.read(f, ext='1p')
        data_1m = fitsio.read(f, ext='1m')

        # from 1p ext
        tg_sum, tn = get_sums(data_1p, 'noshear')

        tg_sum_1p, tn_1p = get_sums(data_1p, '1p')
        tg_sum_1m, tn_1m = get_sums(data_1p, '1m')

        d['g_ns_sum_1p'] = tg_sum
        d['g_1p_sum_1p'] = tg_sum_1p
        d['g_1m_sum_1p'] = tg_sum_1m

        d['n_ns_1p'] = tn
        d['n_1p_1p'] = tn_1p
        d['n_1m_1p'] = tn_1m

        # from 1m ext
        tg_sum, tn = get_sums(data_1m, 'noshear')

        tg_sum_1p, tn_1p = get_sums(data_1m, '1p')
        tg_sum_1m, tn_1m = get_sums(data_1m, '1m')

        d['g_ns_sum_1m'] = tg_sum
        d['g_1p_sum_1m'] = tg_sum_1p
        d['g_1m_sum_1m'] = tg_sum_1m

        d['n_ns_1m'] = tn
        d['n_1p_1m'] = tn_1p
        d['n_1m_1m'] = tn_1m

    return data


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('flist', nargs='+')
    parser.add_argument('--nsigma', type=int, default=3)
    return parser.parse_args()


def main():
    args = get_args()
    data = read_flist(args)

    m1, m1err = jackknife_m(data)
    tup = (m1, m1err*args.nsigma, args.nsigma)
    print('m1: %g +/- %g (%d sigma)' % tup)


main()
