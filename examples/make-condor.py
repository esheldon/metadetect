import os
import numpy as np
import argparse

SCRIPT_TEMPLATE = r"""#!/bin/bash

export OMP_NUM_THREADS=1

seed=$1
output=$2

logfile=${output}.log

tmpdir=$_CONDOR_SCRATCH_DIR
tmplog=${tmpdir}/logfile

python /astro/u/esheldon/lensing/test-lsst-mdet/lsst_sim.py \
    --trim-output \
    %(grid)s \
    %(dither)s \
    %(dither_range)s \
    %(rotate)s \
    %(cosmic_rays)s \
    %(bad_columns)s \
    %(vary_wcs_shear)s \
    %(vary_scale)s \
    %(nostack)s \
    %(psf_g1)s \
    %(psf_g2)s \
    %(psf_varfac)s \
    %(se_dim)s \
    %(coadd_dim)s \
    %(buff)s \
    --gal-type %(gal_type)s \
    --psf-type %(psf_type)s \
    --bands %(bands)s \
    --nepochs %(nepochs)d \
    --ntrial %(ntrial)d \
    --seed ${seed} \
    --output ${output} &> ${tmplog}

mv -fv $tmplog $logfile
"""

CONDOR_SUBMIT_HEAD = """
Universe        = vanilla

Notification    = Never

# Run this exe with these args
Executable = %(script_file)s

Image_Size       =  1000000

GetEnv = True

kill_sig        = SIGINT

+Experiment     = "astro"
"""

CONDOR_JOB_TEMPLATE = """
+job_name = "%(job_name)s"
Arguments = %(seed)d %(output)s
Queue
"""


def get_run_dir(run):
    return '/astro/u/esheldon/lensing/test-lsst-mdet/runs/%s' % run


def get_condor_script_file(run, seed):
    outdir = get_run_dir(run)
    return os.path.join(
        outdir,
        '%s-%07d.condor' % (run, seed)
    )


def get_script_file(run, seed):
    outdir = get_run_dir(run)
    return os.path.join(
        outdir,
        '%s-%07d.sh' % (run, seed)
    )


def get_output(run, num):
    outdir = get_run_dir(run)
    return os.path.join(
        outdir,
        '%s-%07d.fits' % (run, num)
    )


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--njobs', type=int, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--nepochs', type=int, default=1)
    parser.add_argument('--ntrial', type=int, default=10)

    parser.add_argument('--coadd-dim', type=int)
    parser.add_argument('--buff', type=int)
    parser.add_argument('--se-dim', type=int)

    parser.add_argument('--cosmic-rays', action='store_true')
    parser.add_argument('--bad-columns', action='store_true')
    parser.add_argument('--grid', action='store_true')

    parser.add_argument('--nostack', action='store_true')

    parser.add_argument('--dither', action='store_true')
    parser.add_argument('--dither-range', type=float)
    parser.add_argument('--rotate', action='store_true')
    parser.add_argument('--vary-scale', action='store_true')
    parser.add_argument('--vary-wcs-shear', action='store_true')

    parser.add_argument('--bands', default='r,i,z')

    parser.add_argument('--psf-type', default='gauss')
    parser.add_argument('--psf-g1', type=float)
    parser.add_argument('--psf-g2', type=float)
    parser.add_argument('--psf-varfac', type=float)

    parser.add_argument('--gal-type', default='exp')

    args = parser.parse_args()

    return args


def make_script_text(args):
    assert args.gal_type in ['exp', 'wldeblend']

    if args.cosmic_rays:
        cosmic_rays = '--cosmic-rays'
    else:
        cosmic_rays = ''

    if args.bad_columns:
        bad_columns = '--bad-columns'
    else:
        bad_columns = ''

    if args.grid:
        grid = '--grid'
    else:
        grid = ''

    if args.dither:
        dither = '--dither'
    else:
        dither = ''

    if args.dither_range is not None:
        dither_range = '--dither-range %g' % args.dither_range
    else:
        dither_range = ''

    if args.se_dim is not None:
        se_dim = '--se-dim %d' % args.se_dim
    else:
        se_dim = ''

    if args.rotate:
        rotate = '--rotate'
    else:
        rotate = ''

    if args.nostack:
        nostack = '--nostack'
    else:
        nostack = ''

    if args.vary_wcs_shear:
        vary_wcs_shear = '--vary-wcs-shear'
    else:
        vary_wcs_shear = ''

    if args.vary_scale:
        vary_scale = '--vary-scale'
    else:
        vary_scale = ''

    if args.psf_g1 is not None:
        psf_g1 = '--psf-g1 %g' % args.psf_g1
    else:
        psf_g1 = ''

    if args.psf_g2 is not None:
        psf_g2 = '--psf-g2 %g' % args.psf_g2
    else:
        psf_g2 = ''

    if args.psf_varfac is not None:
        psf_varfac = '--psf-varfac %g' % args.psf_varfac
    else:
        psf_varfac = ''

    if args.coadd_dim is not None:
        coadd_dim = '--coadd-dim %d' % args.coadd_dim
    else:
        coadd_dim = ''

    if args.buff is not None:
        buff = '--buff%d' % args.buff
    else:
        buff = ''

    return SCRIPT_TEMPLATE % {
        'grid': grid,
        'dither': dither,
        'dither_range': dither_range,
        'rotate': rotate,
        'cosmic_rays': cosmic_rays,
        'bad_columns': bad_columns,
        'vary_wcs_shear': vary_wcs_shear,
        'vary_scale': vary_scale,
        'nostack': nostack,
        'bands': args.bands,
        'coadd_dim': coadd_dim,
        'buff': buff,
        'se_dim': se_dim,
        'nepochs': args.nepochs,
        'ntrial': args.ntrial,
        'gal_type': args.gal_type,

        'psf_type': args.psf_type,
        'psf_g1': psf_g1,
        'psf_g2': psf_g2,
        'psf_varfac': psf_varfac,
    }


def main():
    args = get_args()

    rng = np.random.RandomState(args.seed)

    run_dir = get_run_dir(args.run)
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)

    script_file = get_script_file(args.run, args.seed)
    condor_file = get_condor_script_file(args.run, args.seed)

    if os.path.exists(condor_file):
        raise RuntimeError('script already exists: %s' % condor_file)

    with open(script_file, 'w') as fobj:
        script = make_script_text(args)
        fobj.write(script)

    os.system('chmod 755 '+script_file)

    with open(condor_file, 'w') as fobj:
        fobj.write(CONDOR_SUBMIT_HEAD % {'script_file': script_file})

        for i in range(args.njobs):

            seed = rng.randint(0, 2**19)
            output = get_output(args.run, seed)
            job_name = '%s-%07d' % (args.run, seed)

            job_text = CONDOR_JOB_TEMPLATE % {
                'job_name': job_name,
                'seed': seed,
                'output': output,
            }

            # print('%d/%d  %s' % (i+1, args.njobs, job_name))
            fobj.write(job_text)

    print(script_file)
    print(condor_file)
    print('wrote:', args.njobs)


main()
