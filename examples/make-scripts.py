import os
import numpy as np
import argparse

template = r"""
command: |
    export OMP_NUM_THREADS=1
    source activate mystack
    cd /astro/u/esheldon/lensing/test-lsst-mdet
    output=%(output)s

    python detect_simple_full.py \
        --nepochs %(nepoch)d \
        --noise 1 \
        --ntrial %(ntrial)d \
        --seed %(seed)d \
        --output $output

job_name: "%(job_name)s"
"""


def get_run_dir(run):
    return '/astro/u/esheldon/lensing/test-lsst-mdet/runs/%s' % run


def get_script_file(run, num):
    outdir = get_run_dir(run)
    return os.path.join(
        outdir,
        '%s-%07d.yaml' % (run, num)
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
    parser.add_argument('--nepoch', type=int, default=1)
    parser.add_argument('--ntrial', type=int, default=10)
    return parser.parse_args()


def main():
    args = get_args()
    rng = np.random.RandomState(args.seed)

    run_dir = get_run_dir(args.run)
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)

    for i in range(args.njobs):

        while True:
            seed = rng.randint(0, 2**19)
            script_fname = get_script_file(args.run, seed)
            if not os.path.exists(script_fname):
                break

        output = get_output(args.run, seed)
        job_name = '%s-%06d' % (args.run, seed)

        text = template % {
            'seed': seed,
            'ntrial': args.ntrial,
            'nepoch': args.nepoch,
            'output': output,
            'job_name': job_name,
        }
        # print(text)

        print(script_fname)
        with open(script_fname, 'w') as fobj:
            fobj.write(text)


main()
