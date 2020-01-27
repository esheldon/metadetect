import os
import numpy as np
import argparse

CONDOR_SUBMIT_HEAD = """
Universe        = vanilla

Notification    = Never

# Run this exe with these args
Executable = /astro/u/esheldon/lensing/test-lsst-mdet/run_lsst_sim.sh

Image_Size       =  1000000

GetEnv = True

kill_sig        = SIGINT

+Experiment     = "astro"
"""

CONDOR_JOB_TEMPLATE = """
+job_name = "%(job_name)s"
Arguments = %(nepoch)d %(ntrial)d %(seed)d %(output)s
Queue
"""


def get_run_dir(run):
    return '/astro/u/esheldon/lensing/test-lsst-mdet/runs/%s' % run


def get_script_file(run, seed):
    outdir = get_run_dir(run)
    return os.path.join(
        outdir,
        '%s-%07d.condor' % (run, seed)
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

    script_fname = get_script_file(args.run, args.seed)
    print(script_fname)
    if os.path.exists(script_fname):
        raise RuntimeError('script already exists: %s' % script_fname)

    with open(script_fname, 'w') as fobj:
        fobj.write(CONDOR_SUBMIT_HEAD)

        for i in range(args.njobs):

            seed = rng.randint(0, 2**19)

            output = get_output(args.run, seed)
            job_name = '%s-%07d' % (args.run, seed)

            job_text = CONDOR_JOB_TEMPLATE % {
                'seed': seed,
                'ntrial': args.ntrial,
                'nepoch': args.nepoch,
                'output': output,
                'job_name': job_name,
            }
            # print(text)

            print('%d/%d  %s' % (i+1, args.njobs, job_name))
            fobj.write(job_text)

    print(script_fname)


main()
