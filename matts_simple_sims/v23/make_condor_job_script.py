#!/usr/bin/env python
import os

PREAMBLE = """\
#
# always keep these at the defaults
#
Universe        = vanilla
kill_sig        = SIGINT
+Experiment     = "astro"
# copy env. variables to the job
GetEnv          = True

#
# options below you can change safely
#

# don't send email
Notification    = Never

# Run this executable.  executable bits must be set
Executable      = {script_name}

# A guess at the memory usage, including virtual memory
Image_Size       =  1000000

# this restricts the jobs to use the the shared pool
# Do this if your job will exceed 2 hours
#requirements = (cpu_experiment == "sdcc")

# each time the Queue command is called, it makes a new job
# and sends the last specified arguments. job_name will show
# up if you use the condortop job viewer

"""


def _append_job(fp, num, output_dir):
    fp.write("""\
+job_name = "sim-{num:05d}"
Arguments = 200 {num} {output_dir}
Queue

""".format(num=num, output_dir=output_dir))


n_patches = 10_000_000
n_jobs = n_patches // 200
n_jobs_per_script = 500
n_scripts = n_jobs // 500

cwd = os.path.abspath(os.getcwd())
try:
    os.makedirs(os.path.join(cwd, 'outputs'))
except Exception:
    pass

try:
    os.makedirs(os.path.join(cwd, 'outputs', 'logs'))
except Exception:
    pass

script_name = os.path.join(cwd, "job_condor.sh")
output_dir = os.path.join(cwd, "outputs")

script = PREAMBLE.format(script_name=script_name)

job_ind = 1
for snum in range(n_scripts):
    with open('condor_job_%05d.desc' % snum, 'w') as fp:
        fp.write(script)
        for num in range(job_ind, job_ind + 500):
            _append_job(fp, num, output_dir)

    job_ind += 500
