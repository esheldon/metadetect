#!/usr/bin/env python
import os
import sys

PREAMBLE = """\
#
# always keep these at the defaults
#
Universe        = vanilla
kill_sig        = SIGINT
+Experiment     = "astro"
GetEnv          = True       # copy env. variables to the job

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


def _append_job(script, num, output_dir):
    script += """\
+job_name = "sim-{num:05d}"
Arguments = 2 {num} {output_dir}
Queue

""".format(num=num, output_dir=output_dir)
    return script


cwd = ("/astro/u/beckermr/workarea/des_y3_shear/metadetect/"
       "matts_simple_sims/v15")
script_name = os.path.join(cwd, "job_condor.sh")
output_dir = os.path.join(cwd, "outputs")

script = PREAMBLE.format(script_name=script_name)

for num in range(int(sys.argv[1])):
    script = _append_job(script, num, output_dir)

with open('condor_job.desc', 'w') as fp:
    fp.write(script)
