from pssh.clients import SSHClient
from pssh.exceptions import SFTPIOError
from typing import List, Literal, Optional, Union, Pattern
from dataclasses import dataclass, field
from pathlib import Path
import subprocess
import shlex
import tempfile
import os
from .models import ModelCache


@dataclass
class Config:
    # your username on the PIK Cluster
    username: str

    # will be used in naming the outputs
    jobname: str

    # working directory (remote)
    # will correspond to /p/tmp/{username}/{workdir}
    workdir: str

    # https://slurm.schedmd.com/sbatch.html#OPT_mail-user
    email_address: str

    # https://slurm.schedmd.com/sbatch.html#OPT_mem
    # Default units are megabytes. Different units can be specified using the suffix [K|M|G|T]
    memory: Union[int, str]

    # hostname of the cluster head
    host: str = 'cluster.pik-potsdam.de'

    # directory to place code (relative to {workdir})
    codedir: str = 'code'
    # directory to place data (relative to {workdir})
    datadir: str = 'data'
    # directory to place conda env (relative to {workdir})
    envdir: str = 'conda'

    # 2021.11: python 3.9
    # 2020.11: python 3.8.5
    # 2020.07: python 3.6, 3.7, 3.8
    anaconda: Literal['anaconda/2020.07', 'anaconda/2020.11', 'anaconda/2021.11'] = 'anaconda/2021.11'
    python: Literal['3.7', '3.8', '3.9'] = '3.9'

    # https://gitlab.pik-potsdam.de/hpc-documentation/cluster-documentation/-/blob/master/Cluster%20User%20Guide.md#basic-concepts
    # https://gitlab.pik-potsdam.de/hpc-documentation/cluster-documentation/-/blob/master/Cluster%20User%20Guide.md#requesting-a-gpu-node
    partition: Literal['standard', 'gpu', 'largemem', 'io'] = 'standard'
    gpu_type: Optional[Literal['v100', 'k40m']] = 'v100'  # NVidia Tesla K40m or Tesla V100
    n_gpus: Optional[Literal[1, 2]] = 1

    # https://gitlab.pik-potsdam.de/hpc-documentation/cluster-documentation/-/blob/master/Cluster%20User%20Guide.md#qos
    # short: max 24h; medium: max 7d; long: max 30d
    qos: Literal['short', 'medium', 'long'] = 'short'

    # https://gitlab.pik-potsdam.de/hpc-documentation/cluster-documentation/-/blob/master/Cluster%20User%20Guide.md#time-limits
    # Acceptable time formats include "minutes", "minutes:seconds", "hours:minutes:seconds",
    # "days-hours", "days-hours:minutes" and "days-hours:minutes:seconds".
    time_limit: Optional[str] = None

    # https://slurm.schedmd.com/sbatch.html#OPT_mail-type
    alerts: Optional[List[Literal['NONE', 'BEGIN', 'END', 'FAIL', 'REQUEUE', 'ALL']]] = \
        field(default_factory=lambda: ['END', 'FAIL'])

    # environment variables
    env_vars: dict = None
    # environment variables that are only set before running the script (not for prepping)
    env_vars_run: dict = None

    std_out: str = '%x-%j.out'
    std_err: str = '%x-%j.err'

    @property
    def workdir_path(self):
        return os.path.join('/p/tmp', self.username, self.workdir)

    @property
    def codedir_path(self):
        return os.path.join(self.workdir_path, self.codedir)

    @property
    def datadir_path(self):
        return os.path.join(self.workdir_path, self.datadir)

    @property
    def envdir_path(self):
        return os.path.join(self.workdir_path, self.envdir)

    def get_env_header(self, include_run_only: bool = True):
        ret = '\n\n'
        ret += f'export PYTHONPATH=$PYTHONPATH:{self.codedir_path}\n'
        ret += '\n# General environment variables'
        if self.env_vars is not None:
            for k, v in self.env_vars.items():
                ret += f'export {k}={v}\n'
        ret += '\n# Environment variables for script'
        if include_run_only and self.env_vars_run is not None:
            for k, v in self.env_vars_run.items():
                ret += f'export {k}={v}\n'
        ret = '\n\n'
        return ret

    def get_sbatch_header(self):
        ret = '\n\n'
        ret += f'#SBATCH --mail-user={self.email_address}\n'
        ret += f'#SBATCH --workdir={self.workdir_path}\n'
        ret += f'#SBATCH --jobname={self.jobname}\n'
        ret += f'#SBATCH --qos={self.qos}\n'
        if self.time_limit is not None:
            ret += f'#SBATCH --time={self.time_limit}\n'
        ret += f'#SBATCH --partition={self.partition}\n'
        if self.partition == 'gpu':
            ret += f'#SBATCH --gres=gpu:{self.gpu_type}:{self.n_gpus}\n'
        ret += f'#SBATCH --mem={self.memory}\n'
        ret += f'#SBATCH --output={self.std_out}\n'
        ret += f'#SBATCH --error={self.std_err}\n'
        ret += f'#SBATCH --mail-type={",".join(self.alerts)}\n'
        ret += '\n\n'
        return ret


class PythonEnvironment:
    def __init__(self, config: Config, ssh_client: SSHClient):
        self.config = config
        self.ssh_client = ssh_client

    def get_header(self):
        ret = '\n'
        # load the anaconda module
        ret += f'module load {self.config.anaconda}\n'
        # activate the environment
        ret += f'source activate {self.config.envdir_path}\n'
        return ret

    def get_run_command(self, script: str, kwargs: dict = None):
        args = ''
        if kwargs is not None:
            args = ' '.join([f' --{kw}={val}' for kw, val in kwargs.items()])
        return f'python {self.config.codedir_path}/{script} {args}'

    def prepare_environment(self):
        with self.ssh_client.open_shell() as shell:
            # ensure we are using bash
            shell.run('/bin/bash')
            # load the anaconda module
            shell.run(f'module load {self.config.anaconda}')
            # create anaconda environment
            shell.run(f'conda create --prefix={self.config.envdir_path} -y python={self.config.python}')
            # activate the environment
            shell.run(f'source activate {self.config.envdir_path}')
            # install dependencies
            shell.run(f'pip install --no-output -r {os.path.join(self.config.workdir_path, "requirements.txt")}')
        print('\n'.join(list(shell.output.stdout)))
        print('\n'.join(list(shell.output.stderr)))


class FileHandler:
    def __init__(self,
                 config: Config,
                 # local directory root all the following paths are based on
                 local_basepath: str,
                 # directories to look for code that should be uploaded relative to basepath
                 include_dirs: List[str],
                 # rules that we use to filter the list of files in prior parameter
                 # rules are applied relative to basepath
                 exclude_rules: List[str] = None,
                 # local path to the requirements.txt
                 requirements_txt: str = 'requirements.txt',
                 # list of (possibly large) data files to upload relative to basepath
                 data_files: List[str] = None,
                 # path to the local model cache
                 model_cache: str = None,
                 # models to cache
                 required_models: List[str] = None):
        self.config = config
        self.ssh_client = SSHClient(self.config.host, user=self.config.username)
        self.local_basepath = local_basepath
        self.requirements_txt = requirements_txt
        self.include_dirs = include_dirs
        self.exclude_rules = exclude_rules
        self.data_files = data_files
        self.model_cache = ModelCache(model_cache) if model_cache is not None else None
        self.required_models = required_models

    def attach_ssh_client(self, ssh_client: SSHClient):
        self.ssh_client = ssh_client

    @staticmethod
    def _parse_file_listing(listing: List[str], basepath: str = ''):
        ret = []
        for line in listing:
            line_parts = line.split(' ')
            if len(line_parts) >= 8:
                date = line_parts[5]
                time = line_parts[6]
                filepath = line_parts[7]
                ret.append((f'{date} {time}', filepath[len(basepath) + 1:]))
        return ret

    def _get_remote_file_listing(self, path: str, basepath: str = '') -> List[tuple]:
        directory_path = os.path.join(basepath, path)
        out = self.ssh_client.run_command(f'find {directory_path} -type f -print0 | '
                                          f'xargs -0 ls -l --time-style="+%F %T"')
        self.ssh_client.wait_finished(out)
        stderr = list(out.stderr)
        if len(stderr) > 0:
            # raise FileNotFoundError(f'Error retrieving listing for remote path: {directory_path}')
            return []
        stdout = list(out.stdout)
        return self._parse_file_listing(stdout, basepath)

    def _get_local_file_listing(self, path: str, basepath: str = '') -> List[tuple]:
        directory_path = os.path.join(basepath, path)
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f'This local file path does not exist: {directory_path}')
        p1 = subprocess.Popen(shlex.split(f'find {directory_path} -type f -print0'),
                              stdout=subprocess.PIPE)
        p2 = subprocess.Popen(shlex.split('xargs -0 ls -l --time-style="+%F %T"'),
                              stdin=p1.stdout, stdout=subprocess.PIPE)
        p1.stdout.close()
        out = p2.communicate()[0]
        listing = self._parse_file_listing(out.decode().split('\n'), basepath)

        if self.exclude_rules is not None:
            def has_match(fp):
                return sum([bool(p.match(fp)) for p in self.exclude_rules]) > 0

            listing = [(dt, fp) for dt, fp in listing if not has_match(fp)]

        return listing

    def upload_requirements_txt(self):
        local_req_txt = os.path.join(self.local_basepath, self.requirements_txt)
        remote_req_txt = os.path.join(self.config.workdir_path, 'requirements.txt')
        self.ssh_client.copy_file(local_req_txt, remote_req_txt)

    def sync_code(self, force_upload: bool = False):
        for directory in self.include_dirs:
            print('Fetching list of local files to upload...')
            local_listing = self._get_local_file_listing(directory, basepath=self.local_basepath)
            print('Fetching list of remote files to compare...')
            remote_listing = self._get_remote_file_listing(directory, basepath=self.config.codedir_path)
            remote_lookup = {fp: dt for fp, dt in remote_listing}
            for mod_date, local_filepath in local_listing:
                if force_upload or local_filepath not in remote_lookup or mod_date > remote_lookup[local_filepath]:
                    print(self.local_basepath, local_filepath)
                    print(f'Uploading file: {os.path.join(self.local_basepath, local_filepath)}')
                    self.ssh_client.copy_file(os.path.join(self.local_basepath, local_filepath),
                                              os.path.join(self.config.codedir_path, local_filepath))

    def upload_data(self):
        # TODO test if file needs uploading (eg by trying to download it and fetch SFTPIOError)
        for file in self.data_files:
            print(f'Uploading file: {os.path.join(self.local_basepath, file)}')
            self.ssh_client.copy_file(os.path.join(self.local_basepath, file),
                                      os.path.join(self.config.datadir_path, file))

    def cache_upload_models(self):
        # TODO test if model needs uploading (eg by trying to download it and fetch SFTPIOError)
        if self.model_cache is not None:
            for model_name in self.required_models:
                self.model_cache.cache_model(model_name)
                self.ssh_client.copy_file(
                    str(self.model_cache.get_model_path(model_name)),
                    os.path.join(self.config.datadir_path, 'models', self.model_cache.to_safe_name(model_name)),
                    recurse=True)


class ClusterJob:
    def __init__(self,
                 config: Config,
                 main_script: str,
                 script_params: dict):
        self.config = config
        self.ssh_client = SSHClient(self.config.host, user=self.config.username)
        self.py_env = PythonEnvironment(config, ssh_client=self.ssh_client)

        self.main_script = main_script
        self.script_params = script_params

    def _create_upload_slurm_script(self):
        with tempfile.TemporaryDirectory() as tmp_dirname:
            sjob_filename = 'slurm_job.sh'
            sjob_filepath_local = os.path.join(tmp_dirname, sjob_filename)
            sjob_filepath_remote = os.path.join(self.config.workdir_path, sjob_filename)
            with open(sjob_filepath_local, 'w') as f:
                f.write('#!/bin/bash\n')
                f.write('\n')
                f.write(self.config.get_sbatch_header())
                f.write(self.py_env.get_header())
                f.write(self.config.get_env_header())
                f.write(self.py_env.get_run_command(self.main_script, self.script_params))

            self.ssh_client.copy_file(sjob_filepath_local, sjob_filepath_remote)

    def submit_job(self):
        self._create_upload_slurm_script()
        out = self.ssh_client.run_command(f'sbatch {os.path.join(self.config.workdir_path, "slurm_job.sh")}')
        job_id = list(out.stdout)
        if len(job_id) > 0:
            job_id = job_id[0]
            print(f'Job id is: {job_id}')
            print('Follow outputs with:')
            print(f' $ tail -f {self.config.workdir_path}/{self.config.jobname}-{job_id}.out')
            print(f' $ tail -f {self.config.workdir_path}/{self.config.jobname}-{job_id}.err')
            print(f'Check job status with `$ squeue -u {self.config.username}`')
        else:
            print("Something went wrong (Couldn't read job id)...")

    def initialise(self, file_handler: FileHandler):
        file_handler.upload_requirements_txt()
        file_handler.sync_code(force_upload=True)
        self.py_env.prepare_environment()
        file_handler.upload_data()
        file_handler.cache_upload_models()

# export OPENBLAS_NUM_THREADS=1
# export TRANSFORMERS_OFFLINE=1
