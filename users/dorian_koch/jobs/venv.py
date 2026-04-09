from asyncio import subprocess
import venv

from glob import glob
import json
from pathlib import Path
from typing import List, Sequence, Tuple
from sisyphus import Job, Task, tk
import os

class CreateVenv(Job):
	def __init__(self, *, packages: List[List[str]] = []):
		self.out_env_path = self.output_path("venv", directory=True)
		self.out_python_path = self.output_var("venv_python_path")
		self.packages = packages

	@classmethod
	def hash(cls, parsed_args):
		d = dict(**parsed_args)
		#d["__version"] = 5
		return super().hash(d)

	def tasks(self):
		yield Task("run", mini_task=True)

	def run(self):
		venv.create(self.out_env_path, with_pip=True)
		if os.name == 'nt': # Windows
			venv_python = os.path.join(self.out_env_path, 'Scripts', 'python.exe')
		else:               # Mac/Linux
			venv_python = os.path.join(self.out_env_path, 'bin', 'python')

		# Install dependencies if specified
		if self.packages:
			for pkgs in self.packages:
				subprocess.check_call([venv_python, '-m', 'pip', 'install'] + pkgs)

		self.out_python_path.set(venv_python)