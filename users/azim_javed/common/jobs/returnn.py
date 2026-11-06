import os
import gzip
import shutil
import subprocess
import tempfile

from i6_core.returnn.forward import ReturnnForwardJobV2


class ResumableReturnnForwardJobV2(ReturnnForwardJobV2):
    def run(self):
        for name in ("rasr.recog.log", "rasr.align.log"):
            if name not in self.out_files or not os.path.isfile(name):
                continue
            with open(name, "rb") as source:
                if source.read(2) != b"\x1f\x8b":
                    continue
                source.seek(0)
                with tempfile.NamedTemporaryFile(dir=os.path.dirname(os.path.abspath(name)), delete=False) as target:
                    temporary_path = target.name
                    try:
                        with gzip.GzipFile(fileobj=source, mode="rb") as reader:
                            shutil.copyfileobj(reader, target)
                        target.flush()
                        os.fsync(target.fileno())
                        os.replace(temporary_path, name)
                    finally:
                        if os.path.exists(temporary_path):
                            os.unlink(temporary_path)

        call = [
            self.returnn_python_exe.get_path(),
            os.path.join(self.returnn_root.get_path(), "rnn.py"),
            self.out_returnn_config_file.get_path(),
        ]
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = str(self.rqmt["cpu"])
        env["MKL_NUM_THREADS"] = str(self.rqmt["cpu"])

        subprocess.check_call(call, env=env)

        for name, output in self.out_files.items():
            assert os.path.exists(name), f"Output file {name} does not exist"
            output_path = output.get_path()
            # Keep real results in output/ so cleaning work/ is safe. Preserve
            # their work/ paths for resumable runs without duplicating data.
            if os.path.exists(output_path) and os.path.samefile(name, output_path):
                continue
            shutil.move(name, output_path)
            os.symlink(os.path.relpath(output_path, os.path.dirname(os.path.abspath(name))), name)

        self._compress_outputs()
