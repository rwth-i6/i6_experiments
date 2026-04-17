from sisyphus import Job, Task, tk
import os
import shutil


class AggregateFoldersJob(Job):
    def __init__(self, *, inputs: dict[tk.Path, str | tuple[str, str]]):
        self.inputs = inputs
        self.out_dir = self.output_path("out_folders", directory=True)

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        for path, out_path in self.inputs.items():
            prefix = ""
            if isinstance(out_path, tuple):
                out_path, prefix = out_path

            def copy_func(src, dst):
                if prefix:
                    dst = os.path.join(
                        os.path.dirname(dst), prefix + os.path.basename(dst)
                    )
                shutil.copy2(src, dst)

            shutil.copytree(
                path.get(),
                os.path.join(self.out_dir.get(), out_path),
                dirs_exist_ok=True,
                copy_function=copy_func,
            )
