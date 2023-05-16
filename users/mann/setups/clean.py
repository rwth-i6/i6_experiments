from sisyphus import tk, gs

import os

def clean(lattice_caches, mark_func, wer, **_ignored):
    # print("Finished score job: %s" % wer.creator)
    print("Cleaning lattice caches of %s: " % lattice_caches[1].creator, end="")
    for number, cache in lattice_caches.items():
        try:
            os.remove(cache.get_path())
            print(number, end="..")
        except FileNotFoundError:
            pass
    print("done")
    mark_func()
    
class LatticeCleaner:
    def __init__(self, output_subdir=gs.ALIAS_AND_OUTPUT_SUBDIR):
        self.output_subdir = output_subdir
        self._save_file = os.path.join("output", self.output_subdir, "cleaned_lattices.txt")
        self.cleaned_jobs = self.load_cleaned()
    
    def load_cleaned(self):
        try:
            with open(self._save_file) as f:
                return set(f.read().splitlines())
        except FileNotFoundError:
            # with open(self._save_file, "w") as f:
            #     pass
            return set()

    def is_cleaned(self, job):
        return job._sis_path() in self.cleaned_jobs
    
    def mark_cleaned(self, job):
        self.cleaned_jobs.add(job._sis_path())
        with open(self._save_file, "a") as f:
            f.write(job._sis_path())
            f.write("\n")
    
    def clean(self, job, wer):
        if self.is_cleaned(job):
            return
        def mark_cleaned():
            self.mark_cleaned(job)
        tk.register_callback(
            clean,
            job.out_single_lattice_caches,
            wer=wer,
            mark_func=mark_cleaned
        )

