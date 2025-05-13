from enum import Enum


class Criterion(Enum):
    ME = "ME"
    LFMMI = "lf_mmi"
    LFSMBR = "lf_smbr"


    def __str__(self):
        return self.value