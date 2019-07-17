import luigi
from pathlib import Path


class DirParametr(luigi.Parameter):

    def parse(self, x) -> Path:
        return Path(x)

class FileSystemParametr(DirParametr):
    pass