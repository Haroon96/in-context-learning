from typing import Any
from pathlib import Path

class Logger:
    def __init__(self, outfile = None):
        from rich.console import Console
        self.std_console = Console()
        self.file_console = Console(file=open(outfile, "w")) if outfile else None

    def log(
        self,
        *objects: Any,
        sep: str = " ",
        end: str = "\n",
        stdout = True,
        file = True,
    ):
        if stdout: self.std_console.print(*objects, sep=sep, end=end)
        if file and self.file_console: self.file_console.print(*objects, sep=sep, end=end)

def symlink_force(target: Path, link_name: Path):
    import errno
    try:
        # os.symlink(target, link_name)
        link_name.symlink_to(target)
    except OSError as e:
        if e.errno == errno.EEXIST:
            link_name.unlink()
            link_name.symlink_to(target)
            # os.remove(link_name)
            # os.symlink(target, link_name)
        else:
            raise e
