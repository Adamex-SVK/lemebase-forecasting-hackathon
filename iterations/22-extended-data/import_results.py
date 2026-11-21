"""Read a GPU result archive, accepting only the expected forecast arrays and metadata."""
import argparse
from pathlib import Path
import re
from tarfile import open as open_tar

HERE = Path(__file__).resolve().parent
ALLOWED = re.compile(r"(?:test|[ABCD])-(?:0|13|26|39|16|32|final)-(?:zero|own|pooled)/(?:forecast\.npz|complete\.json)")


def load(path):
    out = HERE/"results/gpu"
    with open_tar(path,"r:gz") as archive:
        for item in archive.getmembers():
            name = item.name.removeprefix("./")
            if item.isdir():
                continue
            if not item.isfile() or not ALLOWED.fullmatch(name) or item.size>20_000_000:
                raise ValueError(f"Unexpected GPU archive entry {name!r}")
            payload = archive.extractfile(item).read()
            target = out/name
            if target.exists():
                assert target.read_bytes()==payload, f"Completed result changed: {name}"
                continue
            target.parent.mkdir(exist_ok=True,parents=True)
            temp = target.with_suffix(".part")
            temp.write_bytes(payload)
            temp.replace(target)
    print(f"{len(list(out.glob('*/complete.json')))} completed arms imported")


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("archive",type=Path)
    load(p.parse_args().archive)
