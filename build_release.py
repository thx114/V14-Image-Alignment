"""Build a source-software ZIP from tracked files only; no data or dependencies."""
import hashlib
from pathlib import Path
import subprocess
import zipfile

root = Path(__file__).resolve().parent
paths = subprocess.check_output(['git', 'ls-files', '-z'], cwd=root).decode().split('\0')
dest = root / 'dist'
dest.mkdir(exist_ok=True)
archive = dest / 'V14-Image-Alignment-v1.0.0-source.zip'
with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED) as output:
    for name in filter(None, paths):
        output.write(root / name, 'V14-Image-Alignment/' + name)
with zipfile.ZipFile(archive) as output:
    assert output.testzip() is None
digest = hashlib.sha256(archive.read_bytes()).hexdigest()
(dest / 'SHA256SUMS.txt').write_text(f'{digest}  {archive.name}\n', encoding='ascii')
print(archive)
