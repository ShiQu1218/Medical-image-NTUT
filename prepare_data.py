"""Download the public BUSI mirror; never overwrite source images."""
import hashlib
from pathlib import Path
import shutil
import urllib.request
import zipfile

URL = "https://www.kaggle.com/api/v1/datasets/download/aryashah2k/breast-ultrasound-images-dataset"
SHA256 = "7fffc86a517934da55021f66641d81383058d26a30adb0ffab3780ca7e52d57d"


def main():
    root = Path(__file__).resolve().parent / "data"
    root.mkdir(exist_ok=True)
    archive = root / "BUSI.zip"
    if not archive.exists():
        temporary = archive.with_suffix(".part")
        with urllib.request.urlopen(URL, timeout=120) as response, temporary.open("wb") as out:
            shutil.copyfileobj(response, out)
        temporary.replace(archive)
    digest = hashlib.file_digest(archive.open("rb"), "sha256").hexdigest()
    if digest != SHA256:
        raise ValueError(f"Dataset archive changed: {digest}; verify the source before use")
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            target = (root / member.filename).resolve()
            if not target.is_relative_to(root.resolve()):
                raise ValueError("Unsafe archive path")
            if not target.exists():
                z.extract(member, root)
    print("BUSI verified:", root / "Dataset_BUSI_with_GT")


if __name__ == "__main__":
    main()
