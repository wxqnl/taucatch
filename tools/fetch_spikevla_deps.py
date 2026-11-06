"""SDT-V3 model source (Spike-driven Transformer V3, BICLab) for SpikeVLA's vision backbone -> vendor/sdt_v3/models.py.

The upstream repository has no license file, so the source is fetched at a pinned commit rather than redistributed.
The only change is the unused `import torchinfo` line.

    python tools/fetch_spikevla_deps.py
"""
import hashlib
import urllib.request
from pathlib import Path

COMMIT = "00fcd1fd5d9c65fe9a26579ed512a20c6a303deb"
URL = ("https://raw.githubusercontent.com/BICLab/Spike-Driven-Transformer-V3/"
       f"{COMMIT}/SDT_V3/Classification/Model_Base/models.py")
SHA256 = "6b60714b62a6f9ab38a84c53cbc624307668a2e84e80c334552d4359647a0224"   # after dropping `import torchinfo`
TARGET = Path(__file__).resolve().parents[1] / "vendor/sdt_v3/models.py"


def main():
    text = urllib.request.urlopen(URL, timeout=60).read().decode()
    text = "".join(line for line in text.splitlines(keepends=True) if line.strip() != "import torchinfo")
    digest = hashlib.sha256(text.encode()).hexdigest()
    if digest != SHA256:
        raise SystemExit(f"unexpected SDT-V3 source (sha256 {digest})")
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text(text)
    print(f"{TARGET.relative_to(TARGET.parents[2])} ready")


if __name__ == "__main__":
    main()
