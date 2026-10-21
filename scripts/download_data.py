
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

URL = (
    "https://raw.githubusercontent.com/IBM/"
    "telco-customer-churn-on-icp4d/master/"
    "data/Telco-Customer-Churn.csv"
)

EXPECTED_SHA256 = (
    "16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91"
)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw"
CSV_PATH = DATA_DIR / "telco.csv"
SOURCE_PATH = DATA_DIR / "telco.source.json"


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not CSV_PATH.exists():
        print("Downloading dataset...")
        with urlopen(URL, timeout=30) as response:
            content = response.read()
        CSV_PATH.write_bytes(content)
    else:
        content = CSV_PATH.read_bytes()

    sha256 = hashlib.sha256(content).hexdigest()

    if sha256 != EXPECTED_SHA256:
        CSV_PATH.unlink(missing_ok=True)
        raise ValueError(
            f"SHA-256 mismatch: expected {EXPECTED_SHA256}, got {sha256}"
        )

    metadata = {
        "source": URL,
        "sha256": sha256,
    }
    SOURCE_PATH.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Dataset: {CSV_PATH}")
    print(f"SHA-256: {sha256}")
    print(f"Metadata: {SOURCE_PATH}")


if __name__ == "__main__":
    main()
