import os
import sys
from pathlib import Path

INFRA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(INFRA / "processor"))
sys.path.insert(0, str(INFRA.parent / "shared" / "src"))  # cpcb_aqi (installed as a package in the image)
os.environ.setdefault("AWS_REGION", "ap-south-1")
os.environ.setdefault("AWS_DEFAULT_REGION", "ap-south-1")
os.environ.setdefault("TABLE_NAME", "test-table")
os.environ.setdefault("BUCKET_NAME", "test-bucket")
os.environ.pop("BEDROCK_MODEL_ID", None)  # template unless a test sets it
