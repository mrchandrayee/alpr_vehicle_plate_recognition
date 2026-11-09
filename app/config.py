from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1]
CONFIG=yaml.safe_load((ROOT/"configs"/"config.yaml").read_text())
UPLOAD_DIR=ROOT/"uploads"
OUTPUT_DIR=ROOT/"outputs"
for p in (UPLOAD_DIR,OUTPUT_DIR): p.mkdir(exist_ok=True)
