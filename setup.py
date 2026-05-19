
from pathlib import Path


with open(Path("artifacts/data_validation/status.txt"), "r") as f:
    status = f.read().split(" ")[-1]

print(type(status))