"""Compatibility entry; use run_batch.py for new commands."""
import sys
from run_batch import main
if __name__ == "__main__":
 sys.argv[1:1]=["--model","hunyuan"]
 main()
