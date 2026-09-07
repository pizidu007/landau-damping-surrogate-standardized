#!/wangx/home/duxinxu/software/micromamba-root-v1/envs/gkeyll-build/bin/python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from landau_surrogate.tools.freeze_gkeyll_continuum_v1_splits import main


if __name__ == "__main__":
    main()
