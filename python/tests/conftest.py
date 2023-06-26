import sys
import os
from pathlib import Path

#this dir
this_dir = Path(os.path.dirname(os.path.abspath(__file__)))
#parent dir
parent_dir = this_dir.parent
# module dir
module_dir = parent_dir / "module"

print(module_dir)
sys.path.append(str(module_dir))