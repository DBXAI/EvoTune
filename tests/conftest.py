import sys
from pathlib import Path

# Get the project root directory
root_dir = Path(__file__).parent.parent

# Add the project root directory to Python path
sys.path.append(str(root_dir))

# Runtime modules import packages from the src directory as top-level packages
# (for example, agent.*). Keep tests aligned with the application entrypoints.
src_dir = root_dir / "src"
sys.path.append(str(src_dir))
