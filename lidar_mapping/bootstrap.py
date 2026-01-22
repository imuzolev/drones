"""Bootstrap for ProjectAirSim imports (keeps original behavior)."""

import asyncio
import argparse
import contextlib
import json
import math
import os
import random
import sys
import tempfile
import threading
import time
from collections import deque
from pathlib import Path
from typing import Optional, Tuple, List

# Root of the ProjectAirSim repository (lidar_mapping/ lives directly inside it)
REPO_ROOT = Path(__file__).resolve().parents[1]


def _ensure_projectairsim_on_path() -> None:
    """
    Делает импорт `projectairsim` работоспособным при запуске скрипта из корня репозитория,
    без установки пакета в site-packages.
    """
    client_python = REPO_ROOT / "client" / "python"
    if client_python.exists():
        sys.path.insert(0, str(client_python))


_ensure_projectairsim_on_path()

from projectairsim import ProjectAirSimClient, Drone, World  # noqa: E402


