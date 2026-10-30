"""Run the exported synthetic tests with outbound socket connections blocked."""
from pathlib import Path
import socket
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def blocked(*args, **kwargs):
    raise AssertionError("Network access is disabled during offline validation")


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
    with patch.object(socket, "create_connection", side_effect=blocked), \
         patch.object(socket.socket, "connect", side_effect=blocked), \
         patch.object(socket.socket, "connect_ex", side_effect=blocked):
        result = unittest.TextTestRunner(verbosity=1).run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
