from __future__ import annotations

import sys

from .app_core import GameApp


def main():
    app = GameApp(difficulty="normal", target_fps=30)
    app.run()


if __name__ == "__main__":
    sys.exit(main())
