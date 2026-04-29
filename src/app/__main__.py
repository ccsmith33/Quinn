"""`python -m src.app` invokes this module — see `quinn-agent.service`.

Kept thin: the real entrypoint is `src.app.main()`.
"""

from __future__ import annotations

import sys

from . import main

if __name__ == "__main__":
    sys.exit(main())
