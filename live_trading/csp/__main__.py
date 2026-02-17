"""Allow running as `python -m csp`."""

from csp.main import main
import asyncio

asyncio.run(main())
