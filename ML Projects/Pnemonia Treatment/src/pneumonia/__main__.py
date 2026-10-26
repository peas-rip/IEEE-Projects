"""Allow ``python -m pneumonia`` to invoke the CLI."""
from .cli import main

raise SystemExit(main())
