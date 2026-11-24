"""Entry point kept for ``uvicorn server.app:app`` and ``openenv serve``.

The server lives in ``recruitenv.server`` (installed with the ``openenv``
extra); this module is not part of the package.
"""

from recruitenv.server import app, main  # noqa: F401

if __name__ == "__main__":
    main()
