"""The HTTP surface: four routes in one module, ``api/main.py``.

Four routes do not need five files. ``routes_meta.py`` / ``routes_wall.py`` /
``routes_results.py`` / ``routes_upload.py`` / ``main.py`` was the shape this package
started as, and the only thing that split bought was four import cycles around one
process-wide replay loop that all of them had to share.

There is no ``POST /api/sequence``. The upload path is cut deliberately: it is a
file-upload attack surface on an unattended public box during a 14-day judging window,
and cutting it is a responsible-operation gain rather than a schedule one.
"""

__all__ = ["main"]
