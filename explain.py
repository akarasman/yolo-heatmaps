#!/usr/bin/env python
"""
Repo-local convenience shim for `yolo_lrp.cli` - lets `python
explain.py ...` keep working from a checkout without installing the
package. The real implementation lives in yolo_lrp/cli.py; once
installed (`pip install .`), use the `yolo-lrp` console script
instead.
"""

from yolo_lrp.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
