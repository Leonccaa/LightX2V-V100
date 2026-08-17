__version__ = "0.1.0"
__author__ = "LightX2V Contributors"
__license__ = "Apache 2.0"

import os

import lightx2v_platform.set_ai_device

if os.getenv("LIGHTX2V_MINIMAL_IMPORT", "0") == "1":
    # The isolated H3 TP validation runner imports only its native stack. Avoid
    # eagerly importing every optional model backend through pipeline.py.
    __all__ = ["__version__", "__author__", "__license__"]
else:
    from lightx2v import common, models, utils
    from lightx2v.pipeline import LightX2VPipeline

    __all__ = [
        "__version__",
        "__author__",
        "__license__",
        "models",
        "common",
        "utils",
        "LightX2VPipeline",
    ]
