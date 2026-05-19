from __future__ import annotations

import logging


def suppress_http_logs() -> None:
    for logger_name in ("httpx", "httpcore", "huggingface_hub"):
        logging.getLogger(logger_name).setLevel(logging.WARNING)
