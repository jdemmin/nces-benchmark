import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ToDictJSONEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if hasattr(obj, "to_dict") and callable(obj.to_dict):
            return obj.to_dict()
        # numpy scalars (e.g. leaked from a DataFrame) unbox to native
        # Python types instead of being silently stringified.
        if hasattr(obj, "item"):
            return obj.item()

        return str(obj)


def write_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            indent=2,
            ensure_ascii=False,
            cls=ToDictJSONEncoder,
        )

    logger.info("Wrote %s", path)