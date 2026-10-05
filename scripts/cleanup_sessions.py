"""Delete expired auth sessions, password-reset tokens, and pre-auth CSRF tokens."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from persistence.store import SQLiteConversationStore


def main() -> int:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    counts = SQLiteConversationStore().delete_expired_auth_tokens(now)
    print(json.dumps({"deleted": counts}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
