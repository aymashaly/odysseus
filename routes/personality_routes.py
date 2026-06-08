# routes/personality_routes.py
"""Routes for loading the bundled personality files (Rose, etc.) into the
Harness editor. The memory.md file in the project root is treated as the
source of truth for the Rose personality; this endpoint just streams it
as plain text so the frontend can paste it into the harness textarea.
"""
import os
import logging
from fastapi import APIRouter, HTTPException, Request

from core.constants import BASE_DIR
from src.auth_helpers import get_current_user

logger = logging.getLogger(__name__)

# Whitelist of personality files that can be served. The keys are the
# `name` passed in the URL (`/api/personality/<name>`), the values are
# the relative paths from BASE_DIR. We never serve anything else — this
# is read-only file disclosure, so keep the surface small.
_PERSONALITY_FILES = {
    "rose": "memory.md",
}


def setup_personality_routes():
    router = APIRouter(prefix="/api/personality", tags=["personality"])

    @router.get("/{name}")
    def get_personality(request: Request, name: str):
        # Auth-gated: same posture as the rest of the API. Auth-disabled
        # single-user mode still works (get_current_user returns None and
        # is treated as authorized).
        _ = get_current_user(request)

        rel = _PERSONALITY_FILES.get(name)
        if not rel:
            raise HTTPException(404, f"Unknown personality '{name}'")

        # Resolve and confirm the path is inside BASE_DIR. Defense in
        # depth — `_PERSONALITY_FILES` is hard-coded, but a future edit
        # should not be able to escape the project root.
        path = os.path.normpath(os.path.join(BASE_DIR, rel))
        base = os.path.normpath(BASE_DIR)
        if not (path == base or path.startswith(base + os.sep)):
            logger.warning("personality file %s escaped BASE_DIR; refusing", name)
            raise HTTPException(404, f"Unknown personality '{name}'")

        if not os.path.isfile(path):
            raise HTTPException(404, f"Personality file '{name}' not found")

        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
        except OSError as e:
            logger.error("Failed to read personality file %s: %s", path, e)
            raise HTTPException(500, "Failed to read personality file")

        return {"name": name, "content": content}

    return router
