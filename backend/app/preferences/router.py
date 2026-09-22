"""Settings endpoints — CRUD for user preferences."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.core.constants import MASKED_API_KEY
from app.preferences.user_settings import (
    UserSettings,
    get_user_settings,
    sync_to_runtime,
    update_user_settings,
)
from app.stt.config import stt_settings

router = APIRouter(prefix="/settings", tags=["Settings"])

_KEY_FIELDS = {"gemini_api_key", "groq_api_key"}


def _mask_keys(s: UserSettings) -> UserSettings:
    """Return a copy of s with key fields replaced by the masked placeholder or empty string."""
    return s.model_copy(update={
        f: (MASKED_API_KEY if getattr(s, f) else "")
        for f in _KEY_FIELDS
    })


class SettingsUpdateResponse(BaseModel):
    settings: UserSettings
    warning: str | None = None


@router.get("", response_model=UserSettings)
async def get_settings():
    return _mask_keys(get_user_settings())


@router.put("", response_model=SettingsUpdateResponse)
async def put_settings(updates: dict):
    """Apply a partial settings update.

    ``update_user_settings`` raises ``ConfigurationError`` for a bad directory or model size, which
    answers 400 app-wide; ``ValueError`` is pydantic, ``RuntimeError`` a relocate that broke midway.
    """
    allowed_fields = set(UserSettings.model_fields.keys())
    filtered = {
        k: v for k, v in updates.items()
        if k in allowed_fields
        and not (k in _KEY_FIELDS and v == MASKED_API_KEY)
    }
    try:
        outcome = update_user_settings(filtered)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    changed_stt = sync_to_runtime(outcome.settings)
    if changed_stt:
        from app.stt.local_setup import maybe_prewarm_local

        maybe_prewarm_local(stt_settings)
    return SettingsUpdateResponse(settings=_mask_keys(outcome.settings), warning=outcome.warning)


class CloudKeyStatus(BaseModel):
    gemini_key_set: bool
    groq_key_set: bool


@router.get("/cloud-status", response_model=CloudKeyStatus)
async def cloud_key_status():
    """Whether each Cloud API key is currently active in the runtime config.

    Checks the runtime STT settings (not UserSettings) so that keys provided
    via .env are correctly reflected even if the user has never opened Settings → Keys.
    """
    return CloudKeyStatus(
        gemini_key_set=bool(stt_settings.gemini_api_key),
        groq_key_set=bool(stt_settings.groq_api_key),
    )

