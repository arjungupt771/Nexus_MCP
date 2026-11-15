from datetime import UTC, datetime

from fastapi import APIRouter

from apps.backend.app.core.config import settings

router = APIRouter(tags=["Health"])


@router.get("/health")
async def health_check() -> dict[str, str]:
    return {
        "status": "healthy",
        "application": settings.app_name,
        "version": settings.app_version,
        "timestamp": datetime.now(UTC).isoformat(),
    }
