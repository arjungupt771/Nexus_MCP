from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from apps.backend.app.api.routes.health import router as health_router
from apps.backend.app.core.config import settings
from apps.backend.app.core.logging import configure_logging

configure_logging(settings.log_level)
logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    logger.info(
        "application_starting",
        app_name=settings.app_name,
        environment=settings.app_env,
    )
    yield
    logger.info("application_stopping")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Multi-server MCP integration and agent orchestration platform.",
    debug=settings.debug,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router, prefix=settings.api_v1_prefix)


@app.get("/")
async def root() -> dict[str, str]:
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
        "health": f"{settings.api_v1_prefix}/health",
    }
