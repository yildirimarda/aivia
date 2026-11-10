"""Local folder delivery: copy finished artifacts into a directory."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

from aivia.core.config.http import HttpClientConfig
from aivia.core.models.base import utcnow
from aivia.core.models.provider import ProviderMeta, ProviderPrice
from aivia.providers.base import Provider
from aivia.providers.protocols.delivery import (
    CAPABILITY_DELIVERY,
    DeliveryArtifact,
    DeliveryMethod,
    DeliveryReceipt,
    DeliveryRequest,
)

__all__ = ["FolderDeliveryProvider", "LocalFolderDeliveryProvider"]

_DEFAULT_METADATA = ProviderMeta(
    name="folder",
    capabilities=[CAPABILITY_DELIVERY],
    pricing={CAPABILITY_DELIVERY: ProviderPrice(amount_usd=0.0)},
)


def _artifact_destination(artifact: DeliveryArtifact, index: int) -> str:
    if artifact.filename:
        return artifact.filename
    if artifact.data is not None:
        digest = hashlib.sha256(artifact.data).hexdigest()[:12]
        return f"artifact-{index}-{digest}"
    if artifact.path is not None:
        return Path(artifact.path).name or f"artifact-{index}"
    return f"artifact-{index}"


class LocalFolderDeliveryProvider(Provider):
    """Copy finished artifacts into a local folder.

    ``path`` is the destination directory. Relative paths are resolved against
    the current working directory when a delivery runs, so a test can pass a
    temporary absolute path and a deployment can pin an absolute output dir.
    """

    default_metadata = _DEFAULT_METADATA

    def __init__(
        self,
        path: str,
        *,
        name: str | None = None,
        health_timeout_seconds: float | None = None,
        close_timeout_seconds: float | None = None,
        http_config: HttpClientConfig | None = None,
    ) -> None:
        super().__init__(
            name=name,
            health_timeout_seconds=health_timeout_seconds,
            close_timeout_seconds=close_timeout_seconds,
            http_config=http_config,
        )
        self._path = Path(path)

    @property
    def path(self) -> Path:
        """The destination directory."""
        return self._path

    def _folder(self) -> Path:
        folder = self._path
        if not folder.is_absolute():
            folder = Path.cwd() / folder
        return folder

    async def deliver(self, request: DeliveryRequest) -> DeliveryReceipt:
        """Copy every artifact into the folder and acknowledge the delivery."""
        if not isinstance(request, DeliveryRequest):
            raise TypeError("request must be a DeliveryRequest")
        if request.method != DeliveryMethod.FOLDER:
            raise ValueError(
                f"folder provider cannot deliver with method {request.method.value!r}"
            )
        folder = self._folder()
        folder.mkdir(parents=True, exist_ok=True)
        for index, artifact in enumerate(request.artifacts):
            destination = folder / _artifact_destination(artifact, index)
            if artifact.data is not None:
                destination.write_bytes(artifact.data)
            elif artifact.path is not None:
                shutil.copyfile(artifact.path, destination)
            else:
                raise ValueError(
                    f"folder delivery cannot fetch artifact {index} from a URL"
                )
        return DeliveryReceipt(
            method=DeliveryMethod.FOLDER,
            target=str(folder),
            provider=self.name,
            delivered_at=utcnow(),
            metadata={"folder": str(folder), "artifacts": len(request.artifacts)},
        )


FolderDeliveryProvider = LocalFolderDeliveryProvider
"""Alias for :class:`LocalFolderDeliveryProvider`."""
