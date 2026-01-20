class ArtifactError(Exception):
    """Base class for errors related to benchmark artifacts."""


class ArtifactCorruptedError(ArtifactError):
    """Raised when an artifact is missing required metadata or has invalid structure."""

    def __init__(self, message: str, path: str | None = None):
        self.path = path
        super().__init__(f"{message} (Path: {path})" if path else message)


class InconsistentInstanceError(ArtifactError, TypeError):
    """Raised when instances in an artifact are not of the same class."""

    pass


class BenchmarkNotFoundError(ImportError):
    """Raised when a requested benchmark cannot be found in the library."""

    pass
