"""SuperMimic for the MS-Human-700 whole-body musculoskeletal model, on mjlab."""

# `import mjlab` loads every `mjlab.tasks` entry point, ours included (muscle_supermimic.tasks). Importing
# it first, before any submodule of this package starts initializing, keeps that from re-entering a
# half-imported submodule when the first import of this package is not through mjlab.
import mjlab  # noqa: F401
