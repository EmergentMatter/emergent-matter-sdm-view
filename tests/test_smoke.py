"""Smoke test: package imports cleanly without bpy on the path."""


def test_io_imports():
    """io subpackage is pure-Python, must work outside Blender."""
    # Importability is what's under test; the names themselves are unused.
    from sdm_view import io  # noqa: F401
    from sdm_view.io import annotation_state, vdb_export  # noqa: F401


def test_version():
    import sdm_view

    assert sdm_view.__version__
