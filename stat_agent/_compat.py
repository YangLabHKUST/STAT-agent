"""Compatibility patches for third-party packages.

Applied at import time to fix known incompatibilities between
dependencies (e.g., SpatialDE requiring scipy.misc.derivative
which was removed in scipy >= 1.14).
"""
import importlib


def _patch_scipy_derivative():
    """Restore scipy.misc.derivative for packages that need it (e.g., SpatialDE)."""
    try:
        from scipy.misc import derivative  # noqa: F401
        return  # Already available, no patch needed
    except ImportError:
        pass

    # Define a minimal finite-difference derivative matching the old scipy API
    def derivative(func, x0, dx=1.0, n=1, args=(), order=3):
        """Numerical derivative using central differences."""
        if n == 1:
            return (func(x0 + dx, *args) - func(x0 - dx, *args)) / (2.0 * dx)
        elif n == 2:
            return (func(x0 + dx, *args) - 2.0 * func(x0, *args) + func(x0 - dx, *args)) / (dx ** 2)
        else:
            raise NotImplementedError(f"Only n=1,2 supported, got n={n}")

    # Inject into scipy.misc so `from scipy.misc import derivative` works
    try:
        import scipy.misc
        scipy.misc.derivative = derivative
    except Exception:
        pass


# Apply patches on import
_patch_scipy_derivative()
