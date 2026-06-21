from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("histvae")
except PackageNotFoundError:  # pragma: no cover - fallback for non-installed usage
    __version__ = "0+unknown"

from .core import HistVAE
from .preprocessing import AxisPreprocessingSpec, HistogramPreprocessor

from .visualization import (
    histogram_bin_edges,
    plot_hist,
    plot_reconstruction,
    plot_scatter,
    prepare_histogram_for_plot,
)
