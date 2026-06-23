from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("histvae")
except PackageNotFoundError:  # pragma: no cover - fallback for non-installed usage
    __version__ = "0+unknown"

from .core import HistVAE
from .preprocessing import (
    AxisPreprocessingSpec,
    GroupCoordinateNormalizer,
    HistogramPreprocessor,
)

from .visualization import (
    histogram_bin_edges,
    plot_hist,
    plot_reconstruction,
    plot_scatter,
    prepare_histogram_for_plot,
)

from .optimal_transport import (
    JointSinkhornDivergence,
    build_joint_bin_support,
    validate_ot_config,
)

from .reference import (
    aggregate_latent_views,
    empirical_reference_percentile,
    euclidean_reference_knn,
    reference_knn_from_distances,
)
