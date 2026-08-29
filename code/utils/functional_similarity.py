"""Pairwise functional similarity between simultaneously imaged neurons.

The prepared table `cell_cell_correlations_by_stimulus_coregistered.feather` holds one
number per neuron pair per stimulus condition: the **Pearson correlation** of the two
ΔF/F traces, computed over the frames belonging to that condition. It is produced by
`code/workshops/Functional Data Cell-Cell Correlations.ipynb`, which reads the V1DD
two-photon NWB-Zarr sessions, and its `session_correlation_table` is the function these
helpers generalise.

Pearson r is a choice, not the only measure of "these two neurons do the same thing".
It is invariant to each cell's baseline and gain, counts every frame equally, and is
linear. A different question wants a different measure — signal correlation over
trial-averaged responses if you care about tuning rather than shared noise, cosine
similarity if absolute response level matters, rank correlation if you distrust the
amplitude of ΔF/F. So `pairwise_similarity_table` takes the measure as an argument:

    from functional_similarity import pairwise_similarity_table, cosine_similarity

    corr_df = pairwise_similarity_table(dff, stim_mask, stim_names)                 # Pearson
    cos_df = pairwise_similarity_table(dff, stim_mask, stim_names, cosine_similarity)

Both return the same long format the prepared table uses, so anything downstream of it
works unchanged.
"""

from typing import Callable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import rankdata

__all__ = [
    "pearson_correlation",
    "cosine_similarity",
    "spearman_correlation",
    "negative_euclidean_distance",
    "signal_correlation",
    "pairwise_similarity_table",
]


# ----------------------------------------------------------------- similarity measures
# Each takes traces of shape (n_neurons, n_frames) and returns an (n_neurons, n_neurons)
# matrix. Write your own with the same signature to pass it to the functions below.


def pearson_correlation(traces: np.ndarray) -> np.ndarray:
    """Correlation of the two traces. Invariant to each cell's baseline and gain.

    This is what the prepared correlation table contains.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.corrcoef(traces)


def cosine_similarity(traces: np.ndarray) -> np.ndarray:
    """Cosine of the angle between the traces, *without* subtracting the mean.

    Unlike Pearson, two cells that are both simply active a lot score highly even if
    their fluctuations are unrelated. Use it when the absolute response level is part
    of what you mean by similarity.
    """
    norms = np.linalg.norm(traces, axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        unit = traces / norms
    return unit @ unit.T


def spearman_correlation(traces: np.ndarray) -> np.ndarray:
    """Pearson correlation of the ranked traces.

    Sensitive only to the ordering of frames, so a few large transients cannot dominate
    the estimate. Useful if you do not trust ΔF/F amplitudes to be comparable.
    """
    return pearson_correlation(rankdata(traces, axis=1))


def negative_euclidean_distance(traces: np.ndarray) -> np.ndarray:
    """Negated Euclidean distance between traces, so that larger is still more similar.

    A *distance* rather than a correlation: it does care about offset and scale, and it
    has no fixed range, which matters if you compare values across sessions.
    """
    sq = np.sum(traces**2, axis=1)
    d2 = sq[:, None] + sq[None, :] - 2 * (traces @ traces.T)
    return -np.sqrt(np.maximum(d2, 0.0))


def signal_correlation(trial_responses: np.ndarray) -> np.ndarray:
    """Pearson correlation over *trial-averaged* responses, i.e. tuning similarity.

    Pass an array of shape (n_neurons, n_conditions) holding each neuron's mean
    response per stimulus condition, rather than raw frames. Shared trial-to-trial
    noise then cannot inflate the estimate — what is left is agreement about which
    stimuli drive the cell.
    """
    return pearson_correlation(trial_responses)


# ----------------------------------------------------------------- the table


def _pair_indices(n: int, symmetric: bool) -> Tuple[np.ndarray, np.ndarray]:
    """Row/column indices of the off-diagonal pairs, in the prepared table's convention."""
    i_upper, j_upper = np.triu_indices(n, k=1)
    if symmetric:
        # Emit (i, j) and (j, i). The prepared table does this, so a merge on an
        # ordered (pre, post) key finds a row whichever way round the pair appears.
        return (
            np.concatenate([i_upper, j_upper]),
            np.concatenate([j_upper, i_upper]),
        )
    return i_upper, j_upper


def pairwise_similarity_table(
    traces: np.ndarray,
    stimulus_mask: Optional[np.ndarray] = None,
    stimulus_names: Optional[Sequence] = None,
    metric: Callable[[np.ndarray], np.ndarray] = pearson_correlation,
    neuron_ids: Optional[Sequence] = None,
    id_columns: Tuple[str, str] = ("pre_pt_root_id", "post_pt_root_id"),
    min_frames: int = 60,
    symmetric: bool = True,
    dtype=np.float32,
) -> pd.DataFrame:
    """One row per neuron pair, one column per stimulus condition.

    Parameters
    ----------
    traces:
        (n_neurons, n_frames) activity, e.g. `session_data["dff"]` from the functional
        data notebook.
    stimulus_mask:
        Length-`n_frames` labels saying which condition each frame belongs to, e.g.
        `stim_mask`. `None` treats every frame as one condition named `"all"`.
    stimulus_names:
        Optional mapping from mask value to column name — a dict, or a sequence aligned
        with `np.unique(stimulus_mask)`. Mask values are used as-is when omitted.
    metric:
        Any (n_neurons, n_frames) -> (n_neurons, n_neurons) callable. Defaults to
        `pearson_correlation`, reproducing the prepared table.
    neuron_ids:
        Length-`n_neurons` identifiers written into the two id columns. Defaults to
        positional indices, which is what you want before coregistration attaches EM
        root ids.
    id_columns:
        Names for the pre/post id columns. The defaults match the prepared table, so
        the result is a drop-in replacement for it.
    min_frames:
        Conditions with fewer frames than this are skipped, with a warning — a
        correlation over a handful of frames is mostly noise.
    symmetric:
        Emit both orderings of each pair (the prepared table's convention).

    Returns
    -------
    A DataFrame with `id_columns` plus one float column per retained condition.
    """
    traces = np.asarray(traces)
    if traces.ndim != 2:
        raise ValueError(f"traces must be (n_neurons, n_frames), got {traces.shape}")
    n_neurons, n_frames = traces.shape
    if n_neurons < 2:
        raise ValueError(f"need at least 2 neurons, got {n_neurons}")

    if stimulus_mask is None:
        stimulus_mask = np.zeros(n_frames, dtype=int)
        stimulus_names = {0: "all"}
    stimulus_mask = np.asarray(stimulus_mask)
    if stimulus_mask.shape != (n_frames,):
        raise ValueError(
            f"stimulus_mask must have one entry per frame ({n_frames}), "
            f"got {stimulus_mask.shape}"
        )

    codes = np.unique(stimulus_mask)
    if stimulus_names is None:
        names = {code: code for code in codes}
    elif isinstance(stimulus_names, dict):
        names = stimulus_names
    else:
        names = dict(zip(codes, stimulus_names))

    if neuron_ids is None:
        neuron_ids = np.arange(n_neurons)
    neuron_ids = np.asarray(neuron_ids)
    if len(neuron_ids) != n_neurons:
        raise ValueError(
            f"neuron_ids must have one entry per neuron ({n_neurons}), "
            f"got {len(neuron_ids)}"
        )

    pre_idx, post_idx = _pair_indices(n_neurons, symmetric)
    out = {
        id_columns[0]: neuron_ids[pre_idx],
        id_columns[1]: neuron_ids[post_idx],
    }

    for code in codes:
        frames = stimulus_mask == code
        n = int(frames.sum())
        name = names.get(code, code)
        if n < min_frames:
            print(f"  skipping {name}: {n} frames < min_frames={min_frames}")
            continue
        similarity = metric(traces[:, frames])
        similarity = np.asarray(similarity)
        if similarity.shape != (n_neurons, n_neurons):
            raise ValueError(
                f"metric returned {similarity.shape}, expected "
                f"({n_neurons}, {n_neurons}) — it must map "
                f"(n_neurons, n_frames) to (n_neurons, n_neurons)"
            )
        out[name] = similarity[pre_idx, post_idx].astype(dtype)
        n_nan = int(np.isnan(out[name]).sum())
        print(f"  {str(name):<30s} {n:6d} frames"
              + (f"   ({n_nan:,} NaN pairs, e.g. from constant traces)" if n_nan else ""))

    return pd.DataFrame(out)
