"""Pure sparse-cross-term operations; no study data or stopping-time access."""
import numpy as np


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f'{name} must be an integer')
    return int(value)


def pair_indices(active):
    """Return lexicographic active pairs as a (P, 2) integer array.

    ``active`` may be a one-dimensional boolean mask or unique nonnegative
    integer indices. Each returned pair contains original mode indices i < j.
    """
    active = np.asarray(active)
    if active.ndim != 1:
        raise ValueError('active must be one-dimensional')
    if active.dtype.kind == 'b':
        indices = np.flatnonzero(active)
    elif not active.size:
        indices = np.empty(0, dtype=int)
    elif active.dtype.kind in 'iu':
        indices = np.sort(active.astype(int))
        if np.any(indices < 0) or len(np.unique(indices)) != len(indices):
            raise ValueError('active indices must be unique and nonnegative')
    else:
        raise TypeError('active must contain booleans or integer indices')
    left, right = np.triu_indices(len(indices), k=1)
    return np.column_stack((indices[left], indices[right])).astype(int, copy=False)


def _pairs_array(pairs, width):
    pairs = np.asarray(pairs)
    if pairs.ndim != 2 or pairs.shape[1] != 2 or pairs.dtype.kind not in 'iu':
        raise ValueError('pairs must be a (P, 2) integer array')
    if pairs.size and (np.any(pairs < 0) or np.any(pairs >= width)
                       or np.any(pairs[:, 0] >= pairs[:, 1])):
        raise ValueError('pairs must satisfy 0 <= i < j < width')
    return pairs.astype(int, copy=False)


def rank_pairs(h, f_score, df_score, score_times, f_end, pairs):
    """Return (change_scores, magnitude_scores), each aligned to ``pairs``.

    change[p] = max_t |2 h_ij (f_i'(t) f_j(t) + f_i(t) f_j'(t)) (t+1)|
    magnitude[p] = |2 h_ij f_i(T) f_j(T)|.

    ``f_score`` and ``df_score`` have shape (number_of_times, width),
    ``score_times`` has shape (number_of_times,), and ``f_end`` has shape
    (width,). Derivatives must use the SAME time coordinate as score_times.
    The supplied h already includes any output-channel averaging. No minimum,
    reference-risk curve, noise level, or outcome enters this function.
    """
    h = np.asarray(h, dtype=float)
    f_score = np.asarray(f_score, dtype=float)
    df_score = np.asarray(df_score, dtype=float)
    score_times = np.asarray(score_times, dtype=float)
    f_end = np.asarray(f_end, dtype=float)
    if h.ndim != 2 or h.shape[0] != h.shape[1]:
        raise ValueError('h must be square')
    width = h.shape[0]
    if (score_times.ndim != 1 or not score_times.size
            or f_score.shape != (len(score_times), width)
            or df_score.shape != f_score.shape or f_end.shape != (width,)):
        raise ValueError('score arrays have incompatible shapes')
    if np.any(score_times < 0):
        raise ValueError('score times must be nonnegative')
    if not all(np.all(np.isfinite(a)) for a in (h, f_score, df_score, score_times, f_end)):
        raise ValueError('score inputs must be finite')
    pairs = _pairs_array(pairs, width)
    change = np.empty(len(pairs), dtype=float)
    magnitude = np.empty(len(pairs), dtype=float)
    # Bound the temporary arrays even when the active rank reaches 129.
    for begin in range(0, len(pairs), 256):
        end = min(begin + 256, len(pairs))
        ii, jj = pairs[begin:end].T
        coefficient = 2 * h[ii, jj]
        derivative = coefficient[None, :] * (
            df_score[:, ii] * f_score[:, jj]
            + f_score[:, ii] * df_score[:, jj]
        )
        change[begin:end] = np.max(np.abs(derivative * (score_times[:, None] + 1)), axis=0)
        magnitude[begin:end] = np.abs(coefficient * f_end[ii] * f_end[jj])
    if not np.all(np.isfinite(change)) or not np.all(np.isfinite(magnitude)):
        raise FloatingPointError('nonfinite pair score')
    return change, magnitude


def select_top(scores, K):
    """Return K pair-position indices, decreasing score with stable input ties."""
    scores = np.asarray(scores, dtype=float)
    K = _integer(K, 'K')
    if scores.ndim != 1 or not np.all(np.isfinite(scores)):
        raise ValueError('scores must be a finite one-dimensional array')
    if K < 0 or K > len(scores):
        raise ValueError('K must be between zero and the number of pairs')
    return np.argsort(-scores, kind='stable')[:K]


def masked_h(h, pairs, selected):
    """Copy h's diagonal and selected symmetric pairs; zero every other entry.

    ``selected`` contains positions in the pairs array, as returned by
    select_top. Both original h entries of a selected pair are preserved;
    symmetric h therefore stays symmetric without silently changing values.
    """
    h = np.asarray(h)
    if h.ndim != 2 or h.shape[0] != h.shape[1]:
        raise ValueError('h must be square')
    pairs = _pairs_array(pairs, h.shape[0])
    selected = np.asarray(selected)
    if selected.ndim != 1 or (selected.size and selected.dtype.kind not in 'iu'):
        raise ValueError('selected must be one-dimensional pair-position indices')
    selected = selected.astype(int, copy=False)
    if selected.size and (np.any(selected < 0) or np.any(selected >= len(pairs))):
        raise ValueError('selected pair position is out of range')
    result = np.diag(np.diag(h)).copy()
    if selected.size:
        ii, jj = pairs[selected].T
        result[ii, jj] = h[ii, jj]
        result[jj, ii] = h[jj, ii]
    return result


def budget_size(rank, percent_int):
    """Strict floor of an integer percentage of rank*(rank-1)/2 pairs."""
    rank = _integer(rank, 'rank')
    percent_int = _integer(percent_int, 'percent_int')
    if rank < 0 or not 0 <= percent_int <= 100:
        raise ValueError('rank must be nonnegative and percentage between 0 and 100')
    return (rank * (rank - 1) // 2) * percent_int // 100
