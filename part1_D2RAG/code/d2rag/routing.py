from typing import Optional, Sequence
import numpy as np

def select_threshold(model, X_val: np.ndarray, y_val: Sequence[int]) -> float:
    from sklearn.metrics import precision_recall_fscore_support
    scores = model.predict_proba(X_val)[:, 1]
    (best_f1, best_t) = (-1.0, 0.0)
    for t in np.arange(0.01, 1.0, 0.01):
        y_pred = (scores >= t).astype(int)
        f1 = precision_recall_fscore_support(y_val, y_pred, average='binary', pos_label=0, zero_division=0)[2]
        if f1 > best_f1:
            (best_f1, best_t) = (f1, float(t))
    return best_t

def needs_retrieval(score_no_retrieval: float, threshold: float) -> int:
    return int(score_no_retrieval < threshold)

def choose_strategy(needs_ret: int, d_seper: Optional[float], harmful: float, useful: float) -> str:
    if not needs_ret:
        return 'noctx-normal'
    if d_seper is None:
        return 'ctx-normal'
    if d_seper < harmful:
        return 'ctx-neg-acd'
    if d_seper > useful:
        return 'ctx-pos-acd'
    return 'ctx-normal'
