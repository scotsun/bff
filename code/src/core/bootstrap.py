import numpy as np
from tqdm import trange
from joblib import Parallel, delayed
from sksurv.metrics import cumulative_dynamic_auc
from sklearn.metrics import roc_auc_score, average_precision_score


class AUCBootstrapping:
    def __init__(self, y, yh):
        self.y = y
        self.yh = yh
        self.N = len(y)

    
    def bootstrap(self, B: int) -> None:
        self.boot_aucs: np.ndarray = np.array([np.nan] * B)
        self.boot_prs: np.ndarray = np.array([np.nan] * B)

        def _collect(i: int):
            boot_idx = np.random.choice(self.N, self.N, replace=True)
            boot_y, boot_yh = self.y[boot_idx], self.yh[boot_idx]
            self.boot_aucs[i] = roc_auc_score(boot_y, boot_yh)
            self.boot_prs[i] = average_precision_score(boot_y, boot_yh)
        
        Parallel(n_jobs=4, require="sharedmem")(delayed(_collect)(i) for i in trange(B))
        
    def auc_ci(self, alpha: float = 0.05) -> tuple[np.ndarray]:
        _alpha = 100 * alpha
        return (np.percentile(self.boot_aucs, q=(_alpha / 2, 100 - _alpha / 2), axis=0).round(3), 
                np.percentile(self.boot_prs, q=(_alpha / 2, 100 - _alpha / 2), axis=0).round(3))




class CDAUCBootstrapping:
    def __init__(self, survival_train, survival_eval, estimates, times) -> None:
        self.survival_train = survival_train
        self.survival_eval = survival_eval  # shuffle this
        self.estimates = estimates  # shuffle this
        self.times = times
        self.N = len(survival_eval)

    def bootstrap(self, B: int) -> None:
        self.boot_aucs: np.ndarray = np.array([np.nan] * B)
        t =  self.estimates.shape[1]
        self.boot_aucts: np.ndarray = np.array([[np.nan] * t] * B)

        def _collect(i: int):
            boot_idx = np.random.choice(self.N, self.N, replace=True)
            _auc_t, _auc_integrated = cumulative_dynamic_auc(
                self.survival_train,
                self.survival_eval[boot_idx],
                self.estimates[boot_idx],
                self.times,
            )
            self.boot_aucs[i] = _auc_integrated
            self.boot_aucts[i] = _auc_t

        Parallel(n_jobs=4, require="sharedmem")(delayed(_collect)(i) for i in trange(B))

    def auc_ci(self, alpha: float = 0.05) -> tuple[np.ndarray]:
        _alpha = 100 * alpha
        return (np.percentile(self.boot_aucts, q=(_alpha / 2, 100 - _alpha / 2), axis=0).round(3), 
                np.percentile(self.boot_aucs, q=(_alpha / 2, 100 - _alpha / 2), axis=0).round(3))
