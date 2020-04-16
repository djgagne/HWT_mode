import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss


def brier_skill_score(y_true, y_pred):
    bs_climo = np.mean((y_true.mean() - y_true) ** 2)
    bs = brier_score_loss(y_true, y_pred)
    return 1 - bs / bs_climo


def classifier_metrics(y_true, model_predictions):
    """

    Args:
        self:
        y_true:
        model_predictions:
        out_path:

    Returns:

    """
    metric_names = ["AUC", "Brier_Score", "Brier_Skill_Score"]
    metric_funcs = {"AUC": roc_auc_score,
                    "Brier_Score": brier_score_loss,
                    "Brier_Skill_Score": brier_skill_score
                    }
    metrics = pd.DataFrame(0, index=model_predictions.columns, columns=metric_names)
    for metric in metric_names:
        for model_name in model_predictions.columns:
            metrics.loc[model_name, metric] = metric_funcs[metric](y_true,
                                                                   model_predictions[model_name].values)
    return metrics