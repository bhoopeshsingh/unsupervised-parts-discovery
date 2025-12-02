# metrics.py
# Evaluation metrics for unsupervised part discovery

def compute_nmi(pred_parts, true_parts):
    # TODO: call sklearn NMI
    pass

def compute_ari(pred_parts, true_parts):
    # TODO: call sklearn ARI
    pass

def compute_fg_iou(mask_pred, mask_gt):
    # TODO
    pass

def evaluate_all(config):
    """
    Orchestrates evaluation:
    - Loads model
    - Generates part assignments
    - Computes NMI / ARI / Kp / FG-mIoU etc.
    """
    # TODO run all evaluations
    return {}