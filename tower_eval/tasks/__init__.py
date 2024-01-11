from tower_eval.metrics.bleu.metric import BLEU
from tower_eval.metrics.comet.metric import COMET

__all__ = [BLEU, COMET]


available_metrics = {metric.metric_name(): metric for metric in __all__}
