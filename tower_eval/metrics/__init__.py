from tower_eval.metrics.accuracy.metric import ACCURACY
from tower_eval.metrics.bleu.metric import BLEU
from tower_eval.metrics.bleurt.metric import BLEURT
from tower_eval.metrics.chrf.metric import CHRF
from tower_eval.metrics.comet.metric import COMET
from tower_eval.metrics.comet_kiwi.metric import COMETKiwi
from tower_eval.metrics.errant.metric import ERRANT
from tower_eval.metrics.error_span_detection_f1.metric import ErrorSpanDetectionF1
from tower_eval.metrics.error_span_detection_precision.metric import (
    ErrorSpanDetectionPrecision,
)
from tower_eval.metrics.error_span_detection_recall.metric import (
    ErrorSpanDetectionRecall,
)
from tower_eval.metrics.f1.metric import F1
from tower_eval.metrics.f1_sequence.metric import F1SEQUENCE
from tower_eval.metrics.pearson.metric import PEARSON
from tower_eval.metrics.perplexity.metric import Perplexity
from tower_eval.metrics.spearman.metric import SPEARMAN
from tower_eval.metrics.ter.metric import TER
from tower_eval.metrics.tydiqa_exact_match.metric import TYDIQAExactMatch
from tower_eval.metrics.tydiqa_f1.metric import TYDIQAF1
from tower_eval.metrics.xcomet.metric import XCOMET

__all__ = [
    TER,
    BLEU,
    XCOMET,
    COMET,
    COMETKiwi,
    BLEURT,
    CHRF,
    ERRANT,
    F1,
    F1SEQUENCE,
    ACCURACY,
    PEARSON,
    SPEARMAN,
    ErrorSpanDetectionF1,
    ErrorSpanDetectionRecall,
    ErrorSpanDetectionPrecision,
    TYDIQAF1,
    TYDIQAExactMatch,
    Perplexity,
]


available_metrics = {metric.metric_name(): metric for metric in __all__}
