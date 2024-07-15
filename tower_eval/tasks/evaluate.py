from tower_eval.metrics.metrics_handler import Metric


def run_instantiated_metric(
    metric,
    **kwargs
):
    """ """
    metric_score = metric.run(
        **kwargs)

    return metric_score
