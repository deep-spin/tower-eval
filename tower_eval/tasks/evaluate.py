from tower_eval.metrics.metrics_handler import Metric


def run_instantiated_metric(
    metric,
    hypothesis_path,
    gold_data_path,
    **kwargs
):
    """ """
    metric_score = metric.run(
        hypothesis_path=hypothesis_path, gold_data_path=gold_data_path,
        **kwargs)

    return metric_score
