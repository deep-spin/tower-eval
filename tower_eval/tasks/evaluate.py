from tower_eval.metrics.metrics_handler import Metric


def run_instantiated_metric(
    metric: Metric,
    hypothesis_path,
    gold_data_path,
):
    """ """
    metric_score = metric.run(
        hypothesis_path=hypothesis_path, gold_data_path=gold_data_path
    )

    return metric_score
