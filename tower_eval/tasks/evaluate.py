from tower_eval.metrics.base.metrics_handler import Metric


def run_instantiated_metric(
    metric: Metric,
    hypothesis_path,
    gold_data_path,
    eval_args: dict,
):
    """ """
    metric_score = metric.run(
        hypothesis_path=hypothesis_path,
        gold_data_path=gold_data_path,
        **eval_args,
    )

    return metric_score
