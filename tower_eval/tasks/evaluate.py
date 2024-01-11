from tower_eval.metrics import available_metrics


def run_metric(
    metric_name: str,
    eval_args: dict,
):
    """ """
    metric = available_metrics[metric_name](**(eval_args))
    metric_score = metric.run()

    return metric_score
