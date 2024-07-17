# -*- coding: utf-8 -*-
from tower_eval.metrics.base.metricx import QEMetricX


class MetricXQE(QEMetricX):
    def __init__(self, **kwargs) -> None:
        super().__init__(
            model="google/metricx-23-qe-xl-v2p0", tokenizer="google/mt5-xl", **kwargs
        )

    @staticmethod
    def metric_name():
        return "metricx_qe"
