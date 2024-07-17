# -*- coding: utf-8 -*-
from tower_eval.metrics.base.metricx import RefMetricX


class MetricXLarge(RefMetricX):
    def __init__(self, **kwargs) -> None:
        super().__init__(
            model="google/metricx-23-large-v2p0", tokenizer="google/mt5-large", **kwargs
        )

    @staticmethod
    def metric_name():
        return "metricx_large"
