# -*- coding: utf-8 -*-
from tower_eval.metrics.base.metricx import QEMetricX_24, RefMetricX_24


class MetricX_24_Large(RefMetricX_24):
    def __init__(self, **kwargs) -> None:
        super().__init__(
            model="google/metricx-24-hybrid-large-v2p6",
            tokenizer="google/mt5-xl",
            **kwargs
        )

    @staticmethod
    def metric_name():
        return "metricx_24_large"


class MetricX_24_XL(RefMetricX_24):
    def __init__(self, **kwargs) -> None:
        super().__init__(
            model="google/metricx-24-hybrid-xl-v2p6",
            tokenizer="google/mt5-xl",
            **kwargs
        )

    @staticmethod
    def metric_name():
        return "metricx_24_xl"


class MetricX_24_XXL(RefMetricX_24):
    def __init__(self, **kwargs) -> None:
        super().__init__(
            model="google/metricx-24-hybrid-xxl-v2p6",
            tokenizer="google/mt5-xl",
            **kwargs
        )

    @staticmethod
    def metric_name():
        return "metricx_24_xxl"


### QE ###


class MetricX_24_QE_Large(QEMetricX_24):
    def __init__(self, **kwargs) -> None:
        super().__init__(
            model="google/metricx-24-hybrid-large-v2p6",
            tokenizer="google/mt5-xl",
            **kwargs
        )

    @staticmethod
    def metric_name():
        return "metricx_24_qe_large"


class MetricX_24_QE_XL(QEMetricX_24):
    def __init__(self, **kwargs) -> None:
        super().__init__(
            model="google/metricx-24-hybrid-xl-v2p6",
            tokenizer="google/mt5-xl",
            **kwargs
        )

    @staticmethod
    def metric_name():
        return "metricx_24_qe_xl"


class MetricX_24_QE_XXL(QEMetricX_24):
    def __init__(self, **kwargs) -> None:
        super().__init__(
            model="google/metricx-24-hybrid-xxl-v2p6",
            tokenizer="google/mt5-xl",
            **kwargs
        )

    @staticmethod
    def metric_name():
        return "metricx_24_qe_xxl"
