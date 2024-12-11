# -*- coding: utf-8 -*-
from loguru import logger

from tower_eval.metrics.base.xml_metric import XMLMetric
from tower_eval.metrics.chrf.metric import CHRF


class XML_CHRF(XMLMetric):
    def __init__(self, **kwargs) -> None:
        super().__init__(CHRF, **kwargs)

    @staticmethod
    def metric_name():
        return "xml_chrf"
