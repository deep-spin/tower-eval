# -*- coding: utf-8 -*-
from tower_eval.metrics.base.comet import QECOMET


class XCOMETQEXXL(QECOMET):
    def __init__(self, **kwargs) -> None:
        super().__init__(model="Unbabel/XCOMET-XXL", **kwargs)

    @staticmethod
    def metric_name():
        return "xcomet_qe_xxl"
