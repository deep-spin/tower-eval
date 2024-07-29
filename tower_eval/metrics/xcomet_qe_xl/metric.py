# -*- coding: utf-8 -*-
from tower_eval.metrics.base.comet import QECOMET


class XCOMETQEXL(QECOMET):
    def __init__(self, **kwargs) -> None:
        super().__init__(model="Unbabel/XCOMET-XL", **kwargs)

    @staticmethod
    def metric_name():
        return "xcomet_qe_xl"
