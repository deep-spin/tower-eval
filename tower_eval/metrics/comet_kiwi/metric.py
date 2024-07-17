# -*- coding: utf-8 -*-
from tower_eval.metrics.base.comet import QECOMET


class COMETKiwi(QECOMET):
    def __init__(self, **kwargs) -> None:
        super().__init__(model="Unbabel/wmt22-cometkiwi-da", **kwargs)

    @staticmethod
    def metric_name():
        return "comet_kiwi"
