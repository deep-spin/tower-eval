# -*- coding: utf-8 -*-
from tower_eval.metrics.base.comet import QECOMET


class COMETKiwi23XXL(QECOMET):
    def __init__(self, **kwargs) -> None:
        super().__init__(model="Unbabel/wmt23-cometkiwi-da-xxl", **kwargs)

    @staticmethod
    def metric_name():
        return "comet_kiwi_23_xxl"
