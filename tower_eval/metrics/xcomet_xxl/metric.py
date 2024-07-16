# -*- coding: utf-8 -*-
from tower_eval.metrics.base.comet import RefCOMET


class XCOMETXXL(RefCOMET):
    def __init__(self, **kwargs) -> None:
        super().__init__(model="Unbabel/XCOMET-XXL", **kwargs)

    @staticmethod
    def metric_name():
        return "xcomet_xxl"
