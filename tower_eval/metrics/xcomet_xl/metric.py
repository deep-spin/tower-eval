# -*- coding: utf-8 -*-
from tower_eval.metrics.base.comet import RefCOMET


class XCOMETXL(RefCOMET):
    def __init__(self, **kwargs) -> None:
        super().__init__(model="Unbabel/XCOMET-XL", **kwargs)

    @staticmethod
    def metric_name():
        return "xcomet_xl"
