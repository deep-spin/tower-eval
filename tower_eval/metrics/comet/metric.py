# -*- coding: utf-8 -*-
from tower_eval.metrics.base.comet import RefCOMET


class COMET(RefCOMET):
    def __init__(self, **kwargs) -> None:
        super().__init__(model="Unbabel/wmt22-comet-da", **kwargs)

    @staticmethod
    def metric_name():
        return "comet"
