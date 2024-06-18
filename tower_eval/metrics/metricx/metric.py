# -*- coding: utf-8 -*-
import tempfile

import torch
from datasets import Dataset
from loguru import logger
from metricx23 import models
from tower_eval.metrics.metrics_handler import Metric
from tower_eval.metrics.metricx.result import MetricXResult
from tower_eval.metrics.result_handler import MetricResult
from transformers import AutoTokenizer, Trainer, TrainingArguments

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

class MetricX(Metric):
    def __init__(
        self,
        lowercase: bool = False,
        tokenizer="google/mt5-xl",
        model="google/metricx-23-xl-v2p0",
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.lowercase = kwargs.get("lowercase", lowercase)
        self.model = kwargs.get("model", model)
        self.tokenizer = kwargs.get("tokenizer", tokenizer)

        self.model = models.MT5ForRegression.from_pretrained(self.model)
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer)
        self.model.to(DEVICE)
        self.model.eval()

    @staticmethod
    def metric_name():
        return "metricx"

    def run(self) -> dict:
        hypotheses, gold_data = self._handle_inputs(
            self.hypothesis_path, self.gold_data_path
        )
        references = gold_data["ref"]
        result = self.evaluate(hypotheses, references)
        result.print_result(self.metric_name())
        return result.format_result(self.metric_name())

    def process_result(self, result) -> MetricResult:
        pass

    def evaluate(self, hypotheses: list, references: list) -> MetricXResult:
        """
        Evaluate function receives the hypotheses and the references and returns a COMETResult object.

        :param hypotheses: List of the MT outputs (sentences).
        :param references: List of the reference sentences.
        """
        
        def _make_input(example):
            example["input"] = ("candidate: " + example["hypothesis"] + " reference: " + example["reference"])
            return example

        def _tokenize(example):
            return self.tokenizer(example["input"], max_length=1024,  truncation=True, padding=False)

        def _remove_eos(example):
            example["input_ids"] = example["input_ids"][:-1]
            example["attention_mask"] = example["attention_mask"][:-1]
            return example
    
        samples = [{"hypothesis": h, "reference": r} for h, r in zip(hypotheses, references)]
        ds = Dataset.from_list(samples)
        ds = ds.map(_make_input)
        ds = ds.map(_tokenize)
        ds = ds.map(_remove_eos)
        ds.set_format(
            type="torch",
            columns=["input_ids", "attention_mask"],
            device=DEVICE,
            output_all_columns=True,
        )
        output_dir = tempfile.mkdtemp()

        logger.info( "Created temporary folder to store MetricX predictions: {}.".format(output_dir))
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_eval_batch_size=1,
            dataloader_pin_memory=False,
        )
        trainer = Trainer(
            model=self.model,
            args=training_args,
        )
        predictions, _, _ = trainer.predict(test_dataset=ds)
        predictions = predictions.tolist()
        metricx_result = MetricXResult(
            {
                "system_score": sum(predictions)/len(predictions),
                "segments_scores": predictions,
            }
        )
        return metricx_result