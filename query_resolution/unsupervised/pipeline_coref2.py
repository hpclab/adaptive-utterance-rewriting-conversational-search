from smartpipeline.pipeline import Pipeline
from smartpipeline.stage import Stage, DataItem
import logging
from query_resolution.unsupervised.coreference import NeuralCoref
from query_resolution.unsupervised.source import ConvFileSource


class PrintItem(Stage):
    def process(self, item: DataItem) -> DataItem:
        print(item)
        return item

pipeline = Pipeline().set_source(
        ConvFileSource('../../data/datasets/test_set.tsv')
    ).append_stage(
        "print item",
        PrintItem()
    ).append_stage(
        "neural co-referencing",
        NeuralCoref()
    ).append_stage(
        "print rewritten",
        PrintItem()
)


for item in pipeline.run():
    logging.info(f'Processed document: {item}')