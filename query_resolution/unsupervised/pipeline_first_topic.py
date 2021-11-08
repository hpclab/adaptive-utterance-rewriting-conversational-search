from smartpipeline.pipeline import Pipeline
from smartpipeline.stage import Stage, DataItem
import logging
from query_resolution.unsupervised.source import UttFileSource
from query_resolution.unsupervised.topic import FirstTopic


class PrintItem(Stage):
    def process(self, item: DataItem) -> DataItem:
        print(item)
        return item

pipeline = Pipeline().set_source(
        UttFileSource('../../data/datasets/test_set.tsv')
    ).append_stage(
        "print item",
        PrintItem()
    ).append_stage(
        "first topic",
        FirstTopic()
    ).append_stage(
        "print rewritten",
        PrintItem()
)

for item in pipeline.run():
    logging.info(f'Processed document: {item}')
