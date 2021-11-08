from abc import abstractmethod
from collections import defaultdict
import pandas as pd
from smartpipeline.stage import Source, DataItem


class MyDataItem(DataItem):
    def __str__(self):
        content = '<none>'
        if 'utterance' in self.payload:
            content = str(self.payload["utterance"])
        else:
            content = self.payload["conversation"]
        return "{}\t{}".format(self._payload["id"], content)


class BaseFileSource(Source):
    def __init__(self, file_path):
        self._file_path = file_path
        dataframe = pd.read_csv(file_path, delimiter="\t", header=None)
        self.utt_dict = dict(zip(dataframe[0], dataframe[1]))
        self._iterator = self._iter_data()

    def pop(self):
        ret = next(self._iterator, None)
        if ret is not None:
            item = MyDataItem()
            item.payload.update(ret)
            return item
        else:
            self.stop()

    @abstractmethod
    def _iter_data(self):
        yield None


class UttFileSource(BaseFileSource):
    def _iter_data(self):
        for k, v in self.utt_dict.items():
            yield {'id': k, 'utterance': v}


class ConvFileSource(BaseFileSource):
    def _iter_data(self):
        conversations_dict = defaultdict(list)
        for k, v in self.utt_dict.items():
            new_k = k.split("_")[0]
            conversations_dict[new_k].append(v)
        for k, v in conversations_dict.items():
            yield {'id': k, 'conversation': v}