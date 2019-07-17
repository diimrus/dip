import typing
import luigi
from functools import reduce
from pathlib import Path
from tokenizer import TokenizerPipe
from hashlib import md5
import json

class FastTextPipe(luigi.Task):

    datasets = luigi.ListParameter(default=[])
    epochs = luigi.Parameter(default=10)
    size = luigi.Parameter(default=100)
    name = luigi.Parameter(default=None)
    output_dir = luigi.Parameter(default='./dist/fasttext')
    dataset_dir = luigi.Parameter(default='./datasets/fasttext')

    def requires(self):
        datasets = self.datasets if self.datasets else [ str(file) for file in Path(self.dataset_dir).iterdir() if file.suffix == '.xlsx']
        return [ TokenizerPipe(dataset)  for dataset in datasets] 

    def output(self):
        cur_dir = Path(self.output_dir)
        cur_dir.mkdir(exist_ok=True)
        if self.name is None:
            key = ''
            for dataset in self.datasets:
                key += md5(Path(dataset).read_bytes()).hexdigest()
            key = md5(key.encode()).hexdigest()
            output_file = luigi.LocalTarget(str(cur_dir / f'fasttext-model-hash.{key}-epochs.{self.epochs}-size.{self.size}.bin'))
        else:
            output_file = luigi.LocalTarget(str(cur_dir / f'{self.name}.bin'))

        return output_file

    def run(self):
        from gensim.models import FastText

        tokens = []
        for dataset in self.input():
            with dataset.open() as f:
                buf = json.load(f)
                tokens.extend(buf)
        if tokens:
            len_tokens = reduce(lambda a, b: a + len(b), tokens, 0)
            model = FastText(size=self.size)
            model.build_vocab(tokens)
            model.train(tokens, epochs=self.epochs, total_words=len_tokens)
            model.save(self.output().path)
        

    # @classmethod
    # def load(cls, path):
    #     _cls = cls()
    #     _cls.model = FastText.load(path)
    #     return _cls

    # def save(self, path):
    #     return self.model.save(path)

    # def fit(self, texts: typing.List[typing.List[str]], *args, **kwargs):
    #     def_kw = {
    #         'epochs': 10
    #     }
    #     def_kw.update(kwargs)
    #     self.model.build_vocab(texts)
    #     self.model.train(texts, *args, total_examples=len(texts), **def_kw)


class ExportFromTrainDataTask(luigi.Task):
    datasets = luigi.ListParameter(default=[])
    search_dir = luigi.Parameter(default='./datasets/train')
    output_dir = luigi.Parameter(default='./datasets/fasttext')
    name = luigi.Parameter(default=None)

    def output(self):
        cur_dir = Path(self.output_dir)
        cur_dir.mkdir(exist_ok=True)
        if self.name is None:
            key = ''
            for dataset in self.datasets:
                key += md5(Path(dataset).read_bytes()).hexdigest()
            key = md5(key.encode()).hexdigest()
            output_file = luigi.LocalTarget(str(cur_dir / f'export-excel-hash.{key}.xlsx'))
        else:
            output_file = luigi.LocalTarget(str(cur_dir / f'{self.name}.xlsx'))

        return output_file


    def run(self):
        import pandas as pd

        datasets = list(self.datasets)
        if len(datasets) == 0:
            datasets = [f for f in Path(self.search_dir).iterdir() if f.suffix == '.xlsx' and f.name[0] != '~']
        data = pd.Series()
        
        for file in datasets:
            fbuf = pd.read_excel(Path(file))
            cols = list(fbuf.columns)
            for col in cols:
                data = data.append(fbuf[col], ignore_index=True)
        
        data = data[ data.isna() == False ]
        # data = data.reset_index()
        data.to_excel(str(Path(self.output().path)))


if __name__ == "__main__":
    pass