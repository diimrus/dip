import luigi
from gensim.models import FastText
from pathlib import Path
from hashlib import md5

class FastTextVectorizeTask(luigi.Task):
    datasets = luigi.ListParameter()
    output_dir = luigi.Parameter(default='./dist/fasttext-vectorize')
    max_len = luigi.IntParameter(default=-1)
    fastext_model = luigi.Parameter(default=None)
    name = luigi.Parameter(default=None)

    def requires(self):
        from fasttext_pipe import FastTextPipe
        return FastTextPipe(name=self.fastext_model)

    def output(self):
        path = Path(self.output_dir)
        path.mkdir(exist_ok=True)
        if self.name is None or self.name == 'None':
            name = self.fastext_model
            key = ''
            for dataset in self.datasets:
                key += md5(Path(dataset).read_bytes()).hexdigest()
            key = md5(key.encode()).hexdigest()
            return luigi.LocalTarget(str(path / f'{name}-{key}.pickle'))
        else:
            return luigi.LocalTarget(str(path / f'{self.name}.pickle'))

    def run(self):
        import numpy as np
        import pandas as pd
        # from keras.preprocessing.sequence import pad_sequences
        from tokenizer import TokenizerWithPandasTask
        import json
        import pickle

        max_len = self.max_len if self.max_len > 0 else None
        dataset = pd.DataFrame(columns=['вопрос', 'ответ'])
        self.load_word2vec_model()
        for fdataset in self.datasets:
            ftokens = yield TokenizerWithPandasTask(fdataset)
            tokens = pd.read_pickle(ftokens.path)
            dataset = dataset.append(tokens, ignore_index=True)

        sequences = pd.DataFrame.from_records( {'вопрос': np.array(list(self.text_to_sequinces(text['вопрос']))), 'ответ': text['ответ']} for _, text in dataset.iterrows())
        if max_len is None:
            max_len = int(round(sequences['вопрос'].apply(lambda x: len(x)).quantile(0.95)))

        output_seq = pd.DataFrame(columns=['вопрос', 'ответ'])
        for _, row in sequences.iterrows():
            seq, ans = row['вопрос'], row['ответ']
            l = len(seq)
            if l > max_len:
                output_seq = output_seq.append({
                    'вопрос': np.array([ *seq.tolist()[:max_len - 1], sum(seq[max_len:]) / (l - max_len) ]),
                    'ответ': ans
                    }, ignore_index=True)
            
            elif l == 0: continue
            
            elif l < max_len:
                zero = np.zeros((seq[0].shape)).astype('float64')
                output_seq = output_seq.append({
                    'вопрос': np.array([*seq.astype('float64').tolist(), *([zero]*(max_len - l))]),
                    'ответ': ans
                    }, ignore_index=True)
            else:
                output_seq = output_seq.append({
                    'вопрос': np.array(seq),
                    'ответ': ans
                    }, ignore_index=True)

        pd.to_pickle(output_seq, self.output().path)

    def load_word2vec_model(self):
        self.word2vec = FastText.load(self.input().path)
        return self.word2vec

    def text_to_sequinces(self, tokens):
        for token in tokens:
            try:
                yield self.word2vec[token]
            except KeyError:
                continue
