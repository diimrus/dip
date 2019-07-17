import typing
import luigi
import json
from pathlib import Path
from hashlib import md5
from extensions import DirParametr, FileSystemParametr

class TokenizerPipe(luigi.Task):
    dataset = luigi.Parameter()
    output_dir: Path = DirParametr(default='./dist/tokens')

    def output(self):
        key = md5(Path(self.dataset).read_bytes()).hexdigest()
        path = Path(self.output_dir)
        path.mkdir(exist_ok=True)
        return luigi.LocalTarget(str(path / f'tokens-for-{key}.json'))

    def run(self):
        from keras.preprocessing.text import text_to_word_sequence
        import pandas as pd


        data = pd.read_excel(self.dataset)
        data = data[data.columns[0]]
        datasets = data.tolist()

        tokens = [text_to_word_sequence(text) for text in datasets]
        with self.output().open('w') as f:
            json.dump(tokens, f)



class TokenizerWithPandasTask(luigi.Task):

    dataset = FileSystemParametr()
    output_dir: Path = DirParametr(default='./dist/tokens-pandas')
    tokenize_column = luigi.Parameter(default='вопрос')

    def output(self):
        key = md5(Path(self.dataset).read_bytes()).hexdigest()
        path = Path(self.output_dir)
        path.mkdir(exist_ok=True)
        return luigi.LocalTarget(str(path / f'tokens-for-{key}.pickle'))

    def run(self):
        from keras.preprocessing.text import text_to_word_sequence
        import pandas as pd


        data = pd.read_pickle(self.dataset)
        output = pd.DataFrame(columns=data.columns)
        for _, row in data.iterrows():
            text = row[str(self.tokenize_column)]
            row[str(self.tokenize_column)] = list(text_to_word_sequence(text))
            output = output.append(row, ignore_index=True)

        pd.to_pickle(output, self.output().path)


class DocToTokensPipe:

    def __init__(self, tokenizer: TokenizerPipe, seps=['.', '?', '!', '\n', '\r\n']):
        self.tokenizer = tokenizer
        self.seps = seps

    def split(self, seq:typing.Iterator) -> typing.Iterable[str]:
        out = ''
        for s in seq:
            if s in self.seps:
                if len(out) > 0:
                    yield out
                    out = ''
            else:
                out += s
        if len(out) > 0:
            return out

    def proccessing(self, text:str) -> typing.Iterator[typing.List[str]]:
        return ( self.tokenizer.proccessing(s) for s in self.split(text) )
        