import luigi
from extensions import FileSystemParametr
from pathlib import Path


class SimpleExcelDatasetTransformTask(luigi.Task):
    input_file = FileSystemParametr()
    output_dir = FileSystemParametr(default='./dist/dataset-transforms/')
    output_file_name = luigi.Parameter(default=None)

    def output(self):
        path = Path(self.output_dir)
        path.mkdir(exist_ok=True)
        input_file = Path(self.input_file)
        name = self.output_file_name if self.output_file_name and self.output_file_name != 'None' else input_file.stem
        return luigi.LocalTarget(str(path / f'{name}.pickle'))

    def run(self):
        import pandas as pd
        data_source = pd.read_excel(self.input_file) # sheet_name='Все'
        data_raw = pd.DataFrame(columns=['вопрос', 'ответ'])
        for c in data_source.columns:
            for _, item in data_source[c].iteritems():
                if pd.notnull(item):
                    data_raw = data_raw.append({
                        'вопрос':str(item),
                        'ответ': c
                    }, ignore_index=True)

        pd.to_pickle(data_raw, self.output().path)