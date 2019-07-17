from tokenizer import TokenizerPipe, DocToTokensPipe
from fasttext_pipe import FastTextPipe, ExportFromTrainDataTask
from vectorize import FastTextVectorizeTask
from preproccesors import SimpleExcelDatasetTransformTask
#from neural_network import MyNN
#from neural_network import MyNN2
from neural_network import MyNN3
#from neural_network import MyNN4
#from neural_network import MyNN5
#from neural_network import MyNN6
#from neural_network import MyNN7
#from neural_network import MyNN8
#from neural_network import MyNN9
#from neural_network import MyNN10
import pandas as pd
from pathlib import Path
import luigi

main_params = dict(
    vectorize_name='fasttext-model-5',
    # model_nasme='kn-main-4-1',
    datasets=[
        './datasets/train/data_05_03_19.xlsx',
        './datasets/train/data-main-Тей.xlsx',
        './datasets/train/data-main-Баранчук.xlsx',
        './datasets/train/data-main-Вепрев.xlsx',
        './datasets/train/data-main-Дарина-К.xlsx',
        './datasets/train/data-main-Денисов.xlsx',
        './datasets/train/data-main-Колесников.xlsx',
        './datasets/train/data-main-Матвейчук.xlsx',
        './datasets/train/data-main-Фаткулин.xlsx',
        './datasets/train/data-main-Шинкаренко.xlsx',
        './datasets/train/data-main-Рызыванов.xlsx',
        # './datasets/train/data-main-Татаринов.xlsx',
        # './datasets/train/data-main-Шицелов.xlsx',
        './datasets/train/Запись в школу.xlsx',
    ],
    order_class=["Загс", "Запись в школу", "Запись к врачу", "Очередь в детский сад", "Электронный дневник",  "Строительство дома", "остальные", "бензин", "спорт"],
    ovr_strategy=True,
    use_shuffle=False,
)


if __name__ == "__main__":
    luigi.build(
        [ 
            
           
            MyNN3(
                model_name='keras-ft3-main-v3',
                **main_params,
            ),
        ],
        local_scheduler=True
    )

    # fasttext_model_file = Path('./ft_model')
    # if not fasttext_model_file.exists():
    #     data = pd.concat([pd.read_excel('dump_-60237511.xlsx'), pd.read_excel('dump_-66710311.xlsx')])
    #     token_pipe = TokenizerPipe()
    #     doc_pipe = DocToTokensPipe(token_pipe)
    #     tokens = ( list( doc_pipe.proccessing(item) ) for item in iter(data.as_matrix()[:, 0]) )
    #     tokens = list(tokens)
        
    #     fasttext = FastTextPipe(size=20)
    #     flat_text = []
    #     for texts in tokens:
    #         for text in texts:
    #             if len(text) > 3:
    #                 flat_text.append(text)

    #     fasttext.fit(flat_text, epochs=50)
    #     fasttext.save(str(fasttext_model_file))
    # else:
    #     fastext = FastTextPipe.load(str(fasttext_model_file))
    #     pass