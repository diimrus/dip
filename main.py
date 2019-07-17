from tokenizer import TokenizerPipe, DocToTokensPipe
from fasttext_pipe import FastTextPipe, ExportFromTrainDataTask
from vectorize import FastTextVectorizeTask
from preproccesors import SimpleExcelDatasetTransformTask
from neural_network import SVMClassificatorTask, KNeighborsClassificatorTask, AdaBoostClassificatorTask, RandomForestClassificatorTask, NeuralNetworkClassificatorTask
import pandas as pd
from pathlib import Path
import luigi

main_params = dict(
    vectorize_name='fasttext-model-3',
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
            # ExportFromTrainDataTask(name='export-data-train'),

            # datasets=['dump_-60237511.xlsx', 'dump_-66710311.xlsx', 'export-data-train.xlsx']
            # FastTextPipe(name='fasttext-model-3', size=5, epochs=50),
            # SimpleExcelDatasetTransformTask(input_file='datasets/data_new_cat_sport.xlsx'),
            # FastTextVectorizeTask(datasets=['dist/dataset-transforms/data_new_cat_sport.pickle'], name='fasttext-vec', fastext_model='fasttext-model-1'),
            # SVMClassificatorTask(vectorize_name='fasttext-model-2', model_name='svm-ft-1'),
            # KNeighborsClassificatorTask(vectorize_name='fasttext-model-2', model_name='kn-ft-1'),
            # AdaBoostClassificatorTask(vectorize_name='fasttext-model-2', model_name='ada-ft-1'),
            # RandomForestClassificatorTask(vectorize_name='fasttext-model-2', model_name='rf-ft-1'),
           
            KNeighborsClassificatorTask(
                model_name='kn-ft3-main-4-1-ovr',
                **main_params,
            ),
            RandomForestClassificatorTask(
                model_name='rf-ft4-main-4-1-ovr-20',
                n_estimators=20,
                **main_params,
            ),
            NeuralNetworkClassificatorTask(
                hidden_layer_sizes=[20,10,5],
                activation='tanh',
                model_name='nn-ft3-main-4-1-(20-10-5)-ovr-tanh',
                **main_params,
            ),
            AdaBoostClassificatorTask(
                model_name='ada-ft3-main-4-1-ovr',
                **main_params,
            ),
            RandomForestClassificatorTask(
                vectorize_name='fasttext-model-3',
                ovr_strategy=True,
                model_name='rf-ft3-submodel-cat-5-ovr', 
                datasets=['./datasets/train/data-cat-1.xlsx'],
                order_class=[
                    "Что такое и зачем нужен электронный дневник",
                    "Безопасен ли электронный дневник, могут ли его взломать?",
                    "Нужен ли теперь бумажный дневник?",
                    "Что такое средний бал?",
                    "Что такое электронный журнал?",
                    "Что такое рейтинг учащихся?",
                    "Каким образом проверить оценки?",
                    "Порядок выставления оценок в электронный дневник",
                    "Можно ли посмотреть домашнее задание в электронном дневнике?",
                    "Можно ли связаться с учителем через электронный дневник?"
                ]
            ),
            RandomForestClassificatorTask(
                vectorize_name='fasttext-model-3',
                ovr_strategy=True,
                model_name='rf-ft3-submodel-cat-3-ovr', 
                datasets=['./datasets/train/data-cat-2.xlsx', './datasets/train/data-cat-3.xlsx'],
                order_class=[
                    "Как вызвать врача на дом?",
                    "Можно ли записаться по телефону?",
                    "Проверка записи на прием, изменение времени, отмена приема",
                    "Вопросы создания личного кабинета?",
                    "информация об услугах",
                    "Вопросы отображения медицинской организации",
                    "как записаться в другое МО",
                    "Отсутствуют талоны",
                    "запись другого человека",
                    "Запись ребенка",
                    "Отображение и отмена талонов",
                    "отображение данных во вкладке профиль",
                    "Определение участка"

                ]
            ),
            RandomForestClassificatorTask(
                vectorize_name='fasttext-model-3', 
                ovr_strategy=True,
                model_name='rf-ft3-submodel-cat-4-ovr', 
                datasets=['./datasets/train/data-cat-4.xlsx'],
                order_class=[
                    "Возраст для разных категорий",
                    "Порядок движения очереди в дет сад",
                    "Льготные категории",
                    "Порядок зачисления в сад",
                    "Информация о садиках",
                    "Нет мест в желаемом садике"
                ]
            ),
            RandomForestClassificatorTask(
                vectorize_name='fasttext-model-3', 
                ovr_strategy=True,
                model_name='rf-ft3-submodel-cat-1-ovr', 
                datasets=['./datasets/train/data-cat-5.xlsx'],
                order_class=[
                    "Ускорение процедуры бракосочетания и сроки подачи заявления",
                    "Один из брачующихся иностранный гражданин",
                    "Вопросы выездной регистрации",
                    "Смена фамилии, замена документов после свадьбы",
                    "Оплата пошлины",
                    "Если Вы уже были в законном браке",
                    "Возварст вступления в брак",
                    "Нужны ли свидетели"
                ]
            ),
            RandomForestClassificatorTask(
                vectorize_name='fasttext-model-3', 
                ovr_strategy=True,
                model_name='rf-ft3-submodel-cat-2-ovr',
                datasets=['./datasets/train/data-cat-6.xlsx'],
                order_class=[
                    "Место и способ подачи заявления",
                    "время подачи заявления",
                    "Возраст поступления в школу",
                    "Участки школ и информация о школе",
                    "Жалобы на школу",
                    "Свободные места в школу",
                    "Переводы из школы",
                    "Документы для поступления",
                    "Требования к детям",
                    "Помощь в заполнении заявления",
                    "Выбор хорошей школы",
                    "Отказы"
                ]
            ),
            RandomForestClassificatorTask(
                vectorize_name='fasttext-model-3', 
                ovr_strategy=True,
                model_name='rf-ft3-submodel-cat-6-ovr', 
                datasets=['./datasets/train/data-cat-7.xlsx'],
                order_class=[
                    "О разрешении на строительство",
                    "Для каких объектов нуно РС",
                    "Требования к получению РС",
                    "Причины отказа",
                    "Сроки выдачи и рассмотрения заявления",
                    "Не получается заполнить заявление",
                    "Построили без разрешения на строительство"
                ]
            ),
        ],
        local_scheduler=False
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