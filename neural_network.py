import luigi
from pathlib import Path


class ClassificatorTask(luigi.Task):
    _name = 'genaral'
    datasets = luigi.ListParameter(default=None)
    output_dir = luigi.Parameter(default='./dist/model/classificators')
    model_name = luigi.Parameter()
    vectorize_name = luigi.Parameter()
    order_class = luigi.ListParameter(default=None)
    ovr_strategy = luigi.BoolParameter(default=False)
    use_shuffle = luigi.BoolParameter(default=False)

    def output(self):
        model_dir = Path(self.output_dir) / self.__class__._name
        model_dir.mkdir(exist_ok=True, parents=True)
        return luigi.LocalTarget(str((model_dir / self.model_name).with_suffix('.pickle')))


    def run(self):
        import pandas as pd
        import numpy as np

        from vectorize import FastTextVectorizeTask
        from preproccesors import SimpleExcelDatasetTransformTask

        self.meta = {}

        if self.datasets is None:
            datasets =  [str(p) for p in Path('./datasets/train').iterdir() if p.is_file() and p.suffix == '.xlsx']
        else:
            datasets = self.datasets


        datasets_files = []

        for dataset in datasets:
            buf = yield SimpleExcelDatasetTransformTask(input_file=dataset)
            datasets_files.append(buf.path)
        # datasets = [item.path for item in luigi.task.getpaths(datasets)]


        input_ = yield FastTextVectorizeTask(datasets=datasets_files, fastext_model=self.vectorize_name)


        data = pd.read_pickle(input_.path)
        if self.use_shuffle:
            data = data.sample(frac=1)
        

        x_train = np.zeros((data['вопрос'].shape[0], *data['вопрос'][0].shape))
        if data['ответ'].dtype.name == 'object':
            y_train = list(data['ответ'].values)
        else:
            y_train = np.zeros((data['ответ'].shape[0], *data['ответ'][0].shape))

        for i, row in data.iterrows():
            x_train[i] = row['вопрос']
            if data['ответ'].dtype.name != 'object':
                y_train[i] = row['ответ']
        
        if self.ovr_strategy:
            self.set_status_message('Use One vs Rest strategy')
        model = self.build_and_train(x=x_train, y=y_train)
        self.model_save(model)

    
    def build_and_train(self, x, y):
        raise NotImplementedError()


    def model_save(self, model):
        import joblib
        import json
        joblib.dump(model, self.output().path)
        classes = list(model.classes_ )
        with Path(self.output().path).with_suffix('.json').open('w') as f:
            json.dump({
                'classes': classes,
                **self.meta,
                # 'word_vec_size': self.meta.get('word_vec_size', None),
                # 'len_words': self.meta.get('len_words', None),
                'order_class': self.order_class or classes,
                'class_map': [classes.index(cls) for cls in (self.order_class or classes)]
            }, f, indent=4)


class SVMClassificatorTask(ClassificatorTask):
    _name = 'svm'

    def build_and_train(self, x, y):
        from sklearn.svm import SVC
        from sklearn.multiclass import OneVsRestClassifier

        lx,ly,lz = x.shape
        self.meta.update({
            "word_vec_size": lz,
            "len_words": ly,
        })
        x_train = x.reshape(lx, ly*lz)
        model = SVC(probability=True)
        if self.ovr_strategy:
            model = OneVsRestClassifier(model)
        model.fit(x_train, y)
        score = model.score(x_train, y)
        self.meta['score'] = score
        self.set_status_message(f'Model fit complete. Score {score}')
        return model


class KNeighborsClassificatorTask(ClassificatorTask):
    _name = 'kneighbors'

    def build_and_train(self, x, y):
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.multiclass import OneVsRestClassifier

        lx,ly,lz = x.shape
        self.meta.update({
            "word_vec_size": lz,
            "len_words": ly,
        })
        x_train = x.reshape(lx, ly*lz)
        model = KNeighborsClassifier()
        if self.ovr_strategy:
            model = OneVsRestClassifier(model)
        model.fit(x_train, y)
        score = model.score(x_train, y)
        self.meta['score'] = score
        self.set_status_message(f'Model fit complete. Score {score}')

        return model


class RandomForestClassificatorTask(ClassificatorTask):
    _name = 'randomforest'

    n_estimators = luigi.IntParameter(default=10)

    def build_and_train(self, x, y):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.multiclass import OneVsRestClassifier

        lx,ly,lz = x.shape
        self.meta.update({
            "word_vec_size": lz,
            "len_words": ly,
        })
        x_train = x.reshape(lx, ly*lz)
        model = RandomForestClassifier(n_estimators=self.n_estimators)
        if self.ovr_strategy:
            model = OneVsRestClassifier(model)
        model.fit(x_train, y)
        score = model.score(x_train, y)
        self.meta['score'] = score
        self.set_status_message(f'Model fit complete. Score {score}')

        return model


class AdaBoostClassificatorTask(ClassificatorTask):
    _name = 'adabooost'
    
    def build_and_train(self, x, y):
        from sklearn.ensemble import AdaBoostClassifier
        from sklearn.multiclass import OneVsRestClassifier

        lx,ly,lz = x.shape
        self.meta.update({
            "word_vec_size": lz,
            "len_words": ly,
        })
        x_train = x.reshape(lx, ly*lz)
        model = AdaBoostClassifier()
        if self.ovr_strategy:
            model = OneVsRestClassifier(model)
        model.fit(x_train, y)
        score = model.score(x_train, y)
        self.meta['score'] = score
        self.set_status_message(f'Model fit complete. Score {score}')

        return model


class NeuralNetworkClassificatorTask(ClassificatorTask):
    _name = 'neural_network'

    solver = luigi.Parameter(default='lbfgs')
    activation = luigi.Parameter(default='relu')
    hidden_layer_sizes = luigi.ListParameter()
    batch_size = luigi.Parameter(default='auto')
    
    def build_and_train(self, x, y):
        from sklearn.neural_network import MLPClassifier
        from sklearn.multiclass import OneVsRestClassifier

        lx,ly,lz = x.shape
        self.meta.update({
            "word_vec_size": lz,
            "len_words": ly,
        })
        x_train = x.reshape(lx, ly*lz)
        model = MLPClassifier(
            hidden_layer_sizes=list(self.hidden_layer_sizes),
            solver=self.solver,
            activation=self.activation,
            batch_size='auto' if self.batch_size == 'auto' else int(self.batch_size)
        )
        if self.ovr_strategy:
            model = OneVsRestClassifier(model)
        model.fit(x_train, y)
        score = model.score(x_train, y)
        self.meta['score'] = score
        self.set_status_message(f'Model fit complete. Score {score}')

        return model


class KerasNeuralNetworkClassificatorTask(ClassificatorTask):
    _name = 'keras_cnn'


    def output(self):
        model_dir = Path(self.output_dir) / self.__class__._name
        model_dir.mkdir(exist_ok=True, parents=True)
        return luigi.LocalTarget(str((model_dir / self.model_name).with_suffix('.h5')))

    def build_and_train(self, x, y):
        from keras.layers import Layer, Dense, Conv1D, MaxPool1D, pooling, MaxPool1D, Flatten
        from keras.models import Sequential
        from sklearn.preprocessing import LabelBinarizer

        encoder = LabelBinarizer()
        onehot_y = encoder.fit_transform(y)
        

        lx,ly,lz = x.shape
        self.meta.update({
            "word_vec_size": lz,
            "len_words": ly,
            "encoder_classes": encoder.classes_
        })

        model = Sequential()
        model.add(Conv1D(2, 3, input_shape=(ly, lz), activation='sigmoid'))
        # model.add(MaxPool1D())
        model.add(Flatten())
        model.add(Dense(3, activation='sigmoid'))
        model.add(Dense(onehot_y.shape[1], activation='softmax'))

        model.compile('Adam', loss='mse')

        model.fit(x, onehot_y, epochs=100)

        score = float(model.test_on_batch(x, onehot_y))
        self.meta['score'] = score
        self.set_status_message(f'Model fit complete. Score {score}')
        return model

    def model_save(self, model):
        from keras.models import save_model
        import json
        save_model(model, self.output().path)
        classes = list(self.meta['encoder_classes'])
        del self.meta['encoder_classes']
        with Path(self.output().path).with_suffix('.json').open('w') as f:
            json.dump({
                'classes': classes,
                **self.meta,
                # 'word_vec_size': self.meta.get('word_vec_size', None),
                # 'len_words': self.meta.get('len_words', None),
                'order_class': self.order_class or classes,
                'class_map': [classes.index(cls) for cls in (self.order_class or classes)]
            }, f, indent=4)


class MyNN (KerasNeuralNetworkClassificatorTask):
    _name = 'mynn'

    def build_and_train(self, x, y):
        from keras.layers import Dense, Activation, Flatten, Dropout
        from keras.models import Sequential
        from sklearn.preprocessing import LabelBinarizer
        import matplotlib.pyplot as plt
            
        encoder = LabelBinarizer()
        onehot_y = encoder.fit_transform(y)

        lx,ly,lz = x.shape
        self.meta.update({
                "word_vec_size": lz,
                "len_words": ly,
                "encoder_classes": encoder.classes_
            })

        model = Sequential()

        model.add(Dense(120, activation='tanh', input_shape=(ly, lz)))
        model.add(Dense(60, activation='tanh', input_shape=(ly, lz)))
        model.add(Dropout(0.1))
        model.add(Dense(30, activation='tanh', input_shape=(ly, lz)))
        model.add(Flatten())
        model.add(Dense(onehot_y.shape[1], activation='softmax'))
        
        model.compile('Adam', loss='mse', metrics=['acc'])
        
        history = model.fit(x, onehot_y, batch_size=5, epochs=200, validation_split=0.1)

        scores = float(model.test_on_batch(x, onehot_y)[0])
        print(f'точность на тестовых данных: {scores*100}')

        plt.figure(0)
        plt.subplot(2, 1, 1)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'], '--')
        plt.title('Loss')
        plt.legend(['Loss', 'Val loss'])

        plt.subplot(2, 1, 2)
        plt.title('Acc')
        plt.plot(history.history['acc'])
        plt.savefig(str(Path(self.output().path).with_suffix('.png')))

        return model

class MyNN2 (KerasNeuralNetworkClassificatorTask):
    _name = 'mynn2'

    def build_and_train(self, x, y):
        from keras.layers import Dense, Activation, Flatten, Dropout
        from keras.models import Sequential
        from sklearn.preprocessing import LabelBinarizer
        import matplotlib.pyplot as plt
            
        encoder = LabelBinarizer()
        onehot_y = encoder.fit_transform(y)

        lx,ly,lz = x.shape
        self.meta.update({
                "word_vec_size": lz,
                "len_words": ly,
                "encoder_classes": encoder.classes_
            })

        model = Sequential()
        #добавил несколько слоев
        model.add(Dense(120, activation='tanh', input_shape=(ly, lz)))
        model.add(Dense(90, activation='relu'))
        #дальше меняются только функции активации, добавляю дропауты
        model.add(Dropout(0.1))
        model.add(Dense(60, activation='tanh'))
        model.add(Dropout(0.1))
        model.add(Flatten())
        model.add(Dense(onehot_y.shape[1], activation='softmax'))
        
        model.compile('Adam', loss='mse', metrics=['acc'])
        
        history = model.fit(x, onehot_y, batch_size=5, epochs=200, validation_split=0.1)

        scores = float(model.test_on_batch(x, onehot_y)[0])
        print(f'точность на тестовых данных: {scores*100}')

        plt.figure(0)
        plt.subplot(2, 1, 1)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'], '--')
        plt.title('Loss')
        plt.legend(['Loss', 'Val loss'])

        plt.subplot(2, 1, 2)
        plt.title('Acc')
        plt.plot(history.history['acc'])
        plt.savefig(str(Path(self.output().path).with_suffix('.png')))

        return model

class MyNN3 (KerasNeuralNetworkClassificatorTask):
    _name = 'mynn3'

    def build_and_train(self, x, y):
        from keras.layers import Dense, Activation, Dropout, Flatten
        from keras.models import Sequential
        from sklearn.preprocessing import LabelBinarizer
        #from keras.callbacks import ModelCheckpoint
        import matplotlib.pyplot as plt
            
        encoder = LabelBinarizer()
        onehot_y = encoder.fit_transform(y)

        lx,ly,lz = x.shape
        self.meta.update({
                "word_vec_size": lz,
                "len_words": ly,
                "encoder_classes": encoder.classes_
            })

        model = Sequential()

        #+drop 0.1
        model.add(Dense(60, input_shape=(ly, lz)))
        model.add(Dropout(0.1))
        #activate fun exp
        model.add(Dense(60, activation='relu'))
        #drop +0.1
        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(onehot_y.shape[1], activation='softmax'))

        model.compile('Adam', loss='mse', metrics=['acc'])
        #checkpointer = ModelCheckpoint(filepath='C:/Users/dkkay/Desktop/Дипломная/train-nn-dev/dist/checkpoints', verbose=1, save_best_only=True)
        
        history = model.fit(x, onehot_y, batch_size=5, epochs=200, validation_split=0.1) #callbacks=[checkpointer])

        scores = float(model.test_on_batch(x, onehot_y)[0])
        print(f'точность на тестовых данных: {scores*100}')

        plt.figure(0)
        plt.subplot(2, 1, 1)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'], '--')
        plt.title('Loss')
        plt.legend(['Loss', 'Val loss'])

        plt.subplot(2, 1, 2)
        plt.title('Acc')
        plt.plot(history.history['acc'])
        plt.savefig(str(Path(self.output().path).with_suffix('.png')))

        return model

class MyNN4 (KerasNeuralNetworkClassificatorTask):
    _name = 'mynn4'

    def build_and_train(self, x, y):
        from keras.layers import Dense, Activation, Flatten, Dropout
        from keras.models import Sequential
        from sklearn.preprocessing import LabelBinarizer
        import matplotlib.pyplot as plt
            
        encoder = LabelBinarizer()
        onehot_y = encoder.fit_transform(y)

        lx,ly,lz = x.shape
        self.meta.update({
                "word_vec_size": lz,
                "len_words": ly,
                "encoder_classes": encoder.classes_
            })

        model = Sequential()

        #
        model.add(Dense(160, input_shape=(ly, lz)))
        model.add(Dense(90, activation='tanh'))
        #drop 0.3
        model.add(Dropout(0.3))
        model.add(Dense(60, activation='relu'))
        model.add(Flatten())
        model.add(Dense(onehot_y.shape[1], activation='softmax'))

        model.compile('Adam', loss='mse', metrics=['acc'])
        
        history = model.fit(x, onehot_y, batch_size=5, epochs=100, validation_split=0.1)

        scores = float(model.test_on_batch(x, onehot_y)[0])
        print(f'точность на тестовых данных: {scores*100}')

        plt.figure(0)
        plt.subplot(2, 1, 1)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'], '--')
        plt.title('Loss')
        plt.legend(['Loss', 'Val loss'])

        plt.subplot(2, 1, 2)
        plt.title('Acc')
        plt.plot(history.history['acc'])
        plt.savefig(str(Path(self.output().path).with_suffix('.png')))

        return model

class MyNN5 (KerasNeuralNetworkClassificatorTask):
    _name = 'mynn5'

    def build_and_train(self, x, y):
        from keras.layers import Dense, Activation, Dropout, Flatten
        from keras.models import Sequential
        from sklearn.preprocessing import LabelBinarizer
        import matplotlib.pyplot as plt
            
        encoder = LabelBinarizer()
        onehot_y = encoder.fit_transform(y)

        lx,ly,lz = x.shape
        self.meta.update({
                "word_vec_size": lz,
                "len_words": ly,
                "encoder_classes": encoder.classes_
            })

        model = Sequential()

        #используем 2*2 одинаковых слоя, drop -0.1
        model.add(Dense(180, input_shape=(ly, lz)))
        model.add(Dropout(0.3))
        model.add(Flatten())
        model.add(Dense(90, activation = "relu"))
        model.add(Dropout(0.2))
        model.add(Dense(180, activation='exponential'))
        model.add(Dropout(0.1))
        model.add(Dense(90, activation = "relu"))
        model.add(Dense(onehot_y.shape[1], activation='softmax'))

        model.compile('Adam', loss='mse', metrics=['acc'])
        
        history = model.fit(x, onehot_y, batch_size=5, epochs=100, validation_split=0.1)

        scores = float(model.test_on_batch(x, onehot_y)[0])
        print(f'точность на тестовых данных: {scores*100}')

        plt.figure(0)
        plt.subplot(2, 1, 1)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'], '--')
        plt.title('Loss')
        plt.legend(['Loss', 'Val loss'])

        plt.subplot(2, 1, 2)
        plt.title('Acc')
        plt.plot(history.history['acc'])
        plt.savefig(str(Path(self.output().path).with_suffix('.png')))

        return model

class MyNN6 (KerasNeuralNetworkClassificatorTask):
    _name = 'mynn6'

    def build_and_train(self, x, y):
        from keras.layers import Dense, Activation, Flatten
        from keras.models import Sequential
        from sklearn.preprocessing import LabelBinarizer
        import matplotlib.pyplot as plt
            
        encoder = LabelBinarizer()
        onehot_y = encoder.fit_transform(y)

        lx,ly,lz = x.shape
        self.meta.update({
                "word_vec_size": lz,
                "len_words": ly,
                "encoder_classes": encoder.classes_
            })

        model = Sequential()

        #разные активации
        model.add(Dense(100, input_shape=(ly, lz)))
        model.add(Dense(50, activation='exponential'))
        model.add(Dense(100, activation='linear'))
        model.add(Dense(150, activation='softplus'))
        model.add(Dense(100, activation='relu'))
        model.add(Flatten())
        model.add(Dense(onehot_y.shape[1], activation='softmax'))

        model.compile('Adam', loss='mse', metrics=['acc'])
        
        history = model.fit(x, onehot_y, batch_size=5, epochs=100, validation_split=0.1)

        scores = float(model.test_on_batch(x, onehot_y)[0])
        print(f'точность на тестовых данных: {scores*100}')

        plt.figure(0)
        plt.subplot(2, 1, 1)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'], '--')
        plt.title('Loss')
        plt.legend(['Loss', 'Val loss'])

        plt.subplot(2, 1, 2)
        plt.title('Acc')
        plt.plot(history.history['acc'])
        plt.savefig(str(Path(self.output().path).with_suffix('.png')))

        return model

class MyNN7 (KerasNeuralNetworkClassificatorTask):
    _name = 'mynn7'

    def build_and_train(self, x, y):
        from keras.layers import Dense, Activation, Conv1D, Dropout, Flatten
        from keras.models import Sequential
        from sklearn.preprocessing import LabelBinarizer
        import matplotlib.pyplot as plt
            
        encoder = LabelBinarizer()
        onehot_y = encoder.fit_transform(y)

        lx,ly,lz = x.shape
        self.meta.update({
                "word_vec_size": lz,
                "len_words": ly,
                "encoder_classes": encoder.classes_
            })

        model = Sequential()
        
        #+ conv1d
        model.add(Dense(120, input_shape=(ly, lz)))
        model.add(Conv1D(filters=1, kernel_size=3, padding='valid', activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(60, activation='tanh'))
        model.add(Flatten())
        model.add(Dense(onehot_y.shape[1], activation='softmax'))

        model.compile('Adam', loss='mse', metrics=['acc'])
        
        history = model.fit(x, onehot_y, batch_size=5, epochs=100, validation_split=0.1)

        scores = float(model.test_on_batch(x, onehot_y)[0])
        print(f'точность на тестовых данных: {scores*100}')

        plt.figure(0)
        plt.subplot(2, 1, 1)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'], '--')
        plt.title('Loss')
        plt.legend(['Loss', 'Val loss'])

        plt.subplot(2, 1, 2)
        plt.title('Acc')
        plt.plot(history.history['acc'])
        plt.savefig(str(Path(self.output().path).with_suffix('.png')))

        return model

class MyNN8 (KerasNeuralNetworkClassificatorTask):
    _name = 'mynn8'

    def build_and_train(self, x, y):
        from keras.layers import Dense, Activation, Dropout, Flatten
        from keras.models import Sequential
        from sklearn.preprocessing import LabelBinarizer
        import matplotlib.pyplot as plt
            
        encoder = LabelBinarizer()
        onehot_y = encoder.fit_transform(y)

        lx,ly,lz = x.shape
        self.meta.update({
                "word_vec_size": lz,
                "len_words": ly,
                "encoder_classes": encoder.classes_
            })

        model = Sequential()

        #много слоев
        model.add(Dense(120, input_shape=(ly, lz)))
        model.add(Dense(160, activation='tanh'))
        model.add(Dropout(0.1))
        model.add(Dense(100, activation='sigmoid'))
        model.add(Dropout(0.2))
        model.add(Dense(60, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(80, activation='hard_sigmoid'))
        model.add(Flatten())
        model.add(Dense(onehot_y.shape[1], activation='softmax'))

        model.compile('Adam', loss='mse', metrics=['acc'])
        
        history = model.fit(x, onehot_y, batch_size=5, epochs=100, validation_split=0.1)

        scores = float(model.test_on_batch(x, onehot_y)[0])
        print(f'точность на тестовых данных: {scores*100}')

        plt.figure(0)
        plt.subplot(2, 1, 1)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'], '--')
        plt.title('Loss')
        plt.legend(['Loss', 'Val loss'])

        plt.subplot(2, 1, 2)
        plt.title('Acc')
        plt.plot(history.history['acc'])
        plt.savefig(str(Path(self.output().path).with_suffix('.png')))

        return model

class MyNN9 (KerasNeuralNetworkClassificatorTask):
    _name = 'mynn9'

    def build_and_train(self, x, y):
        from keras.layers import Dense, Activation, Dropout, Flatten
        from keras.models import Sequential
        from sklearn.preprocessing import LabelBinarizer
        import matplotlib.pyplot as plt
            
        encoder = LabelBinarizer()
        onehot_y = encoder.fit_transform(y)

        lx,ly,lz = x.shape
        self.meta.update({
                "word_vec_size": lz,
                "len_words": ly,
                "encoder_classes": encoder.classes_
            })

        model = Sequential()

        #+layer 'activation', init
        model.add(Dense(100, input_shape=(ly, lz)))
        model.add(Dropout(0.15))
        model.add(Dense(50, init='uniform', activation='sigmoid'))
        model.add(Activation('softplus'))
        model.add(Dropout(0.15))
        model.add(Dense(100, init='uniform', activation='tanh'))
        model.add(Flatten())
        model.add(Dense(onehot_y.shape[1], activation='softmax'))
        
        model.compile('Adam', loss='mse', metrics=['acc'])
        
        history = model.fit(x, onehot_y, batch_size=5, epochs=100, validation_split=0.1)

        scores = float(model.test_on_batch(x, onehot_y)[0])
        print(f'точность на тестовых данных: {scores*100}')

        plt.figure(0)
        plt.subplot(2, 1, 1)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'], '--')
        plt.title('Loss')
        plt.legend(['Loss', 'Val loss'])

        plt.subplot(2, 1, 2)
        plt.title('Acc')
        plt.plot(history.history['acc'])
        plt.savefig(str(Path(self.output().path).with_suffix('.png')))

        return model

class MyNN10 (KerasNeuralNetworkClassificatorTask):
    _name = 'mynn10'

    def build_and_train(self, x, y):
        from keras.layers import Dense, Activation, Flatten
        from keras.models import Sequential
        from sklearn.preprocessing import LabelBinarizer
        import matplotlib.pyplot as plt
            
        encoder = LabelBinarizer()
        onehot_y = encoder.fit_transform(y)

        lx,ly,lz = x.shape
        self.meta.update({
                "word_vec_size": lz,
                "len_words": ly,
                "encoder_classes": encoder.classes_
            })

        model = Sequential()

        #drop?
        model.add(Dense(120, input_shape=(ly, lz)))
        model.add(Dense(60, input_shape=(ly, lz), activation='relu'))
        model.add(Dense(140, input_shape=(ly, lz), activation='tanh'))
        model.add(Dense(70, input_shape=(ly, lz),activation='relu'))
        model.add(Flatten())
        model.add(Dense(onehot_y.shape[1], activation='softmax'))
                
        model.compile('Adam', loss='mse', metrics=['acc'])
        
        history = model.fit(x, onehot_y, batch_size=5, epochs=100, validation_split=0.1)

        scores = float(model.test_on_batch(x, onehot_y)[0])
        print(f'точность на тестовых данных: {scores*100}')

        plt.figure(0)
        plt.subplot(2, 1, 1)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'], '--')
        plt.title('Loss')
        plt.legend(['Loss', 'Val loss'])

        plt.subplot(2, 1, 2)
        plt.title('Acc')
        plt.plot(history.history['acc'])
        plt.savefig(str(Path(self.output().path).with_suffix('.png')))

        return model

class MyNN11(KerasNeuralNetworkClassificatorTask):
    _name = 'mynn11'

    def build_and_train(self, x, y):
        from keras.layers import Dense, Activation, Flatten
        from keras.models import Sequential
        from sklearn.preprocessing import LabelBinarizer
        import matplotlib.pyplot as plt
                
        encoder = LabelBinarizer()
        onehot_y = encoder.fit_transform(y)

        lx,ly,lz = x.shape
        self.meta.update({
                "word_vec_size": lz,
                "len_words": ly,
                "encoder_classes": encoder.classes_
            })

        model = Sequential()
                
        model.add(Dense(120, input_shape=(ly, lz)))

        model.add(Flatten())
        model.add(Dense(onehot_y.shape[1], activation='softmax'))

        model.compile('Adam', loss='mse', metrics=['acc'])
            
        history = model.fit(x, onehot_y, batch_size=5, epochs=100, validation_split=0.1)

        scores = float(model.test_on_batch(x, onehot_y)[0])
        print(f'точность на тестовых данных: {scores*100}')

        plt.figure(0)
        plt.subplot(2, 1, 1)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'], '--')
        plt.title('Loss')
        plt.legend(['Loss', 'Val loss'])

        plt.subplot(2, 1, 2)
        plt.title('Acc')
        plt.plot(history.history['acc'])
        plt.savefig(str(Path(self.output().path).with_suffix('.png')))

        return model

