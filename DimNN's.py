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
        
        history = model.fit(x, onehot_y, batch_size=5, epochs=100, validation_split=0) #callbacks=[checkpointer])

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