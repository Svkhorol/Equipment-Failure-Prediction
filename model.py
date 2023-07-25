import numpy as np
import pandas as pd
import tensorflow as tf

from keras.callbacks import EarlyStopping
from keras import layers
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import MinMaxScaler


def drop_rows(X, y, df):
    """Функция для исключения нужных строк из X и y по датам
    df - датафрейм с данными, которые нужно удалить"""

    rows = []
    start_col = df['ДАТА_НАЧАЛА_НЕИСПРАВНОСТИ']
    end_col = df['ДАТА_УСТРАНЕНИЯ_НЕИСПРАВНОСТИ']

    for start, end in zip(start_col, end_col):
        indexes = X.loc[(X.index >= start) & (X.index <= end)].index
        rows.extend(indexes)

    return X.drop(rows), y.drop(rows)


# Подготовка данных
messages = pd.read_excel('data/processed/messages_processed.xlsx')

for n in [5, 6, 7, 8, 9]:
    X_columns = [
        f'ЭКСГАУСТЕР {n}. ТОК РОТОРА 1',
        f'ЭКСГАУСТЕР {n}. ТОК РОТОРА 2',
        f'ЭКСГАУСТЕР {n}. ТОК СТАТОРА',
        f'ЭКСГАУСТЕР {n}. ДАВЛЕНИЕ МАСЛА В СИСТЕМЕ',
        f'ЭКСГАУСТЕР {n}. ТЕМПЕРАТУРА ПОДШИПНИКА НА ОПОРЕ 1',
        f'ЭКСГАУСТЕР {n}. ТЕМПЕРАТУРА ПОДШИПНИКА НА ОПОРЕ 2',
        f'ЭКСГАУСТЕР {n}. ТЕМПЕРАТУРА ПОДШИПНИКА НА ОПОРЕ 3',
        f'ЭКСГАУСТЕР {n}. ТЕМПЕРАТУРА ПОДШИПНИКА НА ОПОРЕ 4',
        f'ЭКСГАУСТЕР {n}. ТЕМПЕРАТУРА МАСЛА В СИСТЕМЕ',
        f'ЭКСГАУСТЕР {n}. ТЕМПЕРАТУРА МАСЛА В МАСЛОБЛОКЕ',
        f'ЭКСГАУСТЕР {n}. ВИБРАЦИЯ НА ОПОРЕ 1',
        f'ЭКСГАУСТЕР {n}. ВИБРАЦИЯ НА ОПОРЕ 2',
        f'ЭКСГАУСТЕР {n}. ВИБРАЦИЯ НА ОПОРЕ 3',
        f'ЭКСГАУСТЕР {n}. ВИБРАЦИЯ НА ОПОРЕ 3. ПРОДОЛЬНАЯ.',
        f'ЭКСГАУСТЕР {n}. ВИБРАЦИЯ НА ОПОРЕ 4',
        f'ЭКСГАУСТЕР {n}. ВИБРАЦИЯ НА ОПОРЕ 4. ПРОДОЛЬНАЯ.']
    y_from_txt = []
    with open('data/y_train_columns.txt', 'r') as file:
        for line in file:
            y_from_txt.append(line.rstrip('\n'))
    y_columns = [column for column in y_from_txt if column.startswith(f'Y_ЭКСГАУСТЕР А/М №{n}')]

    X = pd.read_parquet('data/X_train.parquet', columns=X_columns)
    y = pd.read_parquet('data/y_train.parquet', columns=y_columns)

    M1 = messages[messages['ВИД_СООБЩЕНИЯ'] == 'M1']
    M1_ex_n = M1[M1['ИМЯ_МАШИНЫ'] == f'ЭКСГАУСТЕР А/М №{n}']
    X, y = drop_rows(X, y, M1_ex_n)

    M3 = messages[messages['ВИД_СООБЩЕНИЯ'] == 'M3']
    M3_ex_n = M3[M3['ИМЯ_МАШИНЫ'] == f'ЭКСГАУСТЕР А/М №{n}']
    y_list = list(M3_ex_n['НАЗВАНИЕ_ТЕХ_МЕСТА'].value_counts().index[:4])
    y = y[[f'Y_ЭКСГАУСТЕР А/М №{n}_{column}' for column in y_list]]

    endless_fail = M3_ex_n[M3_ex_n['ДАТА_УСТРАНЕНИЯ_НЕИСПРАВНОСТИ'].isna()].copy()    # без сообщения о завершении
    endless_fail = endless_fail[endless_fail['НАЗВАНИЕ_ТЕХ_МЕСТА'].apply(lambda x: x in y_list)]

    if len(endless_fail) != 0:
        date = endless_fail['ДАТА_НАЧАЛА_НЕИСПРАВНОСТИ'].iloc[0] + pd.Timedelta(minutes=8)
    else:
        date = len(X)
    del endless_fail

    X = X[:date]
    y = y[:date]

    if (X < 0).any().any():
        X = X[(X >= 0)]
        y = y.loc[X.index]

    X = X.interpolate()
    y.replace(2, 1, inplace=True)

    scaler = MinMaxScaler()
    X_norm = scaler.fit_transform(X)

    X_train = X_norm[:int(0.7 * len(X))]
    y_train = y[:int(0.7 * len(y))]
    X_val = X_norm[int(0.7 * len(X)):]
    y_val = y[int(0.7 * len(X)):]

    timesteps = 1
    samples_train = int(np.floor(X_train.shape[0] / timesteps))
    samples_val = int(np.floor(X_val.shape[0] / timesteps))
    X_train = X_train.reshape(samples_train, timesteps, X_train.shape[1])  # samples, timesteps, sensors
    X_val = X_val.reshape(samples_val, timesteps, X_val.shape[1])

    del X, X_norm

    # Модель
    out = y_train.shape[1]
    shuffle = False
    early_stop = EarlyStopping(monitor='val_loss', patience=5)
    tf.keras.backend.clear_session()

    model = tf.keras.Sequential([
        layers.LSTM(64, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True),
        layers.LSTM(64, activation='relu'),
        layers.Dense(out, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    history = model.fit(X_train, y_train,
                        epochs=10,
                        shuffle=shuffle,  # False
                        validation_data=(X_val, y_val),
                        callbacks=[early_stop])

    y_pred = model.predict(X_val)
    y_pred = np.round(y_pred)

    accuracy = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average='weighted')

    TP = ((y_pred == 1) * (y_val == 1)).sum()
    FP = ((y_pred == 1) * (y_val != 1)).sum()
    FN = ((y_pred != 1) * (y_val == 1)).sum()
    J = TP / (TP + FP + FN)

    print('accuracy:', accuracy)
    print('f1-score:', f1)
    print('J-index', J)
