import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# Для одного столбца y

def plot_failure(X: pd.Series, y: pd.Series):
    '''Функция рисует временной ряд на заданном или всём интервале
    Цветом выделяются поломки:
    зеленый - без аварии
    красный - авария без простоя M3
    синий - авария с простоем M1
    '''

    # Разделяем индексы по поломкам в y
    M0_indx = y.loc[(y == 0)].index  # без аварий
    M1_indx = y.loc[(y == 1)].index  # авария с простоем
    M3_indx = y.loc[(y == 2)].index  # авария без простоя

    # Разделяем Х по поломкам
    M0 = X.loc[M0_indx]
    M1 = X.loc[M1_indx]
    M3 = X.loc[M3_indx]
    print('M0', M0.shape)
    print('M1', M1.shape)
    print('M3', M3.shape)

    plt.figure(figsize=(11, 7))
    sns.lineplot(M0, color='green')
    sns.lineplot(M1, color='blue')
    sns.lineplot(M3, color='red')


def create_sliding_window(X, y, window_size, shift=1):
    """
    Создает набор данных с окнами, скользящими по временному ряду.

    Аргументы:
    X - временной ряд
    window_size - размер окна
    shift - сдвиг окна

    Возвращает временной массив с окнами,
    массив с метками для каждого окна
    """

    dataset, labels = [], []

    for i in range(0, len(X) - window_size, shift):
        X_window = X[i:(i + window_size)]
        dataset.append(X_window)

        y_window = y[i:(i + window_size)]
        max_label = np.argmax(np.bincount(y_window, minlength=3))
        labels.append(max_label)

    return np.array(dataset), np.array(labels)


def restore_labels(y_window, window_size, shift=1):
    """
    Восстанавливает массив меток исходной размерности из массива с окнами.
    Значению присваивается метка из первого окна последовательности.

    """
    labels = []

    labels.extend([y_window[0]] * (window_size - shift))

    for i in range(len(y_window)):
        labels.extend([y_window[i]] * shift)

    return np.array(labels)
