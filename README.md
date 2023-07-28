Дальнейшее развитие одной из задач проекта-финалиста хакатона "Лидеры Цифровой Трансформации 2023" ([ссылка на репозиторий команды](https://github.com/petr-larin/leaders2023-hackathon)).  
В репозитории содержится исходный код модели машинного обучения для прогнозирования неисправностей узлов эксгаустера, не приводящих к простою агломерационной машины.  
  
## Структура репозитория  
[`data`](https://github.com/Svkhorol/Equipment-Failure-Prediction/tree/main/data) - директория для хранения данных  
[`notebook`](https://github.com/Svkhorol/Equipment-Failure-Prediction/tree/main/notebook) - jupyter-ноутбуки с описанием выбраного подхода и метриками  
[`model.py`](https://github.com/Svkhorol/Equipment-Failure-Prediction/blob/main/model.py) - файл с кодом для воспроизведения модели  
  
## Воспроизведение обучения модели  
- Требуется Python версии не ниже 3.10  
  
- Клонировать репозиторий:
```bash
git clone https://github.com/Svkhorol/Equipment-Failure-Prediction.git
```  
- Перейти в папку с проектом, создать и активировать в ней виртуальное окружение:  
```bash
cd Equipment-Failure-Prediction
python -m venv venv
source venv/Scripts/activate
```
- Установить зависимости из файла requirements.txt:
```bash
python -m pip install --upgrade pip  
pip install -r requirements.txt  
```
- Загрузить в папку `data` файлы с исходными данными `X_train.parquet` и `y_train.parquet` 
 
- В папке `Equipment-Failure-Prediction` запустить скрипт обучения модели:  
```bash
python model.py 
```  