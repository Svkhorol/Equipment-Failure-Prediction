Дальнейшее развитие одной из задач проекта-финалиста хакатона "Лидеры Цифровой Трансформации 2023" ([ссылка на репозиторий команды](https://github.com/petr-larin/leaders2023-hackathon)).  
В репозитории содержится исходный код модели машинного обучения для прогнозирования неисправностей узлов эксгаустера, не приводящих к простою агломерационной машины (аномальные периоды работы).  
  
## Структура репозитория  
- [`data`](https://github.com/Svkhorol/Equipment-Failure-Prediction/tree/main/data) - директория для хранения данных  
- [`notebook`](https://github.com/Svkhorol/Equipment-Failure-Prediction/tree/main/notebook) - jupyter-ноутбуки с описанием выбраного подхода и метриками  
- [`model.py`](https://github.com/Svkhorol/Equipment-Failure-Prediction/blob/main/model.py) - файл с кодом для воспроизведения модели  
- [`Dockerfile`](https://github.com/Svkhorol/Equipment-Failure-Prediction/blob/main/Dockerfile) - Dockerfile для редактирования и пересборки образа при необходимости
  
## Воспроизведение обучения модели  
Воспроизвести обучение модели можно двумя способами: с помощью Docker, или запустив код обучения в самостоятельно собранном проекте.

#### Запуск Docker-контейнера
- Требуется установленный и запущенный Docker.
- Запустить контейнер из директории хоста, содержащей файлы с исходными данными `X_train.parquet` и `y_train.parquet`:
```bash
docker run -v "$(pwd)":/src/data khoro/exhaust:v2.0
```
- В результате загрузится образ с Docker Hub, запустится контейнер. По мере работы контейнера будут выводиться метрики обучения.

#### Запуск исходного кода
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
- В папке `Equipment-Failure-Prediction` запустить скрипт обучения модели (`model.py`), который ожидает на входе один параметр: путь до директории с исходными данными `X_train.parquet` и `y_train.parquet`. Пример запуска:  
```bash
python model.py --indir data 
```  