### Реализованный подход к задаче прогнозирования неисправностей M3 
Проблема определения наличия или отсутствия неисправностей может быть сформулирована как задача классификации. В качестве её решения подобрана нейронная сеть с LSTM-слоями. Модель показала лучшую результативность для технических мест, у которых количество примеров M3 в датасете представлено в достаточном количестве для обучения модели. 
 
В jupyter-ноутбуках приведены: 
- разведочный анализ данных (файл [eda_extra.ipynb](https://github.com/Svkhorol/Equipment-Failure-Prediction/blob/main/notebook/eda_extra.ipynb)) 
- порядок предварительной подготовки данных (файл [processing_messages.ipynb](https://github.com/Svkhorol/Equipment-Failure-Prediction/blob/main/notebook/processing_messages.ipynb), файлы в директории [processing/](https://github.com/Svkhorol/Equipment-Failure-Prediction/tree/main/notebook/processing)), отбираются наиболее полные и существенные данные. 
- архитектура нейронной сети и результаты обучения (директория [modeling/](https://github.com/Svkhorol/Equipment-Failure-Prediction/tree/main/notebook/modeling)) 
