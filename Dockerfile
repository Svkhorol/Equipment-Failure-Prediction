FROM python:3.10

WORKDIR /src

COPY requirements.txt .

RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt --no-cache-dir

COPY model.py .
COPY data/processed/messages_processed.xlsx data/processed/
COPY data/y_train_columns.txt data

CMD ["python", "model.py"]
