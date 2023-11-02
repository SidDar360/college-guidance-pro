FROM python:3.9

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV FLASK_APP run.py
ENV DEBUG True

COPY requirements.txt .

# install python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install numpy
RUN pip install pandas
RUN pip install matplotlib
RUN pip install seaborn
RUN pip install -U scikit-learn scipy matplotlib
RUN pip install sklearn_pandas

COPY env.sample .env

COPY . .

RUN flask db init
#RUN flask db migrate
#RUN flask db upgrade
# gunicorn
CMD ["gunicorn", "--config", "gunicorn-cfg.py", "run:app"]
