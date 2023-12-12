From python:3.10

WORKDIR /rag

COPY . .
RUN pip install -r requirements.txt

RUN sed -i  's/^mode=.*/mode=read/g' config/rag_config.ini

ENTRYPOINT ["streamlit", "run", "main.py"]

