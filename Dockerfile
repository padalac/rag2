From python:3.10

WORKDIR /rag
COPY . .

RUN mkdir .streamlit
COPY  config/streamlit/* .streamlit

RUN pip install -r requirements.txt
#RUN pip install -r validation/requirements.txt

RUN sed -i  's/^mode=.*/mode=read/g' config/rag_config.ini

CMD ["streamlit", "run", "streamlit_app.py"]