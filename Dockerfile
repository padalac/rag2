From python:3.10

ADD ./utils ./utils
ADD  ./Chroma_DB ./Chroma_DB/
ADD ./Output ./Output
ADD requirements.txt .
RUN pip install -r requirements.txt

WORKDIR utils
RUN sed -i  's/^mode=.*/mode=read/g' rag_config.ini


EXPOSE 7860
CMD ["python", "main.py"]
