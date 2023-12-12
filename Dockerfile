From streamlit:1.29.0

WORKDIR /rag

COPY . .
RUN pip install -r requirements.txt

RUN sed -i  's/^mode=.*/mode=read/g' config/rag_config.ini


EXPOSE 7860
CMD ["streamlit", "run", "main.py"]
