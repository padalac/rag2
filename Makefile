.DEFAULT_GOAL := none

venv: venv/touchfile

venv/touchfile: requirements.txt
	test -d venv || python3 -m venv venv
	. venv/bin/activate; pip install -Ur requirements.txt
	touch venv/touchfile

clean:
	/bin/rm -rf Output/*
	/bin/rm -rf Chroma_DB/*

UNAME_S := $(shell uname -s)
SED_SPACE=
ifeq ($(UNAME_S),Darwin)
        SED_SPACE=\"\"
endif

build_docker: venv
	sed -i ${SED_SPACE} 's/^mode=.*/mode=update_only/g' config/rag_config.ini
	sed -i ${SED_SPACE} 's/^srvr_mode=.*/srvr_mode=in_memory/g' config/rag_config.ini
	streamlit run main.py
	docker build . -f Dockerfile -t rag:v1


