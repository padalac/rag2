.DEFAULT_GOAL := none

venv: venv/touchfile

venv/touchfile: requirements.txt
	test -d venv || python3 -m venv venv
	. venv/bin/activate; pip install -Ur requirements.txt
	touch venv/touchfile

clean:
	/bin/rm -rf Output/*
	/bin/rm -rf Chroma_DB/*

build_docker: venv
	#sed -i  's/^mode=.*/mode=update_only/g' utils/rag_config.ini
	#sed -i  's/^srvr_mode=.*/srvr_mode=in_memory/g' utils/rag_config.ini
	#cd utils && python3 main.py
	docker build . -f Dockerfile -t rag:v1


