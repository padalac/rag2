.DEFAULT_GOAL := none

clean:
	/bin/rm -rf Output/*
	/bin/rm -rf Chroma_DB/*

build_docker:
	sed -i "" 's/^mode=.*/mode=update_only/g' utils/rag_config.ini
	sed -i "" 's/^srvr_mode=.*/srvr_mode=in_memory/g' utils/rag_config.ini
	cd utils && python3 main.py
	docker build . -f Dockerfile -t rag:v1


