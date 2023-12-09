# rag
Capstone project code for enterprise RAG Q&amp;A solution

Execute python3 main.py  under utils directory as it expects the rag_config.ini file in that directory

rag_config.ini 'mode' has 2 modes 1. update mode and read mode
when in read mode, it expects to use existing embeddings where as in update mode it creates/uses embeddings

rag_config.ini has 2 possible configurations for chroma/srvr_mode. The default srvr_mode is 'in_memory' mode (should have named it file system peristent) which is a  DB created in the local directory. The other srvr_mode config is 'network' which expects a chromaDB server running somewhere (Deployment config). 


Creating a application docker 

1. configure your OPENAI_API_KEY, SERPAPI_API_KEY
2. make clean (cleans up the output and Chroma_DB directories if they exist) 
3. make build_docker
   (Note: if this fails on a linux terminal, complaining about sed,  replace  sed -i "" with sed -i  in makefile. Some non-compatible changes between Linux and Mac)

#How to run the docker instance (replace -d with -it for interactive running)
docker run -d  -e OPENAI_API_KEY=$OPENAI_API_KEY -e SERPAPI_API_KEY=$SERPAPI_API_KEY --name my_rag9 rag:v1 