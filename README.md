# rag
Capstone project code for enterprise RAG Q&amp;A solution

Execute python3 main.py to run the program under utils directory as it expects the rag_config.ini file in that directory.
Currently it waits forever after executing the queries (to prevent docker from crashing. Manually kill it.)

rag_config.ini 'mode' has 2 modes 1. update mode and read mode
when in read mode, it expects to use existing embeddings where as in update mode it creates/uses embeddings

rag_config.ini has 2 possible configurations for chroma/srvr_mode. The default srvr_mode is 'in_memory' mode (should have named it file system peristent) which is a  DB created in the local directory. The other srvr_mode config is 'network' which expects a chromaDB server running somewhere (Deployment config). 


Creating a application docker 

1. configure your OPENAI_API_KEY, SERPAPI_API_KEY
2. If you want to use the existing vector index values, do not do make clean. Go to step 3
2. make clean (cleans up the output and Chroma_DB directories if they exist) 
3. make build_docker
   (Note: if this fails on a mac terminal, complaining about sed,  replace  sed -i  with sed -i "" in makefile. Some non-compatible changes between Linux and Mac)
4. After the docker build is changed manually edit the rag_config.ini file to have mode=update (if you want to create the vector db index values). Otherwise, leave mode=read


# Default port of streamlit app is 8501. You need to map it to another port for accessing the app outside container
# using the switch "-p 8080:8501"
# The app running on docker container can be accessed by using the url http://localhost:8080

# How to run the docker instance (replace -d with -it for interactive running)

docker run -it -p 8080:8501 -e OPENAI_API_KEY=$OPENAI_API_KEY -e SERPAPI_API_KEY=$SERPAPI_API_KEY --name my_rag1 rag:v1
