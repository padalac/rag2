# rag
Capstone project code for enterprise RAG Q&amp;A solution

Execute python3 main.py  under utils directory as it expects the rag_config.ini file in that directory

rag_config.ini 'mode' has 2 modes 1. update mode and read mode
when in read mode, it expects to use existing embeddings where as in update mode it creates/uses embeddings

rag_config.ini has 2 possible configurations for chroma/srvr_mode. The default srvr_mode is 'in_memory' mode (should have named it file system peristent) which is a  DB created in the local directory. The other srvr_mode config is 'network' which expects a chromaDB server running somewhere (Deployment config). 


<p>Runnign a docker instance of Chroma DB.<p>  
Below is docker command to run ChromaDB. The sk-token1 is the authorization key for the client to provide
if you change that here, update the rag_config.ini file 

docker run -p 8000:8000 \
-e CHROMA_SERVER_AUTH_CREDENTIALS_PROVIDER="chromadb.auth.token.TokenConfigServerAuthCredentialsProvider" \
-e ALLOW_RESET=TRUE  -e CHROMA_SERVER_AUTH_PROVIDER="chromadb.auth.token.TokenAuthServerProvider" \
-e CHROMA_SERVER_AUTH_TOKEN_TRANSPORT_HEADER="X_CHROMA_TOKEN" \
-e CHROMA_SERVER_AUTH_CREDENTIALS="sk-token1" chromadb/chroma
