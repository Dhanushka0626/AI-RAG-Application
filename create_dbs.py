import chromadb,os

from chromadb.utils.data_loaders import ImageLoader
from util import ImageEmbeddingFunction, DB_PATH, DATA_PATH
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


#new chromadb client
client_db = chromadb.PersistentClient(path=DB_PATH)

#create images_collection
try:
    client_db.delete_collection(name="imags")
except:
    pass
img_collection = client_db.create_collection(
    name="imags", 
    embedding_function=ImageEmbeddingFunction(),
    data_loader=ImageLoader()
    )

#for each category
for dir_ in os.listdir(DATA_PATH):
    dir_path = os.path.join(DATA_PATH, dir_)
    
    #add images to images_collection
    img_collection.add(
        ids=[f"{dir_}-{img_path}" for img_path in os.listdir(dir_path) if img_path.endswith('.jpeg')],
        uris=[os.path.join(dir_path, img_path) for img_path in os.listdir(dir_path) if img_path.endswith('.jpeg')]
    )
    
    #create documents_collection
    collection = client_db.create_collection(
        name=f"documents_{dir_}",
        embedding_function=SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
    )
    
    #load documents
    loader = DirectoryLoader(dir_path, glob="*.txt")
    documents = loader.load()
    
    #Documents to chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300, 
        chunk_overlap=100,
        length_function=len,
        add_start_index=True
        )
    
    chunks = text_splitter.split_documents(documents)
    
    #add chunks to documents_collection
    collection.add(
        ids=[str(j) for j in range(len(chunks))],
        documents=[chunks[j].page_content for j in range(len(chunks))],
        metadatas=[chunks[j].metadata for j in range(len(chunks))]
    )
    