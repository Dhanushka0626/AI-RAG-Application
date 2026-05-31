import chromadb
import cv2

from langchain_core.prompts import ChatPromptTemplate

from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from util import DB_PATH, ImageEmbeddingFunction
from langchain_ollama import ChatOllama
client_db = chromadb.PersistentClient(path=DB_PATH)

def classify_img(client_db, query_img):
    #Todo: error handling
    # currently onlybgr colorspace support
    #what happen if no similar image found?
    
    collection = client_db.get_collection(name="imags")
    
    embeddingFunction = ImageEmbeddingFunction()
    img = cv2.imread(query_img)
    embeddings = embeddingFunction([img])
    
    results = collection.query(
        query_embeddings=embeddings,
        n_results=1
    )
    
    return results['ids'][0][0].split('-')[0]

def get_most_similar_chunks(client_db,query_question,img_category):
    collection = client_db.get_collection(name=f"documents_{img_category}")
    
    embeddingFunction = SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    query_embedding = embeddingFunction([query_question])
    
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=3
    )
    
    return results['documents'][0], results['metadatas'][0]

def create_response(chunks_text,chunks_metadatas,query_question):
    PROMPT_TEMPLATE = """
    Answer the question based only on the following context:
    
    {context}
    
    Question: {question}
    
    """
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    
    context = "\n---\n".join(chunks_text)
    
    prompt = prompt_template.format(context=context, question=query_question)
    
    model = ChatOllama(model="mistral")
    
    response = model.invoke(prompt)
    
    return response.content,chunks_metadatas

if __name__ == "__main__":
    # input data: query img, query text
    query_img = r"E:\Occupation\Machine Learning\AI RAG Application\data\test-2.jpeg" # path to query image
    query_question = "When was it built?" # query question

    # classify img
    img_category = classify_img(client_db, query_img)

    # get most similar chunks
    chunks_text,chunks_metadatas = get_most_similar_chunks(client_db, query_question,img_category)

    # create response
    text, sources = create_response(chunks_text,chunks_metadatas,query_question)

