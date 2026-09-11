#fastapi
from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os

#langchain
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAI
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate


##Llamma parse
from llama_parse import LlamaParse

#python
from dotenv import load_dotenv
load_dotenv()

import shutil
from pathlib import Path

#Saving file temprory
UPLODED_FOL=Path("uploads")
UPLODED_FOL.mkdir(exist_ok=True)


#fastapi app
app=FastAPI(title="AI Reasearch API")

class QueryModel(BaseModel):
    question:str

#global_vetor
vector_store=None

@app.post('/upload-paper/')
def reseach_paper(file: UploadFile = File(...)):
    global vector_store
    try:
        file_path= UPLODED_FOL/ file.filename
        with open(file_path, 'wb') as buffer:
            shutil.copyfileobj(file.file, buffer)

        llama_api_key = os.getenv("LLAMA_CLOUD_API_KEY")
        if not llama_api_key:
            raise HTTPException(status_code=500, detail="LLAMA_CLOUD_API_KEY environment variable is not set.")

        parse=LlamaParse(
            api_key=llama_api_key,
            result_type='markdown',
            verbose=True
        )

        mark_file=parse.load_data(str(file_path))

        if not mark_file:
            raise HTTPException(status_code=400, detail="LlamaParse could not extract any content from this PDF.")

        langchain_doc = [
            Document(page_content=doc.text, metadata=doc.metadata) for doc in mark_file
        ]

        split=RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks=split.split_documents(langchain_doc)

        embedding=GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")
        vector_store=FAISS.from_documents(chunks,embedding)

        return{
            'status': 'successfull',
            'filename': file.filename,
            'messages': f"pdf successfully stored in vector database with lenth of {len(chunks)} of on chunk"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/question")
async def user_question(ques:QueryModel):
    global vector_store

    if vector_store is None:
        raise HTTPException(status_code=400, detail="Please Upload Research Paper Pdf")

    llm=GoogleGenerativeAI(
         model="gemini-3.6-flash",
         temperature=0.7,
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    system_prompt = (
        "You are a friendly and encouraging AI Research Paper Tutor for beginners.\n"
        "Explain the answer to the following question based on the provided research paper context.\n"
        "Use simple language, analogies, and break down complex technical terms.\n"
        "If there are equations, explain what they represent in simple words.\n\n"
        "Context:\n{context}"
    )

    prompt=ChatPromptTemplate([
        ("system",system_prompt),
        ("human", "{input}")
    ])

    combine_docs_chain = create_stuff_documents_chain(llm, prompt)

    rag_chain=create_retrieval_chain(retriever, combine_docs_chain)

    response=rag_chain.invoke({"input": ques.question})

    return{
        "answer":response["answer"]
    }



            








