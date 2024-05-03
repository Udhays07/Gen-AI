import warnings
warnings.filterwarnings("ignore")

from langchain.embeddings.cohere import CohereEmbeddings
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
import textwrap as tr
from langchain.callbacks import get_openai_callback
from langchain_openai import ChatOpenAI

import os

def retrieve_answers_from_document(document_path, questions):
    global db


    # Initialize Cohere Embeddings with API key
    cohere_api_key="lpgMmaknlgN0iqVbCvtzCCJQROIFT9RN22R1iCpO"
    embeddings = CohereEmbeddings(model="multilingual-22-12", cohere_api_key=cohere_api_key)

    # Load document
    loader = TextLoader(document_path)
    documents = loader.load()

    # Split document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)

    # Create a Chroma vector store from documents
    db = Chroma.from_documents(docs, embeddings)

    # Prepare the prompt template for QA
    prompt_template = """
        Given Passage: [{context}]

        When provided with a question and a student's answer during the chat, follow these steps to evaluate:
        {question}
        1. Receive the Question: [You will insert the question in Tamil during the chat]
        2. Receive the Student's Answer: [You will insert the student's answer in Tamil during the chat]

        ---

        Based on the given passage, question, and student's answer, evaluate the response according to the following criteria:

        1.**Factual Accuracy**: Assess the accuracy of the information provided by the student in relation to the passage given and question asked
            make sure the answers match thr exact key word answers.
        2. **Spelling Mistakes**: Identify any spelling errors in the Tamil language.
        3. **Sentence Formation**: Analyze the grammatical structure and coherence of the sentences in the student's answer.


        Provide a short comment in Tamil (10 words only) based on the correct answer , Students answer, and question. Then, assign a score out of 10 based on these criteria.
        Format your response as follows:
        [
          "TeacherComment": "உங்கள் கருத்து இங்கே",
          "Score": [Insert score out of 10 here]
        ]




    """

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])


    # Initialize OpenAI with API key

    openai_api_key="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"
    llm_o = ChatOpenAI(model = "gpt-4",openai_api_key=openai_api_key,
                       temperature=0,
                       max_tokens=256,
                       top_p=1,
                      frequency_penalty=0,
                      presence_penalty=0)

    # Initialize the QA Chain
    chain_type_kwargs = {"prompt": PROMPT}
    qa = RetrievalQA.from_chain_type(llm=llm_o,
                                     chain_type="stuff",
                                     retriever=db.as_retriever(search_kwargs={"k": 1}),
                                     chain_type_kwargs=chain_type_kwargs,
                                     return_source_documents=True)


    # Process each question
    for question in questions:
        with get_openai_callback() as cb:
          answer = qa({"query": question})
          print(cb)
        result = answer["result"].replace("\n", "").replace("Answer:", "")
        sources = answer['source_documents']
        print("-" * 150, "\n")
        print(f"Question: {question}")
        print(f"Answer: {result}\n")

        # Optionally print sources for debugging or transparency
        print(f"Sources:")
        for idx, source in enumerate(sources):
            source_wrapped = tr.fill(str(source.page_content), width=150)
            print(f"{idx+1}: {source_wrapped}\n")



# Example usage
document_path = "text.txt"
# questions = [
#     """
#     Question: "குரங்கு முதலில் எதில் சென்றது?"
#     Students Answer:  " நம்ம தலைவன் "
#     """
# ]

questions = [
    """
    Question: "குரங்கு எதைத் தொட்டு பார்க்க"
    Students Answer:  "குரங்கு புச்சிய தொட்டு பார்க்க ஆசைப்பட்டது"

    """
]
retrieve_answers_from_document(document_path, questions)

db.similarity_search_with_score("இயற்கையின் முக்கிய செய்கை என்ன ?")
