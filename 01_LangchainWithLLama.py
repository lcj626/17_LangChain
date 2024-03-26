from langchain_community.llms import Ollama
llm = Ollama(model="llama2")

response=llm.invoke("how can langsmith help with testing?")
print(response)

# from langchain_core.prompts import ChatPromptTemplate
# prompt = ChatPromptTemplate.from_messages([
#     ("system", "You are world class technical documentation writer."),
#     ("user", "{input}")
# ])

# chain = prompt | llm


# 프로그램의 실행 단위는 mark로 표시하였다.
# 기본적인 langchain 개념을 학습하기 위한 챕터이다.

# 아래 코드의 노란색 줄은 해당 모듈을 찾을 수 없을 때 생긴다.
# conda install langchain -c conda-forge 설치가 되었다면 vs code의 인터프리터를 수정한다.
# cnt + shift + p => python interprint => 가상환경 선택

# 모델을 가져와 응답을 할 수 있도록 설정해준다.
from langchain_community.llms import Ollama
llm = Ollama(model="llama2")

# mark 1 
# 모델 질문을 한다.
# result = llm.invoke("how can langsmith help with testing?")
# print(result)

# makr1의 내용을 주석으로 처리한다.

# explanation
# 위 질문 과정에서 정확한 답변을 하지 않아야 한다.
# 사전에 langchain이라는 것이 무엇인지 학습이 완료되지 않았기 때문이다.



# 항상 전문 기술 작성자 입장에서 문서를 만들어 주도록 프롬프트를 설정한다.
from langchain_core.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are world class technical documentation writer."),
    ("user", "{input}")
])

#mark2 
# 모델과 프롬프트를 결합하여 질문 요청시 항상 요청 내용에 맞춰서 답변을 할 수 있도록 한다.
# chain = prompt | llm
#result = chain.invoke({"input": "how can langsmith help with testing?"})
#print(result)



# 출력 파서는 llm 및 chat Model의 출력을 보다 구조화된 데이터로 변환하는 역할을 담당한다.
from langchain_core.output_parsers import StrOutputParser
output_parser = StrOutputParser()

# mark3
# chain = prompt | llm | output_parser
# result = chain.invoke({"input": "how can langsmith help with testing?"})
# print(result)


# 검색 체인 연결하기
# 문서 검색의 경우 gpt가 과거 학습 데이터를 기반으로 요청을 하기 때문에
# 많은 양의 데이터를 한번에 전달하여 답변을 요구하는 경우 유용한 설정이다.
from langchain_community.document_loaders import WebBaseLoader

# 해당 url의 내용을 분석하는 로직
loader = WebBaseLoader("https://docs.smith.langchain.com")
docs = loader.load()

# 분석된 내용을 벡터스토어로 인덱싱하고 그를 통해 답변을 할 수 있도록 한다.
# 이를 위해서는 임베딩 모델과 백터 데이터 베이스가 필요하다.
# 임베딩 모델 추가
from langchain_community.embeddings import OllamaEmbeddings
embeddings = OllamaEmbeddings()

# 간단한 벡터값 저장을 위한 패키지 설치
# pip install faiss-cpu
# 위 설치시 파이썬이 3.12버전 이상에서 오류가 발생되는 경우가 있으며
# 해당 오류는 3.12버전에 지원하는 siwg 모듈이 없어서 발생하는 오류로 
# python을 3.11버전으로 다운그레이드한다. 


# 아래의 설정은 faiss의 백터 저장소를 설정하고 사용자의 요청 url 문서를 백터 값에 저장한다.
# 이후 사용자의 요청을 해당 저장소에서 찾아 llm에 전달하고 질문에 답변할 수 있도록 하는 설정이다.
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)
vector = FAISS.from_documents(documents, embeddings)


from langchain.chains.combine_documents import create_stuff_documents_chain

prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:
<context>
{context}
</context>
Question: {input}""")

document_chain = create_stuff_documents_chain(llm, prompt)

from langchain.chains import create_retrieval_chain
retriever = vector.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# mark4 
# 사용자가 지정한 문서를 기준으로 질문을 생성하고 응답을 할 수 있도록 요청을 설정함
# response = retrieval_chain.invoke({"input": "how can langsmith help with testing?"})
# print('first question : ')
# print(response["answer"])
# response = retrieval_chain.invoke({"input": "What question did I just ask?"})

# print('second question : ')
# print(response["answer"])


# 현재까지 설정한 작업은 새로운 문서를 추가하여 문서의 내용을 기반으로 답변할 수 있으나 
# 과거의 내용을 기억하지 못하는 단점이 있으며 이를 극복하기 위해서 다음 작업을 추가하여
# 현재까지의 내용을 기억할 수 있도록 한다.
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
])

retriever_chain = create_history_aware_retriever(llm, retriever, prompt)


from langchain_core.messages import HumanMessage, AIMessage

chat_history = [HumanMessage(content="Can LangSmith help test my LLM applications?"), AIMessage(content="Yes!")]
response =retriever_chain.invoke({
    "chat_history": chat_history,
    "input": "Just tell me briefly how to do it"
})

# mark5
# 대화의 내용은 기억하고 있지만 내화 내용을 기반으로 답변을 하고 있지는 않는다.
# print("first question")
# print(response)
# response = retrieval_chain.invoke({
#     "chat_history": chat_history,
#     "input": "What question did I just ask?"
# })
# print("second question :")
# print(response)


prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the user's questions based on the below context:\n\n{context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
])
document_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever_chain, document_chain)
chat_history = [HumanMessage(content="Can LangSmith help test my LLM applications?"), AIMessage(content="Yes!")]
response= retrieval_chain.invoke({
    "chat_history": chat_history,
    "input": "Tell me how"
})

# mark6
# 응답을 확인하면 성공적으로 이전의 대화 내용을 기억하고 그에 대한 답변을 하는 것을 볼 수 있다.
print("first question : ")
print(response)

response= retrieval_chain.invoke({
    "chat_history": chat_history,
    "input": "what question did i just now?"
})

print("second question : ")
print(response)

# import os
# os.environ["OPENAL_APU_KEY"] = "YOUR_API_KEY"  : 별도로 opanai 키 받아와서 입력