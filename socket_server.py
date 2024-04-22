from openai import OpenAI
from fastapi import FastAPI, WebSocket
from RAG_config import ensemble_retriever


RAG_SELF_REFINE_PROMPT = """
Dưới đây là các nội dung được trích xuất từ Database:
---CONTEXT---
{context}
---END CONTEXT---
Nhiệm vụ của bạn là trả lời câu hỏi dựa trên các nội dung trên, hãy loại bỏ các nội dung không liên quan để tránh gây nhiễu loạn. 
Nếu thư thông tin đưa cho không liên quan đến câu hỏi, vui lòng trả lời "Xin lỗi, vấn đề này không nằm trong phạm vi kiến thức của tôi."
Câu hỏi: {question}
---
Trả lời:
"""

RAG_SELF_REFINE_PROMPT_HISTORY = """
Dưới đây là các nội dung được trích xuất từ Database:
---CONTEXT---
{context}
---END CONTEXT---
Nhiệm vụ của bạn là trả lời câu hỏi dựa trên các nội dung trên, hãy loại bỏ các nội dung không liên quan để tránh gây nhiễu loạn. 
Nếu thư thông tin đưa cho không liên quan đến câu hỏi, vui lòng trả lời "Xin lỗi, vấn đề này không nằm trong phạm vi kiến thức của tôi."
Chat History:
{history}
---
Câu hỏi: {question}
---
Trả lời:
"""


llm = OpenAI(api_key="")

def build_final_context(chunks):
    context = ""
    for index, chunk in enumerate(chunks):
        context +=  f"Context {index + 1}: \n" + chunk + "\n"
    return context

HISTORY = {}
def build_history(session_id):
    history_str = ""
    for human, assistant in HISTORY[session_id]:
        history_str += f"Human: {human}\nAssistant: {assistant}\n"
    return history_str




def gen_prompt(message, session_id):
    history_openai_format = []
    if session_id not in HISTORY or len(HISTORY[session_id]) == 0:
        docs = ensemble_retriever.get_relevant_documents(message)
        retrievals = []
        for doc in docs:
            retrievals.append(doc.page_content)
        contexts = build_final_context(retrievals)
        history_openai_format.append({"role": "user", "content": RAG_SELF_REFINE_PROMPT.format(context=contexts, question=message)})
    else:
        history_str = build_history(session_id)
        docs = ensemble_retriever.get_relevant_documents(message)
        for doc in docs:
            retrievals.append(doc.page_content)
        contexts = build_final_context(retrievals)
        history_openai_format.append({"role": "user", "content": RAG_SELF_REFINE_PROMPT_HISTORY.format(context=contexts, history=history_str, question=message) })
        return history_openai_format

app = FastAPI()
session_id = 0
@app.websocket("/")
async def handle_client(websocket: WebSocket):
    global session_id
    session_id += 1
    await websocket.accept()
    HISTORY[session_id] = []
    while True:  # Loop indefinitely to keep the WebSocket connection open
        message = await websocket.receive_text()
        history_openai_format = gen_prompt(message, session_id)
        response = llm.chat.completions.create(model='gpt-3.5-turbo',
            messages= history_openai_format,
            temperature=1.0,
            stream=True)
        assistant_response = ""
        for chunk in response: 
            if chunk.choices[0].delta.content is not None:
                assistant_response = assistant_response + chunk.choices[0].delta.content
                await websocket.send_text(chunk.choices[0].delta.content)
        HISTORY[session_id].append((message, assistant_response))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8765)
        
                







def predict(message, session_id):
    history_openai_format = []
    if session_id in HISTORY:
        for human, assistant in HISTORY[session_id]:
            history_openai_format.append({"role": "user", "content": human })
            history_openai_format.append({"role": "assistant", "content": assistant})

        # Retrieve relevant documents/products
        docs = ensemble_retriever.get_relevant_documents(message)
        retrievals = []
        # Extract and print only the page content from each document
        for doc in docs:
            retrievals.append(doc.page_content)

        contexts = build_final_context(retrievals)

        history_openai_format.append({"role": "user", "content": RAG_SELF_REFINE_PROMPT.format(context=contexts, question=message)})
    else:
        # Retrieve relevant documents/products
        docs = ensemble_retriever.get_relevant_documents(message)
        retrievals = []
        # Extract and print only the page content from each document
        for doc in docs:
            retrievals.append(doc.page_content)

        contexts = build_final_context(retrievals)
            
        history_openai_format.append({"role": "user", "content": RAG_SELF_REFINE_PROMPT_HISTORY.format(context=contexts, history=history_str, question=message) })

    return history_openai_format

