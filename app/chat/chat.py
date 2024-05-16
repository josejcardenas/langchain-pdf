from app.chat.models import ChatArgs
from app.chat.vector_stores import retriever_map
from app.chat.llms.chatopenai import build_llm
from app.chat.memories.sql_memory import build_memory
from app.chat.chains.retrieval import StreamingConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
import random
from app.web.api import (set_conversation_components, get_conversation_components)

def build_chat(chat_args: ChatArgs):
    """
    :param chat_args: ChatArgs object containing
        conversation_id, pdf_id, metadata, and streaming flag.

    :return: A chain

    Example Usage:

        chain = build_chat(chat_args)
    """

    components = get_conversation_components(
        chat_args.conversation_id
    )
    previous_retriever = components["retriever"]
    if previous_retriever:
        # this is not the first message of the conversation
        # and I need to use the same retriever again
        build_retriever = retriever_map[previous_retriever]
        retriever = build_retriever(chat_args)
    else:
        # this is the first message of the conversation
        # and I need to pick a random retriever
        random_retriever_name = random.choice(list(retriever_map.keys()))
        build_retriever = retriever_map[random_retriever_name]
        retriever = build_retriever(chat_args)
        set_conversation_components(
            conversation_id=chat_args.conversation_id,
            llm = "",
            memory="",
            retriever=random_retriever_name
        )

    llm = build_llm(chat_args)
    condense_question_llm = ChatOpenAI(streaming=False)
    memory = build_memory(chat_args)

    return StreamingConversationalRetrievalChain.from_llm(
        llm=llm,
        condense_question_llm = condense_question_llm,
        memory=memory,
        retriever=retriever
    )
