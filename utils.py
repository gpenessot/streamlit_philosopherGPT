import openai
import base64
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
#from config import QDRANT_HOST, QDRANT_PORT, QDRANT_API_KEY, OPENAI_API_KEY, COLLECTION_NAME
import streamlit as st

# Set org ID and API key
openai.api_key = st.secret['OPENAI']['openai_api_key']

qdrant_client = QdrantClient(
    url=st.secret['QDRANT']['host'],
    port=st.secret['QDRANT']['port'],
    api_key=st.secret['QDRANT']['qdrant_api_key'],
)

retrieval_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

def build_prompt(question: str, references: list) -> tuple[str, str]:
    prompt = f"""
    Tu es Marc Aurèle, Empereur de Rome. Tu donnes un conseil à ton ami qui t'a posé la question suivante: '{question}'
    
    Tu as sélectionné les passages les plus pertinents de tes écrits pour les utiliser comme source de ta réponse. Citez-les dans votre réponse.

    Références:
    """.strip()

    references_text = ""

    for i, reference in enumerate(references, start=1):
        text = reference.payload["text"].strip()
        references_text += f"\n[{i}]: {text}"

    prompt += (
        references_text
        + "\nComment citer une référence : Ceci est une citation [1]. Celle-ci aussi [2]. Et voici une phrase avec de nombreuses citations [3][4].\nRéponse :"
    )
    return prompt, references_text

def ask(question: str):
    similar_docs = qdrant_client.search(
        collection_name=COLLECTION_NAME,
        query_vector=retrieval_model.encode(question),
        limit=3,
        append_payload=True,
    )

    prompt, references = build_prompt(question, similar_docs)

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt},
        ],
        max_tokens=250,
        temperature=0.2,
    )

    return {
        "response": response["choices"][0]["message"]["content"],
        "references": references,
    }
    
def generate_response(question, model):
    ###
    similar_docs = qdrant_client.search(
        collection_name=COLLECTION_NAME,
        query_vector=retrieval_model.encode(question),
        limit=3,
        append_payload=True,
    )

    prompt, references = build_prompt(question, similar_docs)
    ###
    st.session_state['messages'].append({"role": "user", "content": prompt})
    completion = openai.ChatCompletion.create(
        model=model,
        messages=st.session_state['messages']
    )
    response = completion.choices[0].message.content
    st.session_state['messages'].append({"role": "assistant", "content": response})

    # print(st.session_state['messages'])
    total_tokens = completion.usage.total_tokens
    prompt_tokens = completion.usage.prompt_tokens
    completion_tokens = completion.usage.completion_tokens
    return response, references, total_tokens, prompt_tokens, completion_tokens
    
def set_background(main_bg):
    '''
    A function to unpack an image from root folder and set as bg.
 
    Returns
    -------
    The background.
    '''
    # set bg name
    main_bg_ext = "png"
        
    st.markdown(
         f"""
         <style>
         .stApp {{
             background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()});
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )
