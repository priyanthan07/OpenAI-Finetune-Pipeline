import streamlit as st
import time
from src.synthetic_data_generation import data_generation
from src.upload_data import upload
from src.finetune_process import finetune_func
from src.check_progress import progress
from src.cancel_process import cancel_process
from src.delete_model import delete_model
from src.data_preprocess import preprocess
from src.finetuned_models import get_total_finetuned_model_count
from src.inference import Inference_func


st.title("🍁 OPENAI - FINETUNE 🍁")
st.subheader("Fine-tune an OpenAI model for a chatbot")

if "context" not in st.session_state:
    st.session_state.context = ""

if "user" not in st.session_state:
    st.session_state.user = ""

if "assistant" not in st.session_state:
    st.session_state.assistant = ""

if "train_id" not in st.session_state:
    st.session_state.train_id = ""

if "test_id" not in st.session_state:
    st.session_state.test_id = ""

if "finetune_id" not in st.session_state:
    st.session_state.finetune_id = ""

if "finetuned_model_name" not in st.session_state:
    st.session_state.finetuned_model_name = ""

if "train_data" not in st.session_state:
    st.session_state.train_data = []

if "test_data" not in st.session_state:
    st.session_state.test_data = []


with st.sidebar:
    st.title(" INPUTS")
    st.session_state.context = st.text_input("Context")
    prompt  = st.text_input("Prompt")
    st.session_state.user = st.text_input("User")
    st.session_state.assistant = st.text_input("Assistant")

    count = st.text_input("number of conversations")

    if st.button("synthetic data generation"):
        with st.spinner("Processing..."):
            st.session_state.synthetic_data = data_generation(st.session_state.context, prompt, int(count), st.session_state.assistant, st.session_state.user)
            if st.session_state.synthetic_data:
                st.success("synthetic data generated")

st.write("")
if st.button("PREPROCESS "):
    with st.spinner("Processing..."):
        train_dataset, test_dataset = preprocess(st.session_state.context, st.session_state.user, st.session_state.assistant)
        if train_dataset and test_dataset:
            st.session_state.train_data = train_dataset
            st.session_state.test_data = test_dataset
            st.success("preprocessed successfully")


if st.button("UPLOAD"):
    with st.spinner("Processing..."):
        train_id, test_id = upload(st.session_state.train_data, st.session_state.test_data)
        if train_id and test_id:
            st.session_state.train_id = train_id
            st.session_state.test_id = test_id
            st.success("uploaded successfully")


if st.button("FINE-TUNE"):
    with st.spinner("Processing..."):
        finetune_id = finetune_func(st.session_state.train_id , st.session_state.test_id )
        if finetune_id:
            st.session_state.finetune_id = finetune_id
            st.success("finetuning job started")


if st.button("CHECK-PROGRESS"):
    with st.spinner("Processing..."):
        info = progress(st.session_state.finetune_id)
        if info['Status']== "succeeded":
            st.session_state.finetuned_model_name = info["Fine Tuned Model"]
            st.success("finetuned successfully")

        else: 
            st.success("still finetuning...")

if st.button("CANCEL-PROCESS"):
    with st.spinner("Processing..."):
        status = cancel_process(st.session_state.finetune_id)
        if status:
            st.success("canceled successfully")

user_message = st.text_input("USER MESSAGE:")
if st.button("TEST-MODEL"):
    with st.spinner("Processing..."):
        model_response = Inference_func(st.session_state.context, user_message, st.session_state.finetuned_model_name)
        if model_response:
            st.write(model_response)

        
st.write("")
st.markdown("**Delete finetuned models**" )

if st.button("LIST-FINETUNED-MODELS"):
    with st.spinner("Processing..."):
        model_list = get_total_finetuned_model_count()
        if ~model_list.empty:
            st.write(model_list)


model_name = st.text_input("Model name that need to be deleted :")
if st.button("DELETE-MODEL"):
    with st.spinner("Processing..."):
        status = delete_model(model_name)
        if status:
            st.success(f"successfully deleted the model : {model_name}")