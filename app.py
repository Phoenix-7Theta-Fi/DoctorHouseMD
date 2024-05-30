import streamlit as st
from transformers import AutoModelWithLMHead, AutoTokenizer

# Load DialoGPT model and tokenizer
model_name = "microsoft/DialoGPT-medium" 
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelWithLMHead.from_pretrained(model_name)

# Streamlit UI
st.title("AI-Assisted Diagnostic Bot (Demo with DialoGPT)")

# Chat History
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "bot", "content": "Hello!  I can try to understand your symptoms, but I am not a doctor.  Please describe your health concerns."}
    ]

# Chatbot Interaction Area
chat_placeholder = st.empty()
user_input = st.text_input("Patient: ", key="user_input")

# Diagnosis Display Area
st.sidebar.title("Possible Diagnosis")
diagnosis_area = st.sidebar.empty()  

# Treatment Framework Display Area
st.sidebar.title("General Guidance")
treatment_area = st.sidebar.empty() 

def generate_response(user_input):
    new_user_input = "Patient: " + user_input + " Doctor: "  # DialoGPT expects a dialogue format

    # Encode the new input and add it to the conversation history
    bot_input_ids = tokenizer.encode(new_user_input + tokenizer.eos_token, return_tensors='pt')

    # Generate a response 
    chat_history_ids = model.generate(
        bot_input_ids,
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=50, 
        top_p=0.95,
    )

    # Decode the output 
    response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], ignore_special_tokens=True)
    return response

def get_diagnosis(conversation_history):
    # TODO: For a real application, you'd process the conversation using medical knowledge
    # For this demo, we'll keep it very simple
    for message in conversation_history:
        if "headache" in message['content'].lower():
            return "Possible Headache (See a doctor for proper diagnosis)"
        if "fever" in message['content'].lower():
            return "Possible Fever (Consult a medical professional)" 
    return "Not enough information. Please provide more details about your symptoms."

def get_treatment(diagnosis):
    # TODO: Implement treatment suggestions based on diagnosis (use with caution!)
    # This is a placeholder - do not use for real medical advice
    if "Headache" in diagnosis:
        return "General advice: Rest, hydration.  See a doctor if severe or persistent."
    if "Fever" in diagnosis:
        return "General advice: Rest, fluids.  Consult a doctor for diagnosis and treatment." 
    return "Please consult a medical professional for personalized advice."

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    response = generate_response(user_input)
    st.session_state.messages.append({"role": "bot", "content": response})

    # Update the chat history 
    chat_placeholder.text_area("Chat History:", value="\n\n".join([f"{m['role'].upper()}: {m['content']}" for m in st.session_state.messages]), height=400)

    # Provide a possible diagnosis and general guidance based on the conversation history
    diagnosis = get_diagnosis(st.session_state.messages)
    treatment = get_treatment(diagnosis)
    diagnosis_area.write(diagnosis)
    treatment_area.write(treatment)
