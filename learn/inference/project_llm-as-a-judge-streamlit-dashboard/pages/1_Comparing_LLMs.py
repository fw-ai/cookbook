from dotenv import load_dotenv
import os
from PIL import Image
import streamlit as st
import fireworks.client

st.set_page_config(page_title="LLM Comparison Tool", page_icon="🎇")
st.title("LLM-as-a-judge: Comparing LLMs using Fireworks")
st.write("A light introduction to how easy it is to swap LLMs and how to use the Fireworks Python client")

# Clear the cache before starting
st.cache_data.clear()

# Specify the path to the .env file in the env/ directory
dotenv_path = os.path.join(os.path.dirname(__file__), '..','..', 'env', '.env')

# Load the .env file from the specified path
load_dotenv(dotenv_path)

# Get the Fireworks API key from the environment variable
fireworks_api_key = os.getenv("FIREWORKS_API_KEY")

if not fireworks_api_key:
    raise ValueError("No API key found in the .env file. Please add your FIREWORKS_API_KEY to the .env file.")

# Load the image
logo_image = Image.open("img/fireworksai_logo.png")
ash_image = Image.open("img/ash.png")
bulbasaur_image = Image.open("img/bulbasaur.png")
squirtel_image = Image.open("img/squirtel.png")
charmander_image = Image.open("img/charmander.png")

st.divider()
# Streamlit app
st.subheader("Fireworks Playground")

st.write("Fireworks AI is a platform that offers serverless and scalable AI models.")
st.write("👉 Learn more here: [Fireworks Serverless Models](https://fireworks.ai/models?show=Serverless)")
st.divider()

# Sidebar for selecting models
with st.sidebar:
    st.image(logo_image)

    st.write("Select three models to compare their outputs:")

    st.image(bulbasaur_image, width=80)
    option_1 = st.selectbox("Select Model 1", [
        "Text: Meta Llama 3.1 Instruct - 70B",
        "Text: Meta Llama 3.1 Instruct - 8B",
        "Text: Meta Llama 3.2 Instruct - 3B",
        "Text: Gemma 2 Instruct - 9B",
        "Text: Mixtral MoE Instruct - 8x22B",
        "Text: Mixtral MoE Instruct - 8x7B",
        "Text: MythoMax L2 - 13B"
    ], index=2)  # Default to Meta Llama 3.2 Instruct - 3B

    st.image(charmander_image, width=80)
    option_2 = st.selectbox("Select Model 2", [
        "Text: Meta Llama 3.1 Instruct - 70B",
        "Text: Meta Llama 3.1 Instruct - 8B",
        "Text: Meta Llama 3.2 Instruct - 3B",
        "Text: Gemma 2 Instruct - 9B",
        "Text: Mixtral MoE Instruct - 8x22B",
        "Text: Mixtral MoE Instruct - 8x7B",
        "Text: MythoMax L2 - 13B"
    ], index=5)  # Default to Mixtral MoE Instruct - 8x7B

    st.image(squirtel_image, width=80)
    option_3 = st.selectbox("Select Model 3", [
        "Text: Meta Llama 3.1 Instruct - 70B",
        "Text: Meta Llama 3.1 Instruct - 8B",
        "Text: Meta Llama 3.2 Instruct - 3B",
        "Text: Gemma 2 Instruct - 9B",
        "Text: Mixtral MoE Instruct - 8x22B",
        "Text: Mixtral MoE Instruct - 8x7B",
        "Text: MythoMax L2 - 13B"
    ], index=0)  # Default to Gemma 2 Instruct - 9B

    # Dropdown to select the LLM that will perform the comparison
    st.image(ash_image, width=80)
    comparison_llm = st.selectbox("Select Comparison Model", [
        "Text: Meta Llama 3.1 Instruct - 70B",
        "Text: Meta Llama 3.1 Instruct - 8B",
        "Text: Meta Llama 3.2 Instruct - 3B",
        "Text: Gemma 2 Instruct - 9B",
        "Text: Mixtral MoE Instruct - 8x22B",
        "Text: Mixtral MoE Instruct - 8x7B",
        "Text: MythoMax L2 - 13B"
    ], index=5) # Default to MythoMax L2 - 13B

os.environ["FIREWORKS_API_KEY"] = fireworks_api_key

# Helper text for the prompt
st.markdown("### Enter your prompt below to generate responses:")

prompt = st.text_input("Prompt", label_visibility="collapsed")
st.divider()

# Function to generate a response from a text model
def generate_text_response(model_name, prompt):
    return fireworks.client.ChatCompletion.create(
        model=model_name,
        messages=[{
            "role": "user",
            "content": prompt,
        }]
    )

# Function to compare the three responses using the selected LLM
def compare_responses(response_1, response_2, response_3, comparison_model):
    comparison_prompt = f"Compare the following three responses:\n\nResponse 1: {response_1}\n\nResponse 2: {response_2}\n\nResponse 3: {response_3}\n\nProvide a succinct comparison."
    
    comparison_response = fireworks.client.ChatCompletion.create(
        model=comparison_model,  # Use the selected LLM for comparison
        messages=[{
            "role": "user",
            "content": comparison_prompt,
        }]
    )
    
    return comparison_response.choices[0].message.content


# If Generate button is clicked
if st.button("Generate"):
    if not fireworks_api_key.strip() or not prompt.strip():
        st.error("Please provide the missing fields.")
    else:
        try:
            with st.spinner("Please wait..."):
                fireworks.client.api_key = fireworks_api_key
                
                # Create three columns for side-by-side comparison
                col1, col2, col3 = st.columns(3)
                
                # Model 1
                with col1:
                    st.subheader(f"Model 1: {option_1}")
                    st.image(bulbasaur_image)
                    if option_1.startswith("Text"):
                        model_map = {
                            "Text: Meta Llama 3.1 Instruct - 70B": "accounts/fireworks/models/llama-v3p1-70b-instruct",
                            "Text: Meta Llama 3.1 Instruct - 8B": "accounts/fireworks/models/llama-v3p1-8b-instruct",
                            "Text: Meta Llama 3.2 Instruct - 3B": "accounts/fireworks/models/llama-v3p2-3b-instruct",
                            "Text: Gemma 2 Instruct - 9B": "accounts/fireworks/models/gemma2-9b-it",
                            "Text: Mixtral MoE Instruct - 8x22B": "accounts/fireworks/models/mixtral-8x22b-instruct",
                            "Text: Mixtral MoE Instruct - 8x7B": "accounts/fireworks/models/mixtral-8x7b-instruct",
                            "Text: MythoMax L2 - 13B": "accounts/fireworks/models/mythomax-l2-13b"
                        }
                        response_1 = generate_text_response(model_map[option_1], prompt)
                        st.success(response_1.choices[0].message.content)
                
                # Model 2
                with col2:
                    st.subheader(f"Model 2: {option_2}")
                    st.image(charmander_image)
                    response_2 = generate_text_response(model_map[option_2], prompt)
                    st.success(response_2.choices[0].message.content)
                
                # Model 3
                with col3:
                    st.subheader(f"Model 3: {option_3}")
                    st.image(squirtel_image)
                    response_3 = generate_text_response(model_map[option_3], prompt)
                    st.success(response_3.choices[0].message.content)

                # Visual divider between model responses and comparison
                st.divider()

                # Generate a comparison of the three responses using the selected LLM
                comparison = compare_responses(
                    response_1.choices[0].message.content, 
                    response_2.choices[0].message.content, 
                    response_3.choices[0].message.content, 
                    model_map[comparison_llm]
                )
                
                # Display the comparison
                st.subheader("Comparison of the Three Responses:")
                st.image(ash_image)
                st.write(comparison)
                
        except Exception as e:
            st.exception(f"Exception: {e}")
