from dotenv import load_dotenv
import os
from PIL import Image
import random
import streamlit as st
import fireworks.client

# Set page configuration
st.set_page_config(page_title="LLM Parameters Comparison", page_icon="🎇")
st.title("Understanding the Completions Chat API parameters")
st.write("Compare LLM responses with different sets of parameters and evaluate the results using an LLM-as-a-judge.")
st.markdown("Check out our [Chat Completions API Documentation](https://docs.fireworks.ai/api-reference/post-chatcompletions) for more information on the parameters.")

# Add expandable section for parameter descriptions
with st.expander("Parameter Descriptions", expanded=False):
    st.markdown("""
    **Max Tokens**: Maximum number of tokens the model can generate.<br>
    **Prompt Truncate Length**: Number of tokens from the input prompt considered.<br>
    **Temperature**: Controls randomness of the output.<br>
    **Top-p (Nucleus Sampling)**: Cumulative probability of token selection.<br>
    **Top-k**: Limits the number of tokens sampled.<br>
    **Frequency Penalty**: Discourages repeated words or phrases.<br>
    **Presence Penalty**: Encourages new topics.<br>
    **Stop Sequence**: Defines when to stop generating tokens.
    """, unsafe_allow_html=True)

# Load environment variables
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '..', 'env', '.env')
load_dotenv(dotenv_path)

# Get the Fireworks API key from environment variables
fireworks_api_key = os.getenv("FIREWORKS_API_KEY")
if not fireworks_api_key:
    raise ValueError("No API key found in the .env file. Please add your FIREWORKS_API_KEY to the .env file.")

os.environ["FIREWORKS_API_KEY"] = fireworks_api_key

# Load the images
logo_image = Image.open("img/fireworksai_logo.png")
bulbasaur_image = Image.open("img/bulbasaur.png")
charmander_image = Image.open("img/charmander.png")
squirtel_image = Image.open("img/squirtel.png")
ash_image = Image.open("img/ash.png")

# Map models to their respective identifiers
model_map = {
    "Text: Meta Llama 3.1 Instruct - 70B": "accounts/fireworks/models/llama-v3p1-70b-instruct",
    "Text: Meta Llama 3.1 Instruct - 8B": "accounts/fireworks/models/llama-v3p1-8b-instruct",
    "Text: Meta Llama 3.2 Instruct - 3B": "accounts/fireworks/models/llama-v3p2-3b-instruct",
    "Text: Gemma 2 Instruct - 9B": "accounts/fireworks/models/gemma2-9b-it",
    "Text: Mixtral MoE Instruct - 8x22B": "accounts/fireworks/models/mixtral-8x22b-instruct",
    "Text: Mixtral MoE Instruct - 8x7B": "accounts/fireworks/models/mixtral-8x7b-instruct",
    "Text: MythoMax L2 - 13B": "accounts/fireworks/models/mythomax-l2-13b"
}

# Function to generate a response from a text model with parameters
def generate_text_response(model_name, prompt, params):
    return fireworks.client.ChatCompletion.create(
        model=model_name,
        messages=[{
            "role": "user",
            "content": prompt,
        }],
        max_tokens=params["max_tokens"],
        temperature=params["temperature"],
        top_p=params["top_p"],
        top_k=params["top_k"],
        frequency_penalty=params["frequency_penalty"],
        presence_penalty=params["presence_penalty"],
        stop=params["stop"]
    )

# Function to compare the three responses using the selected LLM
def compare_responses(response_1, response_2, response_3, comparison_model):
    comparison_prompt = f"Compare the following three responses:\n\nResponse 1: {response_1}\n\nResponse 2: {response_2}\n\nResponse 3: {response_3}\n\nProvide a succinct comparison."
    
    comparison_response = fireworks.client.ChatCompletion.create(
        model=comparison_model,
        messages=[{
            "role": "user",
            "content": comparison_prompt,
        }]
    )
    
    return comparison_response.choices[0].message.content

# Slightly randomize parameters for sets 2 and 3
def randomize_params():
    return {
        "max_tokens": random.randint(100, 200),
        "prompt_truncate_len": random.randint(100, 200),
        "temperature": round(random.uniform(0.7, 1.3), 2),
        "top_p": round(random.uniform(0.8, 1.0), 2),
        "top_k": random.randint(30, 70),
        "frequency_penalty": round(random.uniform(0, 1), 2),
        "presence_penalty": round(random.uniform(0, 1), 2),
        "n": 1,
        "stop": None
    }

# Sidebar for LLM selection, prompt, and judge LLM
with st.sidebar:
    st.image(logo_image)

    # Select the model for generating responses
    st.subheader("Select LLM for Generating Responses")
    model = st.selectbox("Select a model for generating responses:", [
        "Text: Meta Llama 3.1 Instruct - 70B",
        "Text: Meta Llama 3.1 Instruct - 8B",
        "Text: Meta Llama 3.2 Instruct - 3B",
        "Text: Gemma 2 Instruct - 9B",
        "Text: Mixtral MoE Instruct - 8x22B",
        "Text: Mixtral MoE Instruct - 8x7B",
        "Text: MythoMax L2 - 13B"
    ], index=2)

    # Placeholder prompts
    suggested_prompts = [
        "Prompt 1: Describe the future of AI.",
        "Prompt 2: Write a short story about a cat who becomes the mayor of a small town",
        "Prompt 3: Write a step-by-step guide to making pancakes from scratch.",
        "Prompt 4: Generate a grocery list and meal plan for a vegetarian family of four for one week.",
        "Prompt 5: Generate a story in which a time traveler goes back to Ancient Greece, accidentally introduces modern memes to philosophers like Socrates and Plato, and causes chaos in the philosophical discourse.",
        "Prompt 6: Create a timeline where dinosaurs never went extinct and developed their own civilizations, and describe their technology and cultural achievements in the year 2024.",
        "Prompt 7: Explain the concept of Gödel’s incompleteness theorems in the form of a Dr. Seuss poem, using at least 10 distinct rhyme schemes."
    ]

    # Selectbox for suggested prompts
    selected_prompt = st.selectbox("Choose a suggested prompt:", suggested_prompts)

    # Input box where the user can edit the selected prompt or enter a custom one
    prompt = st.text_input("Prompt", value=selected_prompt)

    # Select the LLM for judging the responses
    st.subheader("Select LLM for Judge")
    judge_llm = st.selectbox("Select a model to act as the judge:", [
        "Text: Meta Llama 3.1 Instruct - 70B",
        "Text: Meta Llama 3.1 Instruct - 8B",
        "Text: Meta Llama 3.2 Instruct - 3B",
        "Text: Gemma 2 Instruct - 9B",
        "Text: Mixtral MoE Instruct - 8x22B",
        "Text: Mixtral MoE Instruct - 8x7B",
        "Text: MythoMax L2 - 13B"
    ], index=2)

# Create three columns for parameter sets side-by-side
col1, col2, col3 = st.columns(3)

# Parameters for Output 1 (Bulbasaur image)
with col1:
    st.subheader("Parameter Set #1")
    st.image(bulbasaur_image, width=100)  # Bulbasaur image
    max_tokens_1 = st.slider("Max Tokens", 50, 1000, 123)
    prompt_truncate_len_1 = st.slider("Prompt Truncate Length", 50, 200, 123)
    temperature_1 = st.slider("Temperature", 0.1, 2.0, 1.0)
    top_p_1 = st.slider("Top-p", 0.0, 1.0, 1.0)
    top_k_1 = st.slider("Top-k", 0, 100, 50)
    frequency_penalty_1 = st.slider("Frequency Penalty", 0.0, 2.0, 0.0)
    presence_penalty_1 = st.slider("Presence Penalty", 0.0, 2.0, 0.0)
    stop_1 = st.text_input("Stop Sequence", "")

    params_1 = {
        "max_tokens": max_tokens_1,
        "prompt_truncate_len": prompt_truncate_len_1,
        "temperature": temperature_1,
        "top_p": top_p_1,
        "top_k": top_k_1,
        "frequency_penalty": frequency_penalty_1,
        "presence_penalty": presence_penalty_1,
        "n": 1,
        "stop": stop_1 if stop_1 else None
    }

# Parameters for Output 2 (Charmander image)
with col2:
    st.subheader("Parameter Set #2")
    st.image(charmander_image, width=100)  # Charmander image
    use_random_2 = st.checkbox("Randomize parameters for Output 2", value=True)
    if use_random_2:
        params_2 = randomize_params()
        st.write("**Random Parameters for Output 2:**")
        st.json(params_2)  # Display random params
    else:
        max_tokens_2 = st.slider("Max Tokens (Output 2)", 50, 1000, 150)
        prompt_truncate_len_2 = st.slider("Prompt Truncate Length (Output 2)", 50, 200, 150)
        temperature_2 = st.slider("Temperature (Output 2)", 0.1, 2.0, 0.9)
        top_p_2 = st.slider("Top-p (Output 2)", 0.0, 1.0, 0.95)
        top_k_2 = st.slider("Top-k (Output 2)", 0, 100, 45)
        frequency_penalty_2 = st.slider("Frequency Penalty (Output 2)", 0.0, 2.0, 0.1)
        presence_penalty_2 = st.slider("Presence Penalty (Output 2)", 0.0, 2.0, 0.1)
        stop_2 = st.text_input("Stop Sequence (Output 2)", "")

        params_2 = {
            "max_tokens": max_tokens_2,
            "prompt_truncate_len": prompt_truncate_len_2,
            "temperature": temperature_2,
            "top_p": top_p_2,
            "top_k": top_k_2,
            "frequency_penalty": frequency_penalty_2,
            "presence_penalty": presence_penalty_2,
            "n": 1,
            "stop": stop_2 if stop_2 else None
        }

# Parameters for Output 3 (Squirtle image)
with col3:
    st.subheader("Parameter Set #3")
    st.image(squirtel_image, width=100)  # Squirtle image
    use_random_3 = st.checkbox("Randomize parameters for Output 3", value=True)
    if use_random_3:
        params_3 = randomize_params()
        st.write("**Random Parameters for Output 3:**")
        st.json(params_3)  # Display random params
    else:
        max_tokens_3 = st.slider("Max Tokens (Output 3)", 50, 1000, 180)
        prompt_truncate_len_3 = st.slider("Prompt Truncate Length (Output 3)", 50, 200, 140)
        temperature_3 = st.slider("Temperature (Output 3)", 0.1, 2.0, 1.1)
        top_p_3 = st.slider("Top-p (Output 3)", 0.0, 1.0, 0.85)
        top_k_3 = st.slider("Top-k (Output 3)", 0, 100, 60)
        frequency_penalty_3 = st.slider("Frequency Penalty (Output 3)", 0.0, 2.0, 0.05)
        presence_penalty_3 = st.slider("Presence Penalty (Output 3)", 0.0, 2.0, 0.2)
        stop_3 = st.text_input("Stop Sequence (Output 3)", "")

        params_3 = {
            "max_tokens": max_tokens_3,
            "prompt_truncate_len": prompt_truncate_len_3,
            "temperature": temperature_3,
            "top_p": top_p_3,
            "top_k": top_k_3,
            "frequency_penalty": frequency_penalty_3,
            "presence_penalty": presence_penalty_3,
            "n": 1,
            "stop": stop_3 if stop_3 else None
        }

# Divider above generate button
st.divider()

# Generate button and logic
st.subheader("Just hit play")
st.write("See the effect of selecting parameters on the responses.")


if st.button("Generate"):
    if not fireworks_api_key.strip() or not prompt.strip():
        st.error("Please provide the missing fields.")
    else:
        try:
            with st.spinner("Please wait..."):
                fireworks.client.api_key = fireworks_api_key

                # Generate responses for each set of parameters
                response_1 = generate_text_response(model_map[model], prompt, params_1)
                response_2 = generate_text_response(model_map[model], prompt, params_2)
                response_3 = generate_text_response(model_map[model], prompt, params_3)

                # Display results in the main section
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.subheader("Response 1")
                    st.image(bulbasaur_image, width=100)
                    st.success(response_1.choices[0].message.content)

                with col2:
                    st.subheader("Response 2")
                    st.image(charmander_image, width=100)
                    st.success(response_2.choices[0].message.content)

                with col3:
                    st.subheader("Response 3")
                    st.image(squirtel_image, width=100)
                    st.success(response_3.choices[0].message.content)

                st.divider()

                # Use the selected LLM as the judge and display Ash image
                st.subheader("LLM-as-a-Judge Comparison")
                st.image(ash_image, width=100)
                comparison = compare_responses(
                    response_1.choices[0].message.content,
                    response_2.choices[0].message.content,
                    response_3.choices[0].message.content,
                    model_map[judge_llm]
                )

                st.write(comparison)

        except Exception as e:
            st.exception(f"Exception: {e}")

# Divider below generate button
st.divider()
