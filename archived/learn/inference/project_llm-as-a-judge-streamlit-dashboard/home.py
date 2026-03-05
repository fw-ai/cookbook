import streamlit as st
from PIL import Image

# Load images
logo_image = Image.open("img/fireworksai_logo.png")
bulbasaur_image = Image.open("img/bulbasaur.png")
charmander_image = Image.open("img/charmander.png")
squirtel_image = Image.open("img/squirtel.png")
ash_image = Image.open("img/ash.png")

# Set page configuration
st.set_page_config(page_title="Fireworks Model Comparison App", page_icon="🎇")

# Fireworks Logo at the top
st.image(logo_image)

# Home page title and description
st.title("Fireworks Model Comparison App")

# Introduction with Pokémon image (Bulbasaur)
st.markdown("""
### Welcome to the Fireworks Model Comparison App!""")

st.image(ash_image, width=100)

st.markdown(""" This app allows you to interact with and compare various Large Language Models (LLMs) hosted on **Fireworks AI**. You can select from a range of models, adjust key model parameters, and run comparisons between their outputs. The app also enables you to evaluate results using an **LLM-as-a-judge** to provide an unbiased comparison of responses.""")

# API Documentation Link
st.markdown("""
[Explore Fireworks API Documentation](https://docs.fireworks.ai/api-reference/post-chatcompletions)
""")

# Objectives of the App with Pokémon image (Charmander)

st.markdown("""
---
### Objectives of the App:
- **Compare Different Models**: Select models from Fireworks AI’s hosted collection and compare their outputs.
- **Modify Parameters**: Adjust settings like **Max Tokens**, **Temperature**, and **Sampling** methods to explore how different configurations affect outputs.
- **Evaluate Using LLM-as-a-Judge**: Generate responses and use another LLM to evaluate and provide a comparison.
- **Simple Interface**: The app uses **Streamlit**, making it easy to use, even for those without coding experience.
""")

# How to use the app with Pokémon image (Squirtle)
st.image(squirtel_image, width=100)
st.markdown("""
---
### How to Use the App:
1. **Select a Model**: Use the dropdown menus to choose models for comparison.
2. **Provide a Prompt**: Enter a prompt that the models will use to generate a response.
3. **Adjust Parameters**: Fine-tune the settings for each model to explore how different configurations affect the results.
4. **Generate and Compare**: View the responses from multiple models side-by-side.
5. **Evaluate with LLM-as-a-Judge**: Use another model to compare and judge the outputs.
""")

# Explanation of Other Pages with Pokémon image (Ash)
st.image(bulbasaur_image, width=100)
st.markdown("""
---
### App Sections:
This Streamlit app consists of two key pages that help you interact with the Fireworks AI platform and perform model comparisons.

- **Page 1: Comparing LLMs**
  - On this page, you can compare the outputs of three selected LLMs from Fireworks AI by providing a single prompt.
  - The outputs are displayed side-by-side for easy comparison, and a selected LLM can act as a judge to evaluate the responses.

- **Page 2: Parameter Exploration for LLMs**
  - This page allows you to adjust various parameters like **Max Tokens**, **Temperature**, and **Sampling Methods** for LLMs.
  - You can provide a prompt and see how different parameter configurations affect the output for each model.
  - The LLM-as-a-Judge is also used to compare and evaluate the generated responses.
""")
st.image(charmander_image, width=100)

# Background Information about Fireworks Models with Pokémon image (Bulbasaur again for symmetry)

st.markdown("""
---
### Fireworks AI Models:
Fireworks AI provides access to a variety of Large Language Models (LLMs) that you can query and experiment with, including:

- **Text Models**: These models are designed for tasks such as text generation, completion, and Q&A.
- **Model Parameters**: By adjusting parameters such as temperature, top-p, and top-k, you can influence the behavior of the models and the creativity or focus of their outputs.

For more information, check out the [Fireworks API Documentation](https://docs.fireworks.ai/api-reference/post-chatcompletions) and learn how to query different models using Fireworks' Python Client.
""")
