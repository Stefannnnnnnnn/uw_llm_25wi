# Streamlit App on Hugging Face Spaces

## Overview

This guide will walk you through the steps to showcase your Streamlit app on Hugging Face Spaces. Hugging Face Spaces is primarily designed for sharing machine learning models, but you can use it to link to your hosted Streamlit app.

### Prerequisites
- Developed and tested your Streamlit app locally or Google Colab.

## Steps

1. **Create your Streamlit App:**

   Develop your Streamlit app locally or Google Colab. Ensure it works as expected on your machine.


2. **Prepare Your Streamlit App for Sharing:**

   Make sure necessary data, dependencies, and configuration files are included in your deployment package.
   Add all necessary libraries to your `requirements.txt` file. 


5. **Create or Update your Hugging Face Space:**
    To make a new Space, visit the [Spaces](https://huggingface.co/spaces) main page after logging into HuggingFace and click on Create new Space. Along with choosing a name for your Space, selecting an optional license, and setting your Space’s visibility choose Public, you’ll be prompted to choose the SDK for your Space. The Hub offers four SDK options: Gradio, Streamlit, Docker and static HTML. Please select “Streamlit” as your SDK option.

6. **Edit the Description:**

   In the description of your Hugging Face Space, provide information about your Streamlit app.

7. **Upload your Streamlit App:**

   Upload your Streamlit app to your Hugging Face Space. You can upload your app by clicking on the Upload button on the top right corner of your Space. You can also drag and drop your app into the Space.

### OR
6. **Clone the Repository:**

    You can easily clone your Space repo locally. Start by clicking on the dropdown menu in the top right of your Space page

For more information, please visit the [Hugging Face Spaces](https://huggingface.co/docs/hub/spaces-overview) documentation or refer to this [youtube video](https://www.youtube.com/watch?v=3bSVKNKb_PY).