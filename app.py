import streamlit as st
from fastai.learner import load_learner
from blurr.text.data.all import *
from blurr.text.modeling.all import *
from langdetect import detect
import pathlib
import torch
import matplotlib.pyplot as plt
import pandas as pd
import logging
import re

# import pathlib
# temp = pathlib.PosixPath
# pathlib.PosixPath = pathlib.WindowsPath


# Define custom hash functions
def hash_tensor(x):
    return hash(tuple(x.flatten().tolist()))

def hash_learner(learner):
    hash_list = [hash_tensor(learner.loss_grad), hash(learner.model_dir)]
    return hash(tuple(hash_list))

# Load the learner (cached)
@st.cache_resource(hash_funcs={torch.Tensor: hash_tensor, 'fastai.learner.Learner': hash_learner})
def load_learner_cached(model_path):
    return load_learner(model_path)

model_path = 'models/sw-news-classifier.pkl'
st.write("Warming up the model...")
learner = load_learner_cached(model_path)
st.write("Model is ready!")

# Clear the cache
if st.button("Clear Cache"):
    st.cache_resource.clear()

# Preprocess the input text
def clean_text(sentence):
    sentence = sentence.lower()
    sentence = re.sub('[‘’“”…]', '', sentence)
    sentence = re.sub("[^a-zA-Z.?!]", " ", sentence)
    sentence = re.sub('\s+', ' ', sentence)
    sentence = sentence.strip()
    return sentence

# Set up the Streamlit app
st.header("Swahili News Classifier App")
st.image("images/banner.gif", use_column_width=True)  # Display the GIF
st.subheader(
    """
Classify Swahili news into different categories.
"""
)

# Form to collect news content
my_form = st.form(key="news_form")
news = my_form.text_input("Input your Swahili news content here")
submit = my_form.form_submit_button(label="Classify news content")

if submit:
    # Display spinner while processing
    with st.spinner('Classifying your input...'):
        # Check if the input text is in Swahili
        if detect(news) == "sw":
            # Preprocess the news content
            cleaned_news = clean_text(news)

            dl = learner.dls.test_dl([cleaned_news])
            predictions, *_ = learner.get_preds(dl=dl)

            # Extract the category and probabilities
            predicted_category_index = torch.argmax(predictions[0]).item()
            predicted_category = learner.dls.vocab[predicted_category_index]
            predicted_probabilities = predictions[0].tolist()

            # Define the correct columns (categories)
            columns = ['Biashara', 'Burudani', 'Kimataifa', 'Kitaifa', 'Michezo']

            # Ensure the lengths of classes and predicted_probabilities match
            if len(predicted_probabilities) > len(columns):
                predicted_probabilities = predicted_probabilities[:len(columns)]
            elif len(predicted_probabilities) < len(columns):
                columns = columns[:len(predicted_probabilities)]

            # Create a DataFrame to align probabilities with categories
            df = pd.DataFrame([predicted_probabilities], columns=columns)

            # Display the results
            st.write("Predicted Category:", predicted_category)

            # Plotting the probabilities
            num_classes = len(df.columns)
            fig, ax = plt.subplots(figsize=(15, 5))
            y_pos = range(num_classes)
            bar_height = 0.35  # Adjust the height of the bars
            bars = ax.barh(y_pos, df.iloc[0], align='center', color='#64ffda')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(df.columns)  # Set y-axis labels to class names
            ax.invert_yaxis()  # Invert y-axis to display classes from top to bottom
            ax.set_xlabel('Probability', fontsize=14)
            ax.tick_params(axis='x', labelsize=12)  # Increase font size of x-axis tick labels

            # Add class labels and probability values to the bars
            for bar, prob, label in zip(bars, df.iloc[0], df.columns):
                ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                        f'{label}: {prob:.3f}',
                        ha='left', va='center',
                        fontsize=14, color='black')

            # Display the bar chart in Streamlit
            st.pyplot(fig)

        else:
            st.write("⚠️ The news content is not in Swahili language. Please make sure the input is in Swahili language.")
