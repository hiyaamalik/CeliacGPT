import streamlit as st
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import faiss
import numpy as np

# Initialize the DistilGPT-2 model (a lighter version of GPT-2)
generator = pipeline("text-generation", model="distilgpt2")

# Define a simple medical knowledge base (this can be expanded)
knowledge_base = [
    "Celiac disease is an autoimmune disorder where the ingestion of gluten (a protein found in wheat, barley, and rye) causes damage to the small intestine.",
    "The disease is triggered by the body's immune response to gluten, which leads to inflammation and damage to the villi in the small intestine.",
    "The villi are tiny hair-like structures that line the small intestine and are essential for nutrient absorption.",
    "When the villi are damaged, the body is unable to properly absorb nutrients, leading to malnutrition, vitamin deficiencies, and a variety of other health problems.",
    "Celiac disease is genetic, and individuals with a first-degree relative with the condition are at an increased risk.",
    "It affects both children and adults, although it can sometimes go undiagnosed for years.",
    "Diagnosis of celiac disease typically involves blood tests to check for specific antibodies (tTG-IgA and EMA), followed by a biopsy of the small intestine to assess the extent of damage to the villi.",
    "The biopsy is the gold standard for confirming celiac disease, but the blood tests are highly indicative and often used as the first step in diagnosis.",
    "The main symptom of celiac disease is diarrhea, but other common symptoms include abdominal pain, bloating, gas, weight loss, and fatigue.",
    "In children, celiac disease may cause delayed growth, irritability, and behavioral issues.",
    "Some people may also experience symptoms outside of the digestive system, such as skin rashes (dermatitis herpetiformis), joint pain, and headaches.",
    "In adults, celiac disease may present as anemia, osteopenia (low bone density), or infertility.",
    "In addition to these common symptoms, some individuals may have more subtle symptoms, or they may be asymptomatic, making diagnosis more difficult.",
    "If left untreated, celiac disease can lead to severe complications such as osteoporosis, infertility, neurological disorders, and an increased risk of certain types of cancer, including lymphoma and small bowel cancer.",
    "Management of celiac disease primarily involves a strict, lifelong gluten-free diet.",
    "This is the only known effective treatment for the disease and helps to heal the intestine, prevent further damage, and alleviate symptoms.",
    "A gluten-free diet means avoiding all foods that contain wheat, barley, rye, and any ingredients derived from these grains.",
    "This includes bread, pasta, cereals, baked goods, and processed foods that contain gluten as an additive or thickener.",
    "Cross-contamination is also a concern, so it is important to ensure that foods are prepared in a gluten-free environment and that utensils, surfaces, and appliances are thoroughly cleaned.",
    "In addition to a gluten-free diet, individuals with celiac disease may need to take supplements to address nutritional deficiencies, such as iron, calcium, vitamin D, and folate.",
    "People with celiac disease may also need to be monitored by a healthcare provider for other conditions associated with the disease, including osteoporosis, thyroid disease, and liver issues.",
    "The management of celiac disease involves regular check-ups and monitoring for complications, as well as lifestyle changes to minimize the risk of gluten exposure.",
    "There is currently no cure for celiac disease, but following a gluten-free diet can allow most people with the condition to live a healthy life and prevent complications.",
    "For some individuals, the disease may be diagnosed later in life, and they may face challenges in adjusting to a gluten-free lifestyle.",
    "Celiac disease can also have a significant impact on mental health, as individuals may feel isolated due to dietary restrictions, or they may experience anxiety around food choices.",
    "Social support from family, friends, and celiac disease support groups can be crucial in managing the emotional aspects of the disease."
]

# Function to encode text into embeddings (using DistilGPT-2 embeddings here for simplicity)
def encode_text(texts):
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    outputs = model.transformer.wte(inputs.input_ids)
    return outputs.mean(dim=1).detach().numpy()

# Encode the knowledge base into vectors
encoded_kb = encode_text(knowledge_base)
index = faiss.IndexFlatL2(encoded_kb.shape[1])  # FAISS index for vector search
index.add(np.array(encoded_kb))  # Add encoded knowledge base to FAISS index

# Function to retrieve relevant information based on user query
def retrieve_info(query):
    query_vec = encode_text([query])
    D, I = index.search(query_vec, k=3)  # Retrieve top 3 most relevant pieces of knowledge
    relevant_info = [knowledge_base[i] for i in I[0]]
    return " ".join(relevant_info)

# Function to generate a response based on retrieved information
def generate_response(query):
    context = retrieve_info(query)
    prompt = f"User Query: {query}\nContext: {context}\nAnswer:"
    response = generator(prompt, max_length=150, num_return_sequences=1)
    return response[0]["generated_text"].strip()

# Streamlit app
st.title("Celiac Disease Information Assistant")
st.write("Ask me anything about Celiac Disease or Gluten Intolerance.")

# Input for the user query
user_query = st.text_input("Your Question:")

if user_query:
    response = generate_response(user_query)
    st.write("Response:")
    st.write(response)
