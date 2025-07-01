import json
from unittest import result
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from tqdm import tqdm
import os
from dotenv import load_dotenv
import requests, certifi
import torch
import requests
from PIL import Image
from io import BytesIO
import open_clip

# Load your JSON data
with open("nikebalance-result.json", "r") as f:
    data = json.load(f)

# Load variables from .env file
load_dotenv()

# Extract text fields (adjust this based on your JSON structure)
 
texts = []

if products := data['products']:
    texts = products
    #print(f"Processing item: {products}")

# Generate image embeddings using 
def get_text_embedding(text):
    embed_model = OpenAIEmbeddings(model="text-embedding-3-small")
    return embed_model.embed_query(text)

# Set compute device to use gpu (cuda) if available or cpu 
# Load model and preprocessing (resize/crop) to match the expected input format of the pre-trained model and covert to tokens
# Load model on compute device
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess, tokenizer = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion400m_e31')
model = model.to(device)

# Generate image embeddings using pyTorch disabling gradient computation 
def get_image_embedding(image_url):
    try:
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content)).convert("RGB")
        image_tensor = preprocess(image).unsqueeze(0).to(device) # Create a mini-batch as expected by the model

        with torch.no_grad():
            image_features = model.encode_image(image_tensor)
        return image_features.cpu().squeeze().tolist()
    except Exception as e:
        print(f"Image error: {e}")
        return None

results = []

# Process each item in the texts list
# tqdm is used to show a progress bar
for item in tqdm(texts):
    #print(f"Processing item: {item}")
    name = item.get("name", "")
    description = item.get("description", "")
    text_input = description if description.strip() else name

    # Get image URL (prefer "extra-large", fallback to "large")
    image_url = None
    image_id = item.get("imageIds", [None])[0]
    if image_id:
        images = item.get("imageResources", {}).get(image_id, [])
        for img in images:
            if img["usage"] == "extra-large":
                image_url = img["url"]
                break
        if not image_url:
            for img in images:
                if img["usage"] == "large":
                    image_url = img["url"]
                    break

    # Get embeddings
    text_embedding = get_text_embedding(text_input)
    #print(len(text_embedding), text_embedding[:5])  
    image_embedding = get_image_embedding(image_url) if image_url else None

    results.append({
        "id": item["id"],
        "name": name,
        "text_embedding": text_embedding,
        "image_embedding": image_embedding,
        "combined_embedding": None  # Optional: average or concatenate embeddings
    })

    print(f"Processing embeddings results: {results}")



# Create the vector store
#vectorstore = FAISS.from_texts(texts, results)

# Save the vector store for later use
#vectorstore.save_local("faiss_index")