### Python project to recommend matching outfits for the user

Steps to generate recommendations:

1. Create text (name, description) and image vectors of all the products from the local json
2. Create text vectors using OpenAIEmbeddings and image vectors using OpenCLIP model ("laion400m_e31") and PyTorch
3. Compute these embeddings on local GPU or CPU
4. Store them in FAISS store
5. Use cosine similarity to recommend outfits

Next:
Use Fashionbert 
