import spacy


# Loading the English language model
nlp = spacy.load("en_core_web_sm")


def semantic_chunking(text, max_chunk_size=1500):
    doc = nlp(text)
    chunks = []
    current_chunk = []


    for sent in doc.sents:
        if len(" ".join(current_chunk + [sent.text])) <= max_chunk_size:
            current_chunk.append(sent.text)
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sent.text]
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks


# Reading the document
with open("municipal_regulations.txt", "r", encoding="utf-8") as file:
    text = file.read()


# Applying the semantic chunking
chunks = semantic_chunking(text)


# Displaying the chunks
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}:\n{chunk}\n")
