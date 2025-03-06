import numpy as np
import pandas as pd

# Function to load GloVe embeddings
def load_glove_embeddings(file_path):
    embeddings = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            values = line.strip().split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings

# Function to compute cosine similarity
def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2)

def get_glove_embeddings(df):
    df['glove'] = None  # init
    for index, row in df.iterrows():
        a = row['a'].lower()
        b = row['b'].lower()
        try:
            a_vec = embeddings[a]
            b_vec = embeddings[b]
            # sim = cos(a_vec,b_vec).item()
            sim = cosine_similarity(a_vec, b_vec)
            df.loc[index, 'glove'] = sim
        except:
            df.loc[index, 'glove'] = 'NA'

        # word1 = 'Kaffee'.lower()
        # word2 = 'Bohne'.lower()
        #
        # # Check if words are in the embeddings
        # if word1 in embeddings and word2 in embeddings:
        #     vec1 = embeddings[word1]
        #     vec2 = embeddings[word2]
        #     similarity = cosine_similarity(vec1, vec2)
        #     print(f"Cosine similarity between '{word1}' and '{word2}': {similarity}")
        # else:
        #     print(f"One or both words not found in the GloVe embeddings.")
    return(df)



if __name__=='__main__':

    # Load GloVe embedding file
    # embeddings = load_glove_embeddings('./glove.6B.300d.txt')
    embeddings = load_glove_embeddings('./multilingual_embeddings/multilingual_embeddings.de')

    # Optional: print vocab size
    print(len(embeddings.keys()))

    filename = 'stimuli_for_embeddings'
    df = pd.read_csv(f'./stone_2023/{filename}.csv',sep=',',encoding='utf-8')

    print('Getting GloVe embeddings...')
    df = get_glove_embeddings(df)

    out_filename = 'stimuli_glove_embeddings.csv'
    print(f'Writing to file: {out_filename}')
    df.to_csv(out_filename,
              sep = ',', encoding = 'utf-8', index = False)

    print('Done.')
