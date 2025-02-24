import torch
import fasttext
import fasttext.util
import pandas as pd

cos = torch.nn.CosineSimilarity(dim=0)

fasttext.util.download_model('de', if_exists='ignore')  #
ft = fasttext.load_model('cc.de.300.bin')

def get_fasttext_embeddings(df):
    df['fasttext'] = None  # init
    for index, row in df.iterrows():
        a = row['a']
        b = row['b']
        a_vec = torch.from_numpy(ft.get_word_vector(a))
        b_vec = torch.from_numpy(ft.get_word_vector(b))
        sim = cos(a_vec,b_vec).item()
        df.loc[index, 'fasttext'] = sim
    return(df)

if __name__=='__main__':

    filename = 'stimuli_for_embeddings'
    df = pd.read_csv(f'./{filename}.csv',sep=',',encoding='utf-8')

    print('Getting fasttext embeddings...')
    df = get_fasttext_embeddings(df)

    out_filename = 'stimuli_fasttext_embeddings.csv'
    print(f'Writing to file: {out_filename}')
    df.to_csv(out_filename,
              sep = ',', encoding = 'utf-8', index = False)

    print('Done.')
