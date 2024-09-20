"""
Description: This script processes object detection results from the GEMINI model by matching them with categories from the MSCOCO dataset using Word2Vec similarity. 
             It loads GEMINI results, filters and replaces object names based on their similarity to MSCOCO categories using pre-trained Word2Vec embeddings. 
             The final results are saved as a pickle file for further use.

Main Features:
1. Loads object detection results from GEMINI and MSCOCO category lists (instance and stuff annotations).
2. Uses a pre-trained Word2Vec model to filter and replace words in GEMINI responses based on their similarity to MSCOCO categories.
3. Supports removal of non-relevant words based on a separate category list (stuff categories).
4. Saves the processed results, including original, filtered, and replaced words, in a pickle file for further analysis.

Author: Kuinan
Date: 2024-09-01
Version: 1.0
"""
from gensim.models import KeyedVectors
import json
from tqdm.auto import tqdm
import pickle

model = KeyedVectors.load_word2vec_format("your path to the model", binary=True)
# If you do not have the model
# Run: wget https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz

with open('/Data/GEMINI_result_dataset.json', 'rb') as file: # choose the gemini response you want to process
    gemini_results = json.load(file)

with open('path/to/MSCOCO instance annotation', 'r') as file:
    tr17 = json.load(file)
with open('path/to/MSCOCO stuff annotation', 'r') as file:
    sttr17 = json.load(file)

cat_ls = []
for dict in tr17['categories']:
    cat_ls.append(dict['name'])

rm_ls = []
for dict in sttr17['categories']:
    cat_ls.append(dict['name'])

prompts = {}
for im_id, result in tqdm(gemini_results.items()):
    if result:
        temp = result # get the response from LLM
        temp = temp.replace('77592','') # remove identification code
        temp = temp.strip()
        response_words = temp.split(', ') # convert reponse to a list of words
        
        filtered_list = []
        final_list = []
        
        # Loop through each word in the response
        for word in response_words:
            if word in cat_ls:
                filtered_list.append(word)
                final_list.append(word)
            elif word in rm_ls:
                filtered_list.append(word)
                # Do not add to final_list since it should be removed
            else:
                max_cat_similarity = -1
                max_rm_similarity = -1
                most_similar_cat_word = None
                most_similar_rm_word = None
                
                # Calculate similarity with cat_ls
                for cat_word in cat_ls:
                    try:
                        similarity = model.similarity(word, cat_word)
                        if similarity > max_cat_similarity:
                            max_cat_similarity = similarity
                            most_similar_cat_word = cat_word
                    except KeyError:
                        continue
                
                # Calculate similarity with rm_ls
                for rm_word in rm_ls:
                    try:
                        similarity = model.similarity(word, rm_word)
                        if similarity > max_rm_similarity:
                            max_rm_similarity = similarity
                            most_similar_rm_word = rm_word
                    except KeyError:
                        continue
                
                # Decide based on highest similarity
                if max_cat_similarity >= max_rm_similarity:
                    if most_similar_cat_word:
                        filtered_list.append(word)
                        final_list.append(most_similar_cat_word)
                else:
                    if most_similar_rm_word:
                        filtered_list.append(word)
                        # Do not add to final_list since it should be removed
        prompts[im_id] = {'original':response_words, 'filtered':filtered_list, 'replaced':final_list}
    else:
        print(f'None results at {im_id}: {result}')

with open('save to the path you like','wb') as file:
    pickle.dump(prompts, file)