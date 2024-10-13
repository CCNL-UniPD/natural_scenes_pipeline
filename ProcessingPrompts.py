"""
Description:
This script processes image caption data by cleaning and filtering words based on their semantic similarity 
to specified categories using a pre-trained Word2Vec model. It removes words that are more similar to 
a set of "removal" categories than to the target categories, while also normalizing word case and removing duplicates.

Requirements:
- Pre-trained Word2Vec model: Google News (300 dimensions)
- MSCOCO annotation files:
    - 'stuff_train2017.json' (for removal category list)
    - 'instances_train2017.json' (for target category list)
- GEMINI result file: 'GEMINI_result_mscoco.json'

Steps:
1. Load the Word2Vec model from a binary file or download via gensim.
2. Load MSCOCO annotation data and extract categories for both removal and target lists.
3. Clean the image caption data by removing unnecessary characters and words from the removal category list.
4. Use the Word2Vec model to compare word similarity between removal and target categories.
5. Retain words that are more similar to target categories, normalize the case, and remove duplicates.
6. Output the cleaned prompts with basic statistics on the word processing.

Usage:
- Modify file paths for the Word2Vec model, MSCOCO annotation files, and GEMINI result file as necessary.
- Run the script to generate a cleaned version of the image caption data.
"""

import json
from tqdm.auto import tqdm
import re
from gensim.models import KeyedVectors
import gensim.downloader as api
import pickle

# Download and load the model if you do not have it on your machine
#model = api.load('word2vec-google-news-300')

# Load pre-trained word2vec model
model = KeyedVectors.load_word2vec_format("word2vec-google-news-300.gz", binary=True)

# Load MSCOCO annotations
with open("stuff_train2017.json", 'rb') as file:
    sttr17 = json.load(file)

with open("instances_train2017.json", 'rb') as file:
    tr17 = json.load(file)

# Extract category lists
cat_ls = [item['name'] for item in tr17['categories']]
rm_ls = [item['name'] for item in sttr17['categories']]

# Load GEMINI results
with open("Data\GEMINI_result_mscoco.json", 'r') as file:
    prompts = json.load(file)

# Clean the GEMINI prompts
prompts = {
    im_id: re.findall(r'(?:[^,]+(?:,\s*|$))', data['response'].replace('77592:', '').strip())
    for im_id, data in tqdm(prompts.items()) if data['response']
}
# if with pascal, uncomment the following
#prompts = {
#    im_id: re.findall(r'(?:[^,]+(?:,\s*|$))', data.replace('77592:', '').strip())
#    for im_id, data in tqdm(prompts.items()) if data}


prompts = {
    im_id: [item.strip().rstrip(',') for item in items]
    for im_id, items in prompts.items()
}

# Remove words from rm_ls
prompts = {
    im_id: [word for word in words if word not in rm_ls]
    for im_id, words in prompts.items()
}

# Helper function for computing similarities
def get_max_similarity(word, word_list, model):
    similarities = [
        model.similarity(word, target_word)
        for target_word in word_list if target_word in model.key_to_index
    ]
    return max(similarities, default=0)

# Initialize counts
total_words = 0
words_compared = 0
keyerror_count = 0

# Initialize new prompts dictionary
new_prompts = {}
model_vocab = set(model.key_to_index.keys())

# Process each image_id and its words
for image_id, words in prompts.items():
    new_words = []
    
    for word in words:
        total_words += 1

        # If word not in model's vocab, keep it
        if word not in model_vocab:
            keyerror_count += 1
            new_words.append(word)
            continue

        # Calculate max similarities
        max_sim_rm = get_max_similarity(word, rm_ls, model)
        max_sim_cat = get_max_similarity(word, cat_ls, model)
        
        words_compared += 1

        # Keep word if it's more similar to cat_ls than rm_ls
        if max_sim_cat >= max_sim_rm:
            new_words.append(word)

    # Normalize to lowercase and remove duplicates
    new_prompts[image_id] = list(set([word.lower() for word in new_words]))
with open('/Data/postprocessed_gemini_pascal.pkl','wb') as file:
    pickle.dump(new_prompts,file)
# Print statistics
print(f"Total number of words: {total_words}")
print(f"Number of words compared in the model: {words_compared}")
print(f"Number of words that raised KeyError: {keyerror_count}")