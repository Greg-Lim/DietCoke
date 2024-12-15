import nltk
from nltk.corpus import stopwords
from collections import defaultdict
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
stop_words = set(stopwords.words('english'))

# POS tags to include
meaningful_pos_tags = ['NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJR', 'JJS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'RB', 'RBR', 'RBS']

def map_words_to_captions(captions):
    # Creates the format in aokvqa_val_caption.json
    word_to_caption_idx = defaultdict(list)
    
    for idx, caption in enumerate(captions):
        words = word_tokenize(caption)
        filtered_words = [word for word in words if word.lower() not in stop_words]
        tagged_words = nltk.pos_tag(filtered_words)
        meaningful_words = [word for word, pos in tagged_words if pos in meaningful_pos_tags]
        
        for word in meaningful_words:
            word_to_caption_idx[word.lower()].append(idx)
    
    return word_to_caption_idx
