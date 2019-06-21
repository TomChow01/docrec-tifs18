# http://pythonhosted.org/pyenchant/tutorial.html
from enchant import Dict
from enchant.tokenize import get_tokenizer
import re

regex = re.compile(r"[^\w\d'\s\-]+")

def merge(text1, text2):

    lines1 = text1.splitlines()
    lines2 = text2.splitlines()
    return [word1+word2 for word1, word2 in zip(lines1, lines2)]

# Interessante: http://www.hasbro.com/scrabble-2/en_US/search.cfm#dictionary
def text2words(text, language='en_US', min_length=3):

    dict_ = Dict(language)
    tknzr = get_tokenizer() # portuguese not available

    # Processed text: punctuation removal (except '-')
    p_text = regex.sub('', text)
    tokens = [token for token, _ in tknzr(p_text)]
    words = [token for token in tokens if len(token) >= min_length]
    words = [word for word in words if dict_.check(word)]
    return words

def remove_punctuation(text):

    return re.sub(r'\p{P}+', '', text)