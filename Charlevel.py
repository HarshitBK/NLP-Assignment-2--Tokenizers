import os
from tqdm import tqdm
import pandas as pd
import argparse
from tokenizers import Tokenizer, trainers, models, pre_tokenizers, Tokenizer
from tokenizers import ByteLevelBPETokenizer
from transformers import PreTrainedTokenizerFast
import argparse
import datetime
import pandas as pd
from tokenizers.normalizers import Lowercase
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.pre_tokenizers import CharDelimiterSplit

def train_tokenizer(data_list, vocab_size=32768, model_name="test"):

    
    bos_tok = "<sos>"
    eos_tok = "<end_of_sen>"

    
    special_char = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ]

    
    tokenizer = Tokenizer(models.WordLevel(unk_token="[UNK]"))
    tokenizer.normalizer = Lowercase()  
    tokenizer.pre_tokenizer = CharDelimiterSplit(delimiter="") 
    trainer = trainers.WordLevelTrainer(
    vocab_size=1000,  
    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
)
    # trainer = trainers.WordPieceTrainer(special_tokens=["<pad>", "<unk>", "<sos>", "<end_of_sen>", "<user>", "<assistant>"]+special_char, vocab_size=200000000)
    tokenizer.train_from_iterator(
        data_list, trainer
    )

   
    transformer_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token = bos_tok,
        eos_token = eos_tok,
        unk_token = "<unk>",
        pad_token = "<pad>",
        mask_token = "<mask>",
        padding_side = "left",
        truncation_side = "right",
        additional_special_tokens = ["<user>", "<assistant>"],
        clean_up_tokenization_spaces = False,
    )

    transformer_tokenizer.save_pretrained(model_name)

input_file = "cleaned_output.txt"
df_1 = pd.read_csv(input_file, sep="\t", header=None)
cleaned_lines = []
for index, row in df_1.iterrows():
       
        line = " ".join(str(item) for item in row)
        print(index)
        
        line = line.strip()

       
        if not line:
            continue
        cleaned_lines.append(line)
cleaned_df = pd.DataFrame(cleaned_lines)

print(len(cleaned_df[0].to_list()))
train_tokenizer(cleaned_df[0].to_list())

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("test")
print("Vocab", len(tokenizer.get_vocab()))


with open(input_file, 'r', encoding='utf-8', errors='ignore') as f:
    data_list = f.readlines()
total_tokens = 0
total_words = 0

for sentence in data_list:
    input_ids = tokenizer.encode(sentence)
    total_tokens += len(input_ids)
    total_words += len(sentence.split())

print("Fertility Score for Char Level of Part 1:", total_tokens / total_words if total_words != 0 else 0)

input_text = "ਨਮਸਤੇ, ਮੈਨੂੰ ਉਮੀਦ ਹੈ ਕਿ ਤੁਸੀਂ ਚੰਗਾ ਕਰ ਰਹੇ ਹੋ".split(" ")
word_count  = len(input_text)
fertility = len(input_ids)/word_count
print(fertility)

# vocab = tokenizer.get_vocab()
# print("Vocab: ", vocab)