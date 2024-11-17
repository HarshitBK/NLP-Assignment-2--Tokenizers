import os
import pandas as pd
from tqdm import tqdm
from tokenizers import ByteLevelBPETokenizer
from transformers import PreTrainedTokenizerFast, AutoTokenizer


def train_tokenizer(data_list, vocab_size=32768, model_name="test_tokenizer"):
    
    tokenizer = ByteLevelBPETokenizer()

    special_tokens = ["<pad>", "<unk>", "<sos>", "<eos>", "<user>", "<assistant>"]

    tokenizer.train_from_iterator(
        data_list,
        vocab_size=vocab_size,
        min_frequency=5,  
        special_tokens=special_tokens,
    )

    os.makedirs(model_name, exist_ok=True)
    tokenizer.save_model(model_name)

    transformer_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<sos>",
        eos_token="<eos>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
        additional_special_tokens=["<user>", "<assistant>"],
    )

    transformer_tokenizer.save_pretrained(model_name)
    print(f"Tokenizer trained and saved in directory: {model_name}")



input_file = "combined_output_1.txt"
dataset_size_kb = os.path.getsize(input_file) / 1024  # Get file size in KB
print(f"Dataset File Size: {dataset_size_kb:.2f} KB")
df = pd.read_csv(input_file, sep="\t", header=None)
cleaned_lines = []

for index, row in tqdm(df.iterrows(), total=len(df)):

    line = " ".join(str(item) for item in row).strip()
    
    if line:
        cleaned_lines.append(line)


print("Starting tokenizer training...")
train_tokenizer(cleaned_lines, vocab_size=32768, model_name="test_tokenizer")


tokenizer = AutoTokenizer.from_pretrained("test_tokenizer")
print(f"Vocabulary size: {len(tokenizer.get_vocab())}")


with open(input_file, 'r', encoding='utf-8', errors='ignore') as f:
    data_list = f.readlines()

total_tokens = 0
total_words = 0

for sentence in data_list:
    input_ids = tokenizer.encode(sentence)
    total_tokens += len(input_ids)
    total_words += len(sentence.split())

fertility_score = total_tokens / total_words if total_words != 0 else 0
print(f"Fertility Score for Byte-Level Tokenizer for part 2: {fertility_score}")


input_text = "ਨਮਸਤੇ, ਮੈਨੂੰ ਉਮੀਦ ਹੈ ਕਿ ਤੁਸੀਂ ਚੰਗਾ ਕਰ ਰਹੇ ਹੋ".split(" ")
input_sentence = " ".join(input_text)
input_ids = tokenizer.encode(input_sentence)
word_count = len(input_text)
fertility = len(input_ids) / word_count
print(f"Fertility Score for Example Input: {fertility}")
