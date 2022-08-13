from torchtext.transforms import BERTTokenizer

VOCAB_FILE = "https://huggingface.co/bert-base-uncased/resolve/main/vocab.txt"
tokenizer = BERTTokenizer(vocab_path=VOCAB_FILE, do_lower_case=True, return_tokens=True)
print(tokenizer("Hello World, How are you!"))
