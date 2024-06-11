#Using Albert (sentence piece tokenizer)

from transformers import AlbertTokenizer
tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')

def tokenize(x):
  return tokenizer.encode(x, truncation=True, max_length=10)

def test():
  x = "hello"
  print(tokenize(x))

def main():
  test()

if name == "main":
  main()