#Using Albert (sentence piece tokenizer)

from transformers import AlbertTokenizer
tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')

def tokenize(input):
  return tokenizer.encode(input)

def test():
  input = "hello"
  print(tokenize(input))

def main():
  test()

if name == "main":
  main()