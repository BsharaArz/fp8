#Using Albert (sentence piece tokenizer)

from transformers import AlbertTokenizer
import jax
tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')

def tokenize(x):
  return tokenizer.encode(x, truncation=True, max_length=10)

#----------------------TESTS------------------------

def test():
  x = "hello"
  print(tokenize(x))

def main():
  test()

if __name__ == "__main__":
    main()