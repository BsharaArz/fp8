import jax
from tqdm import tqdm
import tokenizer

def process_file(file_name):
  #open file
  with open(file_name, 'r') as file:
    file = file.readlines()

  file_tokenized = []

  #tokenize by line
  for x in tqdm(file):
    file_tokenized.extend(tokenizer.tokenize(x))
    
  return file_tokenized

def create_batches(file_tokenized, batch_size, sequence_length):
  #calc num batches
  num_batches = len(file_tokenized)//(batch_size * sequence_length)

  for n in range(num_batches):
    #generate batch (size = batch_size x sequence_length)
    batch = []
    target=[]

    for b in range(batch_size):
      start = n * (batch_size * sequence_length) + (b * sequence_length)
      end = start + sequence_length
      batch.append(file_tokenized[start:end])
      target.append(file_tokenized[start+1:end+1])

    yield (batch, target)

#----------------------TESTS------------------------

def test():
  file_name = 'TinyStories-train.txt'
  file_tokenized = process_file(file_name)
  tokenized_batches = [(batch, target) for batch, target in create_batches(file_tokenized, 10, 32)]

def main():
  test()

if __name__ == "__main__":
    main()