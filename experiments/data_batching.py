import tokenizer
import jax

def process_file(file_name):
  #open file
  print("open") #tracker
  with open(file_name, 'r') as file:
    file = file.readlines()

  print("flattening") #tracker
  file_tokenized = []

  #flatten file
  for x in file:
    file_tokenized.extend(tokenizer.tokenize(x))
    
  print("done") #tracker
  return file_tokenized

def create_batches(file_tokenized, batch_size, sequence_length):
  #calc num batches
  num_batches = len(file_tokenized)//batch_size + (len(file_tokenized) % batch_size != 0)
  for n in range(num_batches):
    #generate batch (size = batch_size x sequence_length)
    batch = []
    target=[]
    for b in range(batch_size):
      start = n * batch_size + (b * sequence_length)
      end = start + sequence_length
      batch.append(file_tokenized[start:end])
      target.append(file_tokenized[start+1:end+1])
    yield (batch, target)

def test():
  file_name = 'TinyStories-train.txt'
  file_tokenized = process_file(file_name)
  tokenized_batches = [(batch, target) for batch, target in create_batches(file_tokenized, 10, 32)]
  #here: send batch to model

def main():
  test()
'''
if name == "main":
  main()'''