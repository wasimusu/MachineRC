## PyTorch Implementation of Dynamic Coattention Network
Implementation of the paper Dynamic Coattention Network https://arxiv.org/pdf/1611.01604.pdf

## Improvement ideas
- Use layer normalization

### Parts of the architecture explained in brief
#### Encoder
Encodes both question and context
#### Coattention Network
Combines attention of question with context
#### Highway Maxout Network
Determines possible start and end points
#### Dynamic Decoding
Determines start and end points

#### What do the py files do
* config.py contains all the configuration
* baseline.py contains a baseline architecture based on tfidf and cosine distance
* vanillaQA.py contains baseline neural network architecture that might possibly work
* squad.py contains data parser for Squad Dataset
* setup.py - you need to run this after installing requirements to download data for nltk

* networks package has all of the networks in separate class for testing purpose