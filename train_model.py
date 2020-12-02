from nltk.corpus import brown   
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

sentences = brown.sents()                

documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(sentences)]

model = Doc2Vec(vector_size=100, min_count=1, epochs=25,dbow_words=1)

model.build_vocab(documents)

model.train(documents, total_examples=model.corpus_count, epochs=model.epochs)

model.save("brown.model")
