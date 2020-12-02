from gensim.models.doc2vec import Doc2Vec
from scipy import spatial

model= Doc2Vec.load("brown.model")

with open('demofile.txt', 'r') as file:
    file_docs1 = file.read().replace('\n', '').replace('.','').replace(',','').lower()
    
with open('demofile2.txt', 'r') as file:
    file_docs2 = file.read().replace('\n', '').replace('.','').replace(',','').lower()     

vec1 = model.infer_vector(file_docs1.split())
vec2 = model.infer_vector(file_docs2.split())

print(file_docs1)
print(file_docs2)   

similairty = spatial.distance.cosine(vec1, vec2)
print("\nSimilarity between the files is %6.3f" %(100/(1+similairty)),"%")
