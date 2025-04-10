# part 1.1, 1.2, 1.3

import gensim
from gensim.models import KeyedVectors

# part 1.1
wv = KeyedVectors.load('embs_train.kv')

big = wv['big']
dog = wv['dog']
# print("Big embedding: ", big, "dtype:", big.dtype)
 
# print("Dog embedding: ", dog, "dytype:", dog.dtype)

# part 1.2

# similar_dog = wv.most_similar('dog', topn=10)
# print("Similar to dog: ", similar_dog)
# similar_man = wv.most_similar('man', topn=10)
# print("Similar to man: ", similar_man)

similar_wonderful = wv.most_similar('wonderful', topn=10)
# print("Similar to wonderful: ", similar_wonderful)

similar_awful = wv.most_similar('awful', topn=10)
# print("Similar to awful: ", similar_awful)

similar_soda = wv.most_similar('soda', topn=10)
# print("Similar to soda: ", similar_soda)

similar_movie = wv.most_similar('movie', topn=10)
# print("Similar to movie: ", similar_movie)

similar_beauty = wv.most_similar('beauty', topn=10)
# print("Similar to beauty: ", similar_beauty)

# part 1.3 
similar_queen = wv.most_similar(positive=['king', 'woman'], negative = ['man'], topn=5 )
# print("Similar to queen: ", similar_queen)

similar_good = wv.most_similar(positive=['bigger', 'good'], negative = ['big'], topn=5 )
# print("Similar to good: ", similar_good)

similar_sister = wv.most_similar(positive=['sister', 'man'], negative =['woman'], topn=10 )
# print("Similar to sister: ", similar_sister)

similar_harder = wv.most_similar(positive=['harder', 'fast'], negative = ['hard'], topn=10 )
# print("Similar to harder: ", similar_harder)

similar_beauty = wv.most_similar(positive=['beauty', 'ugly'], negative = ['beautiful'], topn=10 )
print("Similar to beauty: ", similar_beauty)

similar_movie = wv.most_similar(positive=['movie', 'photo'], negative = ['film'], topn=10 )
print("Similar to movie: ", similar_movie)

similar_soda = wv.most_similar(positive=['soda', 'good'], negative = ['drink'], topn=10 )
print("Similar to soda: ", similar_soda)


