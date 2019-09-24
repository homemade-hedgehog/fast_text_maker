# fast_text_maker
fast_text model make wrapper

### コレは何？
- fasttextのwrapper
- 文をtokenに分割したもののリストからfasttextモデルを生成する

### usage
#### train
```python
from make_fast_text_bare import make_fast_text
import sentencepiece as spm
from tqdm import tqdm

# sentences
sentences = [
    "マイクロ化学デバイスおよび解析装置\n本発明は、マイクロ化学デバイスおよび解析装置に係り、特に、細胞を保持するマイクロウエルが多数形成されたマイクロ化学デバイスおよび解析装置に関する。",
    "刃物ホルダー\n本発明は、例えば複数のウィンナーが連なった連鎖状ウィンナーといった連鎖状食品の結束部を切断する刃物を保持する刃物ホルダーに関するものである。"
]

# sentence -> token like array
sp = spm.SentencePieceProcessor()
sp.load('path_to_sentence_piece_folder/sentence_piece.model')
tokens_list = [sp.EncodeAsPieces(sentence) for sentence in tqdm(sentences)]

# make model
make_fast_text(tokens_list, num_dimension=100) # create "fast_text_*/model_fast_text.bin"
```
#### use model
```python
import fasttext
import sentencepiece as spm
import numpy as np

model = fasttext.load_model("fast_text_4098fe0230900a25ecb1cfa7ed5271f1787f458d/model_fast_text.bin")
# sentence -> token like array
sp = spm.SentencePieceProcessor()
sp.load('path_to_sentence_piece_folder/sentence_piece.model')

# for token
target_token = 'マイクロ化学デバイス'
model.get_word_vector(target_token) # return numpy array, shape=(num_dimension, )

# for tokens: 文の代表ベクトル生成
# ## 単純な加算和の例
# ## 単純な加算和は、文意と関係ない単語も代表ベクトルに含めてしまうので、あんま精度が出ない e.g. "特に"　は有っても無くても文意は変わらない
# ## 現在は重み和を使うのが良いのだけれど、それはこのリポジトリーとは別個に作成予定
sentence = "マイクロ化学デバイスおよび解析装置\n本発明は、マイクロ化学デバイスおよび解析装置に係り、特に、細胞を保持するマイクロウエルが多数形成されたマイクロ化学デバイスおよび解析装置に関する。"
tokens = sp.EncodeAsPieces(sentence)
num_dimension = model.get_dimension()
representative = np.zeros(num_dimension)
for token in tokens:
    representative += model.get_word_vector(token)
representative /= len(tokens)
```

### 注意
- 入力の分割に使用したsentencepieceモデルにしか適用できない
- 別のsentencepieceモデルを使うと、分割の仕方が異なるので、fasttextモデルが意味を獲得していないことがある
- 要はsentencepieceモデルとfasttextモデルはセットにして、一緒に扱うこと
