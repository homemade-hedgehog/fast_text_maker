from tqdm import tqdm
import fasttext
import codecs
import os
import hashlib
import time


def make_save_folder(prefix="", add_suffix=True) -> str:
    """
    1. 現在時刻のハッシュをsuffixにした文字列の生成
    2. 生成した文字列のフォルダが無かったら作る
    :param prefix:save folderの系統ラベル
    :param add_suffix: suffixを付与するかを選ぶフラグ, True: 付与, False: 付与しない
    :return: str, モデルのセーブ先フォルダ名
    """
    if prefix == "":
        prefix = "./fast_text"
    if add_suffix:
        prefix = f"{prefix}_{hashlib.sha1(time.ctime().encode()).hexdigest()}"
    if not prefix.endswith("/"):
        prefix += "/"
    if not os.path.exists(prefix):
        os.mkdir(prefix)
    return prefix


def make_fast_text(tokens_list: list, num_dimension: int, save_folder="", min_count=1) -> bool:
    """
    現在の言語処理では、単語分散表現を使うのがデファクトになった
    単語分散表現は、単語を意味に応じて高次元空間に配置する手法である
    原点からのベクトルに意味をもたせているので、同じ向きを向いている単語同士は意味が近い
    理論的には分布仮説に基づいて、任意の単語に意味を、出現位置近傍の単語で規定する
    2013年のword2vecという手法が初出で、後年に同じ研究者がfasttextを考案した
    以下の点が異なる　[here](http://54.92.5.12/2019/05/09/fasttext%E3%81%A8word2vec%E3%81%AE%E9%81%95%E3%81%84/)
    - subwordと呼ばれる単語の部分文字列でも意味を獲得する
    - fasttextのほうが圧倒的に早い

    ここでは、文を分割したtokenのlistのlistを受け取って、fasttextのモデルをセーブする
    tokenのlistは語順が重要なので、文脈の順を崩さないこと
    :param tokens_list: list of list of str, tokenのリストのリスト
    :param num_dimension: int, word embedding space dimension
    :param save_folder: str, path to save folder
    :param min_count: int, fasttextがモデルに反映する最小の単語出現頻度, １なら全文書中に1度だけ出現
    :return: bool
    """
    # arrange save folder
