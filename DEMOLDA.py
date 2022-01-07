import jieba
# 语料库
from gensim import corpora
# 导入语言模型
from gensim import models
# 稀疏矩阵相识度
from gensim import similarities
# 语料库
l1 = ["你的名字是什么", "你今年几岁了", "你有多高你手掌多大", "你手掌有多大"]
# 用户问的问题
a = "你今年多大了"
all_doc_list = []  #
for doc in l1:
    # 利用jieba分词将语料库中的每一个问题切割
    doc_list = [word for word in jieba.cut(doc)]
    all_doc_list.append(doc_list)
print(all_doc_list)
# [['你', '的', '名字', '是', '什么'], ['你', '今年', '几岁', '了'], ['你', '有', '多', '高', '你', '手多大'], ['你', '手多大']]

# 利用jieba分词将要问的问题切割
doc_test_list = [word for word in jieba.cut(a)]
print(doc_test_list)
# ['你', '今年', '多大', '了']  ==>1685

# 制作语料库
dictionary = corpora.Dictionary(all_doc_list)  # 制作词袋
# 词袋:是根据当前所有的问题即列表all_doc_list中每一个列表中的每一个元素(就是字)为他们做一个唯一的标志,形成一个key:velue的字典
print("token2id", dictionary.token2id)
# print("dictionary", dictionary, type(dictionary))
# token2id {'什么': 0, '你': 1, '名字': 2, '是': 3, '的': 4, '了': 5, '今年': 6, '几岁': 7, '多': 8, '手多大': 9, '有': 10, '高': 11}

# ['你', '的', '名字', '是', '什么'] ==>14230
# ['你', '今年', '几岁', '了']  ==>1675

# 制作语料库
# 这里是将all_doc_list 中的每一个列表中的词语 与 dictionary 中的Key进行匹配
# doc2bow文本变成id,这个词在当前的列表中出现的次数
# ['你', '的', '名字', '是', '什么'] ==>(1,1),(4,1),(2,1),(3,1),(0,1)
# 1是你 1代表出现一次, 4是的  1代表出现了一次, 以此类推 2是名字 , 3是是,0是什么
corpus = [dictionary.doc2bow(doc) for doc in all_doc_list]  # 语料库中的句子的向量表示[(id,词频),(id,词频)]
print("corpus", corpus, type(corpus))
# corpus [[(0, 1), (1, 1), (2, 1), (3, 1), (4, 1)], [(1, 1), (5, 1), (6, 1), (7, 1)], [(1, 2), (8, 1), (9, 1), (10, 1), (11, 1)], [(1, 1), (9, 1)]] <class 'list'>

# ['你', '今年', '多大', '了']  词袋中没有多大165
# 所以需要词向量化

# 将需要寻找相似度的分词列表 做成 语料库 doc_test_vec
# ['你', '今年', '多大', '了']  (1, 1), (5, 1), (6, 1)
doc_test_vec = dictionary.doc2bow(doc_test_list)
print("doc_test_vec", doc_test_vec, type(doc_test_vec))
# doc_test_vec [(1, 1), (5, 1), (6, 1)] <class 'list'>

# 将corpus语料库(初识语料库) 使用Lsi模型进行训练,将语料库变成计算机可识别可读的数字
lsi = models.LsiModel(corpus)
print("lsi", lsi, type(lsi))
# lsi[corpus] <gensim.interfaces.TransformedCorpus object at 0x000001D92EEB43C8>
# 语料库corpus的训练结果
print("lsi[corpus]", lsi[corpus])
# lsi[corpus] <gensim.interfaces.TransformedCorpus object at 0x000001D92EEB43C8>

# 将问题放到放到已经训练好的语料库模型一个一个匹配,获取匹配分值
# 获得语料库doc_test_vec 在 语料库corpus的训练结果 中的 向量表示
print("lsi[doc_test_vec]", lsi[doc_test_vec])
# lsi[doc_test_vec] [(0, 0.9910312948854694), (1, 0.06777365757876067), (2, 1.1437866478720622), (3, -0.015934342901802588)]
# 排过序,数字表示相识度,负数表示无限不接近

# 文本相似度
# 稀疏矩阵相似度 将 主 语料库corpus的训练结果 作为初始值

# 举例:匹配5*5的正方形
# 目前有[(3,3),(3,5),(4,5),(6,6)]
# 设定-(1形状相似2.周长相识3面积相似),根据选定设定好的条件匹配最相似的,涉及算法

# lsi[corpus]==>Lsi训练好的语料库模型
# index是设定的匹配相识度的条件
index = similarities.SparseMatrixSimilarity(lsi[corpus], num_features=len(dictionary.keys()))
print("index", index, type(index))

# 将 语料库doc_test_vec 在 语料库corpus的训练结果 中的 向量表示 与 语料库corpus的 向量表示 做矩阵相似度计算
gongyinshi = lsi[doc_test_vec]
print(gongyinshi)
sim = index[gongyinshi]

print("sim", sim, type(sim))

# 对下标和相似度结果进行一个排序,拿出相似度最高的结果
# cc = sorted(enumerate(sim), key=lambda item: item[1],reverse=True)
cc = sorted(enumerate(sim), key=lambda item: -item[1])
print(cc)

text = l1[cc[0][0]]
if cc[0][1] > 0:
    print(a, text)