#coding=utf-8
from gensim import corpora, models
import jieba.posseg as jp, jieba
str = "专家们介绍,经过科学试验和临床探索,已研发筛选出一些针对轻重症的有效中西药进入诊疗方案,还有一些药物正在开展临床试验,李克强仔细询问这些药的作用和安全性,他说,药物是病毒的克星,全社会对此急迫期待,要进一步做好筛选和临床试验工作,种类不在多,关键要集中精选几种安全可靠、临床效果显著的主打药,让医务人员和群众更好使用,更有效救治重症患者、降低病亡率,更有效阻止轻症转为重症,这样克制病毒、战胜疫情就更有底气,通过实地调查、走访当地村民等方法,了解到原遗址核心区因开挖水塘对遗址本体破坏较为严重,对遗址部分区域进行了小范围勘探,并清理自然断面观察地层堆积情况,根据考古钻探与自然断面清理情况,遗址大部分区域已被近现代活动破坏,仅在遗址南部的零散区域发现有早期文化堆积,厚度在0.2-1.5米不等,残余文化堆积分布面积约9000平方米,童家遗址的发现为了解湘东北地区石家河文化时期的社会面貌和湘鄂赣三地间的文化互动提供了宝贵材料".split(",")
print(str)


# 文本集
texts = []
texts = str
# jieba.add_word('四强', 9, 'n')
flags = ('n', 'nr', 'ns', 'nt', 'eng', 'v', 'd')  # 词性
stopwords = ('仔细','的','显著','更','')
words_ls = []
for text in texts:
    words = [word.word for word in jp.cut(text) if word.flag in flags and word.word not in stopwords]
    words_ls.append(words)
# print(words_ls)
# 去重，存到字典
dictionary = corpora.Dictionary(words_ls)
# print(dictionary)
corpus = [dictionary.doc2bow(words) for words in words_ls]
# print(corpus)
lda = models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=2)
for topic in lda.print_topics(num_words=4):
    print(topic)
# 主题推断
# print(lda.inference(corpus))
text5 = '历史遗迹是人类的财富'
bow = dictionary.doc2bow([word.word for word in jp.cut(text5) if word.flag in flags and word.word not in stopwords])
ndarray = lda.inference([bow])[0]
print(text5)
for e, value in enumerate(ndarray[0]):
    print('\t主题%d推断值%.2f' % (e, value))

