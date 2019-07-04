from gensim import corpora, models
from scipy.sparse import csr_matrix
import os,logging
import jieba
import pickle as pkl


def predict(demo_doc):
    path_doc_root = '../../train_data'  # 根目录 即存放按类分类好的文本
    path_tmp = '../../tmp'  # 存放中间结果的位置
    path_dictionary = os.path.join(path_tmp, 'THUNews.dict')
    path_tmp_tfidf = os.path.join(path_tmp, 'tfidf_corpus')
    path_tmp_lsi = os.path.join(path_tmp, 'lsi_corpus')
    path_tmp_lsimodel = os.path.join(path_tmp, 'lsi_model.pkl')
    path_tmp_predictor = os.path.join(path_tmp, 'predictor.pkl')

    dictionary = corpora.Dictionary.load(path_dictionary)
    lsi_file = open(path_tmp_lsimodel,'rb')
    lsi_model = pkl.load(lsi_file)
    lsi_file.close()
    x = open(path_tmp_predictor,'rb')
    predictor = pkl.load(x)
    x.close()
    files = os.listdir(path_tmp_lsi)
    catg_list = []
    for file in files:
        t = file.split('.')[0]
        if t not in catg_list:
            catg_list.append(t)


    demo_doc = list(jieba.cut(demo_doc,cut_all=False))
    demo_bow = dictionary.doc2bow(demo_doc)
    tfidf_model = models.TfidfModel(dictionary=dictionary)
    demo_tfidf = tfidf_model[demo_bow]
    demo_lsi = lsi_model[demo_tfidf]
    data = []
    cols = []
    rows = []
    for item in demo_lsi:
        data.append(item[1])
        cols.append(item[0])
        rows.append(0)
    demo_matrix = csr_matrix((data,(rows,cols))).toarray()
    x = predictor.predict(demo_matrix)
    x = catg_list[x[0]]
    return x
