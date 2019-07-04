from gensim import corpora, models
from scipy.sparse import csr_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import svm
import numpy as np
import os,re,time,logging
import jieba
import pickle as pkl
import warnings
logging.basicConfig(level=logging.info)
warnings.filterwarnings(action='ignore',category=UserWarning,module='gensim')#忽略warning警告错误


class loadFolders(object):   # 迭代器
    def __init__(self,par_path):
        self.par_path = par_path
    def __iter__(self):
        for file in os.listdir(self.par_path):
            file_abspath = os.path.join(self.par_path, file)
            if os.path.isdir(file_abspath): # if file is a folder
                yield file_abspath

class loadFiles(object):
    def __init__(self,par_path):
        self.par_path = par_path
    def __iter__(self):
        folders = loadFolders(self.par_path)
        for folder in folders:              # level directory
            catg = folder.split(os.sep)[-1]
            for file in os.listdir(folder):     # secondary directory
                file_path = os.path.join(folder,file)
                if os.path.isfile(file_path):
                    this_file = open(file_path,'rb')
                    content = this_file.read().decode('utf8')
                    yield catg,content
                    this_file.close()

def convert_doc_to_wordlist(str_doc,cut_all):
    logging.info('分词，去停用词')
    sent_list = str_doc.split('\n')
    sent_list = map(rm_char, sent_list) # 去掉一些字符，例如\u3000
    word_2dlist = [rm_tokens(jieba.cut(part,cut_all=cut_all)) for part in sent_list] 
    word_list = sum(word_2dlist,[])
    return word_list

def rm_tokens(words): # 去停用词
    words_list = list(words)
    stop_words = get_stop_words()
    for i in range(words_list.__len__())[::-1]:
        if words_list[i] in stop_words:
            words_list.pop(i)
        elif words_list[i].isdigit():
            words_list.pop(i)
    return words_list

def get_stop_words(path=r'stopwords.txt'):
    file = open(path,'rb').read().decode('utf8').split('\n')
    return set(file)

def rm_char(text):
    text = re.sub('\u3000','',text)
    return text

def svm_classify(train_set,train_tag,test_set,test_tag):
    clf = svm.LinearSVC()
    clf_res = clf.fit(train_set,train_tag)
    train_pred  = clf_res.predict(train_set)
    test_pred = clf_res.predict(test_set)

    # train_err_num, train_err_ratio = checkPred(train_tag, train_pred)
    # test_err_num, test_err_ratio  = checkPred(test_tag, test_pred)
    train_acc_num, train_acc_ratio = checkPred(train_tag, train_pred)
    test_acc_num, test_acc_ratio = checkPred(test_tag, test_pred)
    train_rec_num,train_rec_ratio=recall(train_tag,train_pred)
    test_rec_num,test_rec_ratio=recall(test_tag,test_pred)
    # lda = LinearDiscriminantAnalysis(solver="svd", store_covariance=True)
    # lda_res = lda.fit(train_set, train_tag)
    # train_pred = lda_res.predict(train_set)  # 训练集的预测结果
    # test_pred = lda_res.predict(test_set)

    logging.info('=== 分类训练完毕，分类结果如下 ===')
    logging.info('训练集准确率: {e}'.format(e=train_acc_ratio))
    logging.info('检验集准确率: {e}'.format(e=test_acc_ratio))
    logging.info('训练集召回率: {e}'.format(e=train_rec_ratio))
    logging.info('测试集召回率: {e}'.format(e=test_rec_ratio))

    #logging.info('训练集误差: {e}'.format(e=train_pred))
   # logging.info('检验集误差: {e}'.format(e=test_pred))

    return clf_res


def checkPred(data_tag, data_pred):
    if data_tag.__len__() != data_pred.__len__():
        raise RuntimeError('The length of data tag and data pred should be the same')
    accuary_count = 0
    for i in range(data_tag.__len__()):
       # if data_tag[i]!=data_pred[i]:
       if data_tag[i] == data_pred[i]:
            accuary_count += 1
    accuary_ratio = accuary_count / data_tag.__len__()

    return [accuary_count, accuary_ratio]
def recall(data_tag,data_pred):
    if data_tag.__len__() != data_pred.__len__():
        raise RuntimeError('The length of data tag and data pred should be the same')
    accuary_count = 0
    for i in range(data_tag.__len__()):
        # if data_tag[i]!=data_pred[i]:
        if data_tag[i] == data_pred[i]:
            accuary_count += 1
    recall_ratio = accuary_count / (data_tag.__len__() + data_pred.__len__())
    return [accuary_count,recall_ratio]





if __name__=='__main__':
    path_doc_root = '../../train_data' # 根目录 即存放按类分类好的文本
    path_tmp = '../../tmp'  # 存放中间结果的位置
    path_dictionary     = os.path.join(path_tmp, 'THUNews.dict')
    path_tmp_tfidf      = os.path.join(path_tmp, 'tfidf_corpus')
    path_tmp_lsi        = os.path.join(path_tmp, 'lsi_corpus')
    path_tmp_lsimodel   = os.path.join(path_tmp, 'lsi_model.pkl')
    path_tmp_predictor  = os.path.join(path_tmp, 'predictor.pkl')

    n = 10  # n 表示抽样率， n抽1
    dictionary = None
    corpus_tfidf = None
    corpus_lsi = None
    lsi_model = None
    predictor = None
    if not os.path.exists(path_tmp):
        os.makedirs(path_tmp)     #创建路径文件夹

    logging.info('一、遍历文档，生成词典，并去掉频率较少的词')
    if not os.path.exists(path_dictionary):
        logging.info('未检测到有词典存在，开始遍历生成词典')
        dictionary = corpora.Dictionary()
        files = loadFiles(path_doc_root)
        for i,msg in enumerate(files):    #遍历file
            if i%n==0:
                catg    = msg[0]
                file    = msg[1]
                file = convert_doc_to_wordlist(file,cut_all=False)
                dictionary.add_documents([file])#更新当前字典；字典value为文章内容
                if int(i/n)%1000==0:
                    logging.info('{t} *** {i} \t docs has been dealed'
                          .format(i=i,t=time.strftime('%Y-%m-%d %H:%M:%S',time.localtime())))
        logging.info('去掉词典中出现次数过少的词')
        small_freq_ids = [tokenid for tokenid, docfreq in dictionary.dfs.items() if docfreq < 5 ]
        dictionary.filter_tokens(small_freq_ids)
        dictionary.compactify()
        dictionary.save(path_dictionary)#将生成的字典数持久化
        logging.info('=== 词典已经生成 ===')
    else:
        logging.info('=== 检测到词典已经存在，跳过该阶段 ===')

    #   第二阶段，  开始将文档转化成tfidf
    if not os.path.exists(path_tmp_tfidf):
        logging.info('=== 未检测到有tfidf文件夹存在，开始生成tfidf向量 ===')
        if not dictionary:
            logging.info('=== 未检测到有字典文件夹存在，开始生成字典 ===')
            dictionary = corpora.Dictionary.load(path_dictionary)
            logging.info('字典生成完成')
        os.makedirs(path_tmp_tfidf)
        files = loadFiles(path_doc_root)
        tfidf_model = models.TfidfModel(dictionary=dictionary)
        corpus_tfidf = {}
        for i, msg in enumerate(files):
            if i%n==0:
                catg    = msg[0]
                file    = msg[1]
                word_list = convert_doc_to_wordlist(file,cut_all=False)  #分词去停用词
                file_bow = dictionary.doc2bow(word_list)                  #doc2bow转化为词袋
                file_tfidf = tfidf_model[file_bow]                        #建立tf-idf
                tmp = corpus_tfidf.get(catg,[])   #返回指定键的值，如果值不在字典中返回default值
                tmp.append(file_tfidf)
                if tmp.__len__()==1:
                    corpus_tfidf[catg] = tmp
            if i%10000==0:
                logging.info('{i} files is dealed'.format(i=i))
        # 将tfidf中间结果储存起来
        catgs = list(corpus_tfidf.keys())
        for catg in catgs:
            corpora.MmCorpus.serialize('{f}{s}{c}.mm'.format(f=path_tmp_tfidf,s=os.sep,c=catg),
                                       corpus_tfidf.get(catg),
                                       id2word = dictionary
                                       )#corpora.MmCorpus.serialize 将corpus持久化到磁盘中
            logging.info('catg {c} has been transformed into tfidf vector'.format(c=catg))
        logging.info('=== tfidf向量已经生成 ===')
    else:
        logging.info('=== 检测到tfidf向量已经生成，跳过该阶段 ===')

    # # ===================================================================
    # # # # 第三阶段，  开始将tfidf转化成lsi
    #用lsi进行降维
    if not os.path.exists(path_tmp_lsi):
        logging.info('=== 未检测到有lsi文件夹存在，开始生成lsi向量 ===')
        if not dictionary:
            dictionary = corpora.Dictionary.load(path_dictionary)
        if not corpus_tfidf:
            logging.info('--- 未检测到tfidf文档，开始从磁盘中读取 ---')
            # 从对应文件夹中读取所有类别
            files = os.listdir(path_tmp_tfidf)
            catg_list = []
            for file in files:
                t = file.split('.')[0]
                if t not in catg_list:
                    catg_list.append(t)

            # 从磁盘中读取corpus
            corpus_tfidf = {}
            for catg in catg_list:
                path = '{f}{s}{c}.mm'.format(f=path_tmp_tfidf,s=os.sep,c=catg)
                corpus = corpora.MmCorpus(path)
                corpus_tfidf[catg] = corpus
            logging.info('--- tfidf文档读取完毕，开始转化成lsi向量 ---')

        # 生成lsi model
        os.makedirs(path_tmp_lsi)
        corpus_tfidf_total = []
        catgs = list(corpus_tfidf.keys())
        for catg in catgs:
            tmp = corpus_tfidf.get(catg)
            corpus_tfidf_total += tmp
        lsi_model = models.LsiModel(corpus = corpus_tfidf_total, id2word=dictionary, num_topics=50)
        # 将lsi模型存储到磁盘上
        lsi_file = open(path_tmp_lsimodel,'wb')
        pkl.dump(lsi_model, lsi_file)
        lsi_file.close()
        del corpus_tfidf_total # lsi model已经生成，释放变量空间
        logging.info('--- lsi模型已经生成 ---')

        # 生成corpus of lsi, 并逐步去掉 corpus of tfidf
        corpus_lsi = {}
        for catg in catgs:
            corpu = [lsi_model[doc] for doc in corpus_tfidf.get(catg)]
            corpus_lsi[catg] = corpu
            corpus_tfidf.pop(catg)
            corpora.MmCorpus.serialize('{f}{s}{c}.mm'.format(f=path_tmp_lsi,s=os.sep,c=catg),
                                       corpu,
                                       id2word=dictionary)
        logging.info('=== lsi向量已经生成 ===')
    else:
        logging.info('=== 检测到lsi向量已经生成，跳过该阶段 ===')

    # # ===================================================================
    # # # # 第四阶段，  分类
    if not os.path.exists(path_tmp_predictor):
        logging.info('=== 未检测到判断器存在，开始进行分类过程 ===')
        if not corpus_lsi: # 如果跳过了第三阶段
            logging.info('--- 未检测到lsi文档，开始从磁盘中读取 ---')
            files = os.listdir(path_tmp_lsi)
            catg_list = []
            for file in files:
                t = file.split('.')[0]
                if t not in catg_list:
                    catg_list.append(t)
            # 从磁盘中读取corpus
            corpus_lsi = {}
            for catg in catg_list:
                path = '{f}{s}{c}.mm'.format(f=path_tmp_lsi,s=os.sep,c=catg)
                corpus = corpora.MmCorpus(path)
                corpus_lsi[catg] = corpus
            logging.info('--- lsi文档读取完毕，开始进行分类 ---')

        tag_list = []
        doc_num_list = []
        corpus_lsi_total = []
        catg_list = []
        files = os.listdir(path_tmp_lsi)
        for file in files:
            t = file.split('.')[0]
            if t not in catg_list:
                catg_list.append(t)
        for count,catg in enumerate(catg_list):
            tmp = corpus_lsi[catg]
            tag_list += [count]*tmp.__len__()
            doc_num_list.append(tmp.__len__())
            corpus_lsi_total += tmp
            corpus_lsi.pop(catg)

        #通过scipy模块将数据处理为sklearn可训练的格式
        # 将gensim中的mm表示转化成numpy矩阵表示
        data = []
        rows = []
        cols = []
        line_count = 0
        for line in corpus_lsi_total:
            for elem in line:
                rows.append(line_count)
                cols.append(elem[0])
                data.append(elem[1])
            line_count += 1
        lsi_matrix = csr_matrix((data,(rows,cols))).toarray()
        # 生成训练集和测试集
        rarray=np.random.random(size=line_count)
        train_set = []
        train_tag = []
        test_set = []
        test_tag = []
        for i in range(line_count):
            if rarray[i]<0.8:
                train_set.append(lsi_matrix[i,:])
                train_tag.append(tag_list[i])
            else:
                test_set.append(lsi_matrix[i,:])
                test_tag.append(tag_list[i])

        # 生成分类器
        predictor = svm_classify(train_set,train_tag,test_set,test_tag)
        x = open(path_tmp_predictor,'wb')
        pkl.dump(predictor, x)
        x.close()
    else:
        logging.info('=== 检测到分类器已经生成，跳过该阶段 ===')

    # # ===================================================================
    # # # # 第五阶段，  对新文本进行判断
    if not dictionary:
        dictionary = corpora.Dictionary.load(path_dictionary)
    if not lsi_model:
        lsi_file = open(path_tmp_lsimodel,'rb')
        lsi_model = pkl.load(lsi_file)
        lsi_file.close()
    if not predictor:
        x = open(path_tmp_predictor,'rb')
        predictor = pkl.load(x)
        x.close()
    files = os.listdir(path_tmp_lsi)
    catg_list = []
    for file in files:
        t = file.split('.')[0]
        if t not in catg_list:
            catg_list.append(t)
    demo_doc = """
    新浪娱乐讯 3月29日，贾跃亭老婆甘薇[微博]更新一条微博：“是非曲直苦难辩，自有日月道分明。白衣惹灰土，只需心如故！”似乎是在倾诉心事。

　　自乐视资金链危机爆发以来，作为老板娘的甘薇受丈夫贾跃亭委托全权处理债务问题。1月3日，甘薇曾发布近四千字的长文《一位妻子的内心独白》。
甘薇在文中澄清，贾跃亭并没有放弃乐视，并且自己将作为委托人处理债务问题。现在个人及家庭负债累累，但是依然会坚持。文章一出，让人感动夫妻共患难的伉俪情深。
1月7日，甘薇发布微博称已为贾跃亭解决部分债务。
"""
    logging.info("原文本内容为：")
    logging.info(demo_doc)
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
    logging.info('分类结果为：{x}'.format(x=catg_list[x[0]]))
    x = catg_list[x[0]]
