from ckipclient import CKIPClient
from sklearn.feature_extraction.text import CountVectorizer
import os
#import sys
#sys.path.append("C:\\Users\\user\\Desktop\\專題\\演算法_報告\\CKIP中文斷詞分析\\CKIPWS_20190107\\CKIPWS")
#import CKIPWS as newc
import math
import numpy as np
import jieba

'''def newsegmentText (NumberOfText , AllText):#斷詞   
    ini = 'ws.ini'
    main_dll = 'CKIPWS.dll'
    py_dll = 'PY_CKIPWS.dll'
    newc.initial(main_dll, py_dll, ini)
    Result = []    
    #ckip = CKIPClient("140.109.19.104" , 1501 , 'aaa1aaa' , 'bbb2bbb')    
    for i in range(NumberOfText):
        print(i)
        segment_results = newc.segment(str(AllText[i]))#把list轉成string,才能分割
        string = ""#segmentresult包含詞跟詞性 ,分割成word[0]=單詞,word[1]=詞性
        for sentence in segment_results:    
            for word in sentence:
                string=string+word[0]+" "#word[0]以空白間隔存在string裡
        Result.append(string)#string丟到Result裡
    return Result'''

def readText (path,Direct,Total):#讀檔案
    Text = []
    for index in range(Total):    
        tempList=[]           #path為python檔所在的資料夾,Direct為資料夾裡的資料
        with open(os.path.join(path,Direct[index]),"r") as f:#把讀進來的檔案命名為f
            for i in f:                          
                tempList.append(i)
        Text.append(tempList)#把templist裡的東西丟到text 讓text有8個文檔
    return Text

def cut(NumberOfText , AllText):
    Result = []      
    for i in range(NumberOfText):
        segment_results = jieba.cut(str(AllText[i]))
        string = ""#segmentresult包含詞跟詞性 ,分割成word[0]=單詞,word[1]=詞性
        for sentence in segment_results:               
            string=string+sentence+" "#word[0]以空白間隔存在string裡
        Result.append(string)#string丟到Result裡
    return Result
            
                    #8          #所有檔案
def segmentText (NumberOfText , AllText):#斷詞    
    Result = []    
    #ckip = CKIPClient(HOST , PORT , ACCOUNT , PASSWORD)     # change this line to your account
    for i in range(NumberOfText):
        print(i)
        segment_results = ckip.segment(str(AllText[i]))#把list轉成string,才能分割
        string = ""#segmentresult包含詞跟詞性 ,分割成word[0]=單詞,word[1]=詞性
        for sentence in segment_results:    
            for word in sentence:
                string=string+word[0]+" "#word[0]以空白間隔存在string裡
        Result.append(string)#string丟到Result裡
    return Result


def StandardTFvector (NumberOfText,TextVector):#標準化TF向量,TextVector=Result的向量形式
    TF_matrix = []    
    for i in range(NumberOfText): #i=row(文章數) u=column(詞數)          
        standard = 0
        for u in range(len(TextVector.toarray()[0])):#有幾個col跑幾次
            if TextVector.toarray()[i][u] != 0:#if詞有出現就記1次
                standard+=pow(TextVector.toarray()[i][u],2)#power(,2)=2次方
                
        tempList=[]    
        standard = pow(standard,0.5)#power(,0.5)=開根號
        #standard=分母=出現次數平方加總開根號
        for u in range(len(TextVector.toarray()[0])):#總辭彙數
            if TextVector.toarray()[i][u] != 0:
                tempList.append(TextVector.toarray()[i][u]/standard)#templist=標準化TF                
            else:              #TextVector.toarray()[i][u]=分子=詞u出現次數
                tempList.append(0)
        TF_matrix.append(tempList)#templist            
    
    TF = []
    for i in range(len(TextVector.toarray()[0])):#i=column(詞數)
        Sum = 0
        for u in range(NumberOfText):#u=row(文章數)
            Sum+=TF_matrix[u][i]#TF值加總
        TF.append(Sum)#把sum丟到TF裡
    return TF


def PDFvector (NumberOfText,NumberOfTerm,TextVector):#PDF向量
    PDF = []
    for i in range(NumberOfTerm):#詞數
        count = 0
        for u in range(NumberOfText):#文章數
            if TextVector.toarray()[u][i] != 0:
                count += 1#計算出現的文章數
        PDF.append(round(math.exp(count/NumberOfText),3))#取到小數3位
    return PDF#PDF=exp(出現次數/文章總數)


def sort (Title , array):#排序    
    TempDic = {}
    count=0
    for i in Title:        
        TempDic[i] = array[count]
        count+=1    
    sorted_dic = [(k,v) for k,v in TempDic.items()]    
    sorted_dic.sort(key = takeSecond,reverse = True)    
    return sorted_dic
    
def takeSecond ( element ) :
    return element [ 1 ]


if __name__ == "__main__":#主程式
    all_TextInDirect = os.listdir(".")#資料夾的文件數(8個txt+1個py)
    Total = len(all_TextInDirect)-1#扣掉py=8
    path = os.getcwd()    #取得路徑
    
    AllTextList = readText(path,all_TextInDirect,Total)#ALLTextList=所有讀進來的檔案   
    SegmentResultText = segmentText(Total,AllTextList)#SegmentResultText=分割完的結果
    
    vectorizer = CountVectorizer()#轉成向量形式的方法
    TextVector = vectorizer.fit_transform(SegmentResultText)#TextVector=分割結果的向量形式
    FeatureWord= vectorizer.get_feature_names()#FeatureWord=詞(名稱)
    
    TF = StandardTFvector(Total,TextVector)
    PDF = PDFvector(Total,len(FeatureWord),TextVector)
    
    TFPDF = np.asarray(TF)*np.asarray(PDF)#對應位置相乘
    sortedTFPDF = sort(FeatureWord,TFPDF)#名稱和TFPDF排序
    
    
    print("TF*PDF計算值由高到低前50筆如下 :")
    for i in range(50):
        print("%-2s\t%-2s%3.4f" %(sortedTFPDF[i][0],":",sortedTFPDF[i][1]))
    
        







