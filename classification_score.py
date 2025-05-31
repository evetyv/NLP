# -*- coding: utf-8 -*-
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
import nltk
from sklearn.metrics import confusion_matrix

#%%Функция для предобработки текста (preprocess_text(text))

def preprocess_text(text):
    if not isinstance(text, str):
        return ''  
    # Токенизация
    tokens = nltk.word_tokenize(text.lower())
    # Удаление стоп-слов 
    tokens = [word for word in tokens if word.isalpha()]
    return ' '.join(tokens)



#%% выделение контекста заданного размера из предложения (extract_contexts(df, context_length))
import re

def extract_contexts(df, context_length):
 
    result_df = df.copy()
    result_df['context'] = None
    
    for idx, row in df.iterrows():
        keyword = row['Center']
        left_context = row['Left context']
        right_context = row['Right context']
        sentence = row['Full context']
        
        # Разбиваем контексты на слова
        left_context_words = left_context.split() if pd.notna(left_context) else []
        right_context_words = right_context.split() if pd.notna(right_context) else []
        
        # Получаем граничные слова контекстов
        last_left_word = left_context_words[-1] if left_context_words else ''
        first_right_word = right_context_words[0] if right_context_words else ''
        
        # Разбиваем предложение на слова (учитывая только буквенно-цифровые символы)
        sentence_words = re.findall(r'\b\w+\b', sentence)
        
        # Находим все позиции ключевого слова в предложении
        keyword_positions = [i for i, word in enumerate(sentence_words) if word == keyword]
        
        # Проверяем каждое вхождение ключевого слова
        for pos in keyword_positions:
            # Проверка левого контекста
            left_ok = True
            if left_context_words:
                if pos == 0:  
                    left_ok = False
                else:
                    left_word_in_sentence = sentence_words[pos - 1]
                    left_ok = (left_word_in_sentence == last_left_word)
            
            # Проверка правого контекста
            right_ok = True
            if right_context_words:
                if pos == len(sentence_words) - 1:  
                    right_ok = False
                else:
                    right_word_in_sentence = sentence_words[pos + 1]
                    right_ok = (right_word_in_sentence == first_right_word)
            
            # Если оба условия выполнены
            if left_ok and right_ok:
                # Вычисляем границы для извлечения
                start = max(0, pos - context_length)
                end = min(len(sentence_words), pos + context_length + 1)  # +1 чтобы включить ключевое слово
                
                # Извлекаем слова и объединяем в строку
                result_words = sentence_words[start:end]
                result_df.at[idx, 'context'] = ' '.join(result_words)
                break  # переходим к следующей строке после первого совпадения
    
    return result_df
#%% классификаторы func(df)


# Логистическая регрессия
def logistic_regression(df, score, con_len):
    X_train, X_test, y_train, y_test = train_test_split(df['context'], df['Label'], test_size=0.3, random_state=42)

    # Векторизация текста с использованием TF-IDF
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Обучение логистической регрессии
    model = LogisticRegression()
    model.fit(X_train_tfidf, y_train)

    # Прогнозирование и оценка качества
    y_pred = model.predict(X_test_tfidf)
    print(classification_report(y_test, y_pred))
    score.at[int(con_len-1), 'accuracy'] = classification_report(y_test, y_pred, output_dict=True)['accuracy']
    score.at[con_len-1, 'weighted_average'] = classification_report(y_test, y_pred, output_dict=True)['weighted avg']

    cm = confusion_matrix(y_test, y_pred)
    print('Confusion matrix\n',cm)
    
   
# SVM Метод опопрных векторов
def svm_classification(df, score, con_len):
    X_train, X_test, y_train, y_test = train_test_split(
        df['context'], df['Label'], test_size=0.3, random_state=42
    )

    # Векторизация текста с использованием TF-IDF
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Обучение SVM-классификатора 
    model = SVC(kernel='linear', random_state=42)
    model.fit(X_train_tfidf, y_train)

    # Предсказания и оценка качества
    y_pred = model.predict(X_test_tfidf)
    print(classification_report(y_test, y_pred))
    score.at[int(con_len-1), 'accuracy'] = classification_report(y_test, y_pred, output_dict=True)['accuracy']
    score.at[con_len-1, 'weighted_average'] = classification_report(y_test, y_pred, output_dict=True)['weighted avg']

    cm = confusion_matrix(y_test, y_pred)
    print('Confusion matrix\n',cm)
    
    
# Random Forest classification
def rf_classification(df, score, con_len):
    X_train, X_test, y_train, y_test = train_test_split(
        df['context'], df['Label'], test_size=0.3, random_state=42
    )

    # Векторизация текста с использованием TF-IDF
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Обучение модели случайного леса
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_tfidf, y_train)

    # Предсказания и оценка качества
    y_pred = model.predict(X_test_tfidf)
    print(classification_report(y_test, y_pred))
    score.at[int(con_len-1), 'accuracy'] = classification_report(y_test, y_pred, output_dict=True)['accuracy']
    score.at[con_len-1, 'weighted_average'] = classification_report(y_test, y_pred, output_dict=True)['weighted avg']

    cm = confusion_matrix(y_test, y_pred)
    print('Confusion matrix\n',cm)
    
    
# Ada Boost
from sklearn.ensemble import AdaBoostClassifier

def adaboost_classification(df, score, con_len):
    X_train, X_test, y_train, y_test = train_test_split(
        df['context'], df['Label'], test_size=0.3, random_state=42
    )

    # Векторизация текста с использованием TF-IDF
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Обучение AdaBoost-классификатора (используем базовый классификатор по умолчанию - DecisionTree)
    model = AdaBoostClassifier(n_estimators=50, random_state=42)
    model.fit(X_train_tfidf, y_train)

    # Предсказания и оценка качества
    y_pred = model.predict(X_test_tfidf)
    print(classification_report(y_test, y_pred))
    score.at[int(con_len-1), 'accuracy'] = classification_report(y_test, y_pred, output_dict=True)['accuracy']
    score.at[con_len-1, 'weighted_average'] = classification_report(y_test, y_pred, output_dict=True)['weighted avg']

    cm = confusion_matrix(y_test, y_pred)
    print('Confusion matrix\n',cm)
#%% main(df, func)
    
context_length = 30 # Максимальная длина контекста
def main(df, func):
    score = pd.DataFrame()
    score['context_length'] = None  
    score['accuracy'] = None
    score['weighted_average'] = None
    for i in range(1, context_length+1):
        df_contexts = extract_contexts(df,  i)
        score.at[i-1, 'context_length'] = i  
        print(f'Показатели для длины контекста', i)
        func(df_contexts,score, i)
    return score
#%% чтение файла, предобработка текста     
        
df = pd.read_excel(r"C:\Users\tvn4175\лук_датасет_processed.xlsx", usecols=['Left context','Center','Punct','Label','Right context', 'Full context'])
print('\nData types of the columns:')
print(df.dtypes)
print(df['Label'].value_counts())

df['Left context'] = df['Left context'].apply(preprocess_text)
df['Right context'] = df['Right context'].apply(preprocess_text)
df['Full context'] = df['Full context'].apply(preprocess_text)
df['Center'] = df['Center'].apply(preprocess_text)

#%%
print('Logistic regression')
score_logistic_reg = main(df, logistic_regression)

# Строим график
plt.figure(figsize=(8, 6))
plt.plot(score_logistic_reg['context_length'], score_logistic_reg['accuracy'], 
         marker='o',  
         linestyle='-',  
         color='blue',
         linewidth=2,
         markersize=8)

# Подписи осей и заголовок
plt.xlabel('context_length')
plt.ylabel('accuracy')
plt.title('динамика изменения accuracy с увеличением размера контекста (Logistic regression)')
plt.grid(True)  

plt.show()
#%%
print('SVM classification')
score_svm = main(df, svm_classification)
# Строим график
plt.figure(figsize=(8, 6))
plt.plot(score_svm['context_length'], score_svm['accuracy'], 
         marker='o',  
         linestyle='-',  
         color='blue',
         linewidth=2,
         markersize=8)

# Подписи осей и заголовок
plt.xlabel('context_length')
plt.ylabel('accuracy')
plt.title('динамика изменения accuracy с увеличением размера контекста (svm)')
plt.grid(True)  

plt.show()
#%%
print('Random Forest classification')
score_rf = main(df, rf_classification)
# Строим график
plt.figure(figsize=(8, 6))
plt.plot(score_rf['context_length'], score_rf['accuracy'], 
         marker='o',  
         linestyle='-',  
         color='blue',
         linewidth=2,
         markersize=8)

# Подписи осей и заголовок
plt.xlabel('context_length')
plt.ylabel('accuracy')
plt.title('динамика изменения accuracy с увеличением размера контекста (RF)')
plt.grid(True)  

plt.show()
#%%
print('Adaboost classification')
score_adaboost = main(df, adaboost_classification)
# Строим график
plt.figure(figsize=(8, 6))
plt.plot(score_adaboost['context_length'], score_adaboost['accuracy'], 
         marker='o',  
         linestyle='-',  
         color='blue',
         linewidth=2,
         markersize=8)

# Подписи осей и заголовок
plt.xlabel('context_length')
plt.ylabel('accuracy')
plt.title('динамика изменения accuracy с увеличением размера контекста (ADABoost)')
plt.grid(True)  

plt.show()

        
#%% K-means
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.model_selection import train_test_split
import pandas as pd

def kmeans_clustering(df, score, con_len):
    X_train, X_test, y_train, y_test = train_test_split(
        df['context'], df['Label'], test_size=0.3, random_state=42
    )
    
    # Векторизация текста
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Определяем количество кластеров
    n_clusters = len(df['Label'].unique())  
    
    # Инициализация и обучение K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X_train_tfidf)
    
    # Предсказание кластеров
    train_clusters = kmeans.predict(X_train_tfidf)
    test_clusters = kmeans.predict(X_test_tfidf)
    
    # Оценка качества кластеризации
    silhouette = silhouette_score(X_test_tfidf, test_clusters)
    ari = adjusted_rand_score(y_test, test_clusters)  # Сравнение с истинными метками
    
    print(f"\nОценки кластеризации для контекста длины {con_len}:")
    print(f"Silhouette Score: {silhouette:.4f}")
    print(f"Adjusted Rand Index: {ari:.4f}")
    
    # Запись результатов в датафрейм score
    score.at[int(con_len-1), 'silhouette'] = silhouette
    score.at[int(con_len-1), 'adjusted_rand'] = ari
    
#%%   
print('K-means clustering')
score_k_means = main(df, kmeans_clustering)

# Строим график
plt.figure(figsize=(8, 6))
plt.plot(score_k_means['context_length'], score_k_means['silhouette'], 
         marker='o',  
         linestyle='-',  
         color='blue',
         linewidth=2,
         markersize=8)

# Подписи осей и заголовок
plt.xlabel('context_length')
plt.ylabel('silhouette')
plt.title('динамика изменения silhouette с увеличением размера контекста')
plt.grid(True)  

plt.show()
#%%