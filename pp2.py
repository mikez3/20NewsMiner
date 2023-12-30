train_folder_path = '20news-bydate/20news-bydate-train'
test_folder_path = '20news-bydate/20news-bydate-test'

import re
import os
import pandas as pd
from tmtoolkit.bow.bow_stats import tfidf
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from statistics import mean


# Load the contents of each text file in the 20news-bydate-train folder
char_remov = ["Subject:", "Article-I.D.:", "Organization:", "Keywords:"]
data = []
filenames=[]
i=0
docs_categories = {}
all_categories = []
exclude_words = ["From:", "Lines:", "NNTP-Posting-Host:", "Distribution:", "Article-I.D."]
exclude_words_re = "|".join(exclude_words)
print('\n============= Start of Preprocessing =============\n')
for root, dirs, files in os.walk(train_folder_path):
    if (os.path.basename(root) != '20news-bydate-train'):
        all_categories.append(os.path.basename(root))
    filenames_percategory = []
    for file in files:
        with open(os.path.join(root, file), 'r', encoding='utf-8', errors='ignore') as f:
                lines=[]
                for line in f:
                    if not re.match(f"^({exclude_words_re})", line, re.IGNORECASE):
                        line = line.replace('|','')
                        line = line.replace('/',' ')
                        line = re.sub(r'\d+', '', line)
                        for str1 in char_remov:
                            line = line.replace(str1, '')
                        lines.append(line)
                content = ''.join(lines)
                if file in filenames:
                    print('-----SOS-----')
                    print(file, os.path.basename(root))
                    position = filenames.index(file)
                    print("Element found at position", position)
                filenames.append(i)
                filenames_percategory.append(i)
                data.append(content)
                docs_categories[i] = os.path.basename(root)
                i+=1

# ==========TEST SET===========
test_data = []
test_filenames=[]
i=0
test_docs_categories = {}
test_all_categories = []

for root, dirs, files in os.walk(test_folder_path):
    if (os.path.basename(root) != '20news-bydate-test'):
        test_all_categories.append(os.path.basename(root))
    test_filenames_percategory = []
    total_files = 0
    for file in files:
        total_files += 1
        with open(os.path.join(root, file), 'r', encoding='utf-8', errors='ignore') as f:
                lines=[]
                for line in f:
                    if not re.match(f"^({exclude_words_re})", line, re.IGNORECASE):
                        line = line.replace('|','')
                        line = line.replace('/',' ')
                        line = re.sub(r'\d+', '', line)
                        for str1 in char_remov:
                            line = line.replace(str1, '')
                        lines.append(line)
                content = ''.join(lines)
                test_filenames.append(i)
                test_filenames_percategory.append(i)
                test_data.append(content)
                test_docs_categories[i] = os.path.basename(root)
                i+=1

# ==============END================

# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

def clean_text(data):
    res = []
    for text in data:
        tokens = word_tokenize(text)
        tokens = [token.lower() for token in tokens if token.isalpha()]
        tokens = [token for token in tokens if not token in stop_words]
        tokens = [stemmer.stem(token) for token in tokens]
        # tokens = [lemmatizer.lemmatize(token) for token in tokens]
        content = ' '.join(tokens)
        res.append(content)
    return res

cleaned_data = clean_text(data)
print("Training Dataset Cleaned!")
test_cleaned_data = clean_text(test_data)
print("Testing Dataset Cleaned!")
print('\n============= End of Preprocessing =============')

token_dict = {}
test_token_dict = {}
for i in range (len(filenames)):
    token_dict[filenames[i]] = cleaned_data[i]
for i in range (len(test_filenames)):
    test_token_dict[test_filenames[i]] = test_cleaned_data[i]

print("\n============= Calculating TD-IDF Vectorizer (for train and test set) =============")
tfidf = TfidfVectorizer()
test_tfidf = TfidfVectorizer()
tfs = tfidf.fit_transform(token_dict.values())
test_tfs = test_tfidf.fit_transform(test_token_dict.values())

feature_names = tfidf.get_feature_names_out()
test_feature_names = test_tfidf.get_feature_names_out()

tfidf_df = pd.DataFrame(tfs.toarray(), columns=feature_names, index = filenames)
del tfs, feature_names
test_tfidf_df = pd.DataFrame(test_tfs.toarray(), columns=test_feature_names, index = test_filenames)
del test_tfs, test_feature_names

# Returns top-k words with the category in the list of word, top-k words without the category and the category of each doc
def top_k_words_train(df, k):
    i=0
    result = []
    categories = []
    words_no_cat = []
    words_with_cat = []
    for index, row in df.iterrows():
        sorted_row = row.sort_values(ascending=False)
        top_k = sorted_row[:k]
        words_original = top_k.index.tolist()
        words = top_k.index.tolist()
        words_no_cat.append(words_original)
        words_with_cat.append(words)
        words_with_cat[i].append(docs_categories[index])
        categories.append(docs_categories[index])
        result = words_with_cat
        i+=1
    return result, words_no_cat, categories
topWords, words_no_cat, words_cat = top_k_words_train(tfidf_df,15)

del tfidf_df
print("\n============= Finding the Top-15 words per document (for train and test set) =============")

def top_k_words_test(df, k):
    result = []
    test_true_categories = []
    for index, row in df.iterrows():
        sorted_row = row.sort_values(ascending=False)
        top_k = sorted_row[:k]
        words = top_k.index.tolist()
        test_true_categories.append(test_docs_categories[index])
        result.append(words)
    return result, test_true_categories
test_topWords, test_true_categories = top_k_words_test(test_tfidf_df,15)

print('\n============= Finding Freq. Itemsets and association rules =============')
print('============= ! RAM ALERT ! =============')

del test_tfidf_df
te_no_cat = TransactionEncoder()
te_ary_no_cat = te_no_cat.fit(words_no_cat).transform(words_no_cat)
df_no_cat = pd.DataFrame(te_ary_no_cat, columns=te_no_cat.columns_)
del  te_ary_no_cat, te_no_cat

frequent_itemsets_no_cat=apriori(df_no_cat, min_support=0.0031, use_colnames=True)
del df_no_cat

association_rules_train_no_cat = association_rules(frequent_itemsets_no_cat, metric="confidence", min_threshold=0.15)
print("\nFreq. itemsets with no category: \n", frequent_itemsets_no_cat)
print("\nAssociation Rules with no category: \n", association_rules_train_no_cat)
del frequent_itemsets_no_cat, association_rules_train_no_cat

te = TransactionEncoder()
te_ary = te.fit(topWords).transform(topWords)
print('\n============= Category Added =============')
df = pd.DataFrame(te_ary, columns=te.columns_)
del te_ary, te

frequent_itemsets=apriori(df, min_support=0.0031, use_colnames=True)
print("\nFreq. itemsets with category: \n",frequent_itemsets)
del df

association_rules_train = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.2)
del frequent_itemsets

frset_cat = frozenset(all_categories)
consequent_sele = association_rules_train[association_rules_train['consequents'].apply(lambda x: len(x)==1 and next(iter(x)) in frset_cat)]
del association_rules_train
print("\nAssociation Rules with category: \n",consequent_sele)

print('\n============= Making the Predictions =============')
x=0
def predict_category(rules, transaction):
    global x
    results = {}
    for i in test_all_categories:
        results[i] = 0
    for _, rule in rules.iterrows():
        for word in transaction:
            if word in rule['antecedents']:
                category = "".join(rule['consequents'])
                results[category] += 1
    if (max(results.values()) == 0):
        x+=1 
        return None
    return (max(results, key=results.get))

predicted_categories = []
predicted_categories_dict = {}
for transaction in test_topWords:
    category = predict_category(consequent_sele, set(transaction))
    predicted_categories.append(category)
print("X =", x)
all_stats={}
correct_predicted_stats={}
false_predicted_stats={}

for i in test_all_categories:
    all_stats[i] = 0
    correct_predicted_stats[i] = 0
    predicted_categories_dict[i] = 0
    false_predicted_stats[i] = 0

print('\n============= RESULTS =============\n')
for i in range (len(predicted_categories)):
    if (predicted_categories[i] == test_true_categories[i]):
        correct_predicted_stats[test_true_categories[i]] += 1
        predicted_categories_dict[predicted_categories[i]] += 1
    elif (not(predicted_categories[i] == None)):
        false_predicted_stats[predicted_categories[i]] += 1
        predicted_categories_dict[predicted_categories[i]] += 1
    all_stats[test_true_categories[i]] += 1

print("Original correct results:")
for key, value in all_stats.items():
        print(key, ' : ', value)

print('\n')
print("All Predictions:")
for key, value in predicted_categories_dict.items():
    print(key, ' : ', value)

print('\n')
print("Correct results (TP):")
for key, value in correct_predicted_stats.items():
        print(key, ' : ', value)

print('\n')
print("False Positives: (FP)")
for key, value in false_predicted_stats.items():
    print(key, ' : ', value)

print('\n')
print("False Negatives: (FN)")
for key, value in all_stats.items():
    print(key, ' : ', value- correct_predicted_stats[key])

precisions = []
recalls = []
# Precision, Recall per category (Macro Averaging)
for i in test_all_categories:
    TP = correct_predicted_stats[i]
    FP = false_predicted_stats[i]
    FN = all_stats[i] - correct_predicted_stats[i]
    precisions.append(TP / (TP + FP))
    recalls.append(TP / (TP + FN))
pre = round(mean(precisions),4)
rec = round(mean(recalls),4)
print('\n============= Precision and Recall =============')
print("\nPrecision: {} \nRecall: {}".format(pre,rec))
print('\n============= END =============\n')
