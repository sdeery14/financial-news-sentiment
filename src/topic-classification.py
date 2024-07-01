# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 17:34:55 2023

Group 2: Tessa Florek, Evan Helig, Dimetrius Johnson, and Sean Deery

Text Classification by Industry

Description: This code creates classifier to distinguish which industry topic a new article is talking about. The
industry topics the classifier will predict are agriculture, housing, stocks, manufacturing, and technology.

Data: Gathered from newsapi.org.
"""


# IMPORT LIBRARIES

# basics
import requests  ## for getting data from a server GET
import re   ## for regular expressions
import numpy as np
import pandas as pd    ## for dataframes and related
import random as rd
import os
import time
import datetime
import matplotlib.cbook
import warnings
warnings.filterwarnings("ignore")

# vectorization
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
import pyLDAvis.lda_model as LDAvis
import pyLDAvis

# visualizations
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import graphviz
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from graphviz import Source


# models
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import tree
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics import accuracy_score, classification_report

from sklearn import preprocessing
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import MDS
from scipy.cluster.hierarchy import ward, dendrogram






# COLLECT API DATA TO CSV

# returns a list of date ranges for the past 20 days
def get_date_ranges():
    
    # get todays date
    today = datetime.datetime.today()
    
    # create an empty list to store the date ranges
    date_ranges = []
    
    # get 20 date ranges
    for i in range(20):
        
        # append the dates to the list
        date_ranges.append((today.strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d")))
        
        # reset today as the start day
        today = today - datetime.timedelta(days=1)
    
    # return the list of tuples
    return date_ranges
    

# set the topics
topics=["agriculture", "housing", "stocks", "manufacturing", "technology"]
# set the endpoint base
endpoint="https://newsapi.org/v2/everything"
# set the data ranges
#date_ranges = [('2023-07-12', '2023-07-27'), ('2023-07-28', '2023-08-12')]
date_ranges = get_date_ranges()
# set the csv file name
filename="data/NewHeadlines.csv"



# =============================================================================
# # open the file
# my_file=open(filename,"w")  # "a"  for append   "r" for read
# # write in the column names
# column_names="LABEL,Date,Source,Title,Headline\n"
# my_file.write(column_names)
# my_file.close()
# 
# # function to clean text data
# def clean_api_text(string):
#     string=str(Title)
#     ## replace punctuation with space
#     ## accept one or more copies of punctuation        
#     ## plus zero or more copies of a space
#     ## and replace it with a single space
#     string=re.sub(r'[,.;@#?!&$\-\']+', ' ', str(string), flags=re.IGNORECASE)
#     string=re.sub(' +', ' ', str(string), flags=re.IGNORECASE)
#     string=re.sub(r'\"', ' ', str(string), flags=re.IGNORECASE)
#     # remove anything that is not alpabetical
#     string=re.sub(r'[^a-zA-Z]', " ", str(string), flags=re.VERBOSE)
#     # remove commas
#     string=string.replace(',', '')
#     string=' '.join(string.split())
#     string=re.sub("\n|\r", "", string)
#     return string
# 
# 
# # loop over each topic in each date range to collect the data
# for from_date, to_date in date_ranges:
#     for topic in topics:
#         
#         # set url post params
#         URLPost = {'apiKey':os.environ['NEWS_API_KEY'],
#                     'q':topic,
#                     'language':'en',
#                     'from': from_date,
#                     'to': to_date
#         }
#         
#         # get the data
#         response=requests.get(endpoint, URLPost)
#         
#         # parse to json
#         jsontxt = response.json()
#     
#         # open the file for append
#         MyFILE=open(filename, "a") ## "a" for append to add stuff
#         LABEL=topic
#         
#         # add each of the articles to the csv
#         for items in jsontxt["articles"]:
#             
#             # add the source
#             Source=items["source"]["name"]
#             
#             # set the date
#             Date=items["publishedAt"]
#             NewDate=Date.split("T")
#             Date=NewDate[0]
#             
#             # set the title
#             Title=items["title"]
#             Title=clean_api_text(Title)
#             
#             # set the headline
#             Headline=items["description"]
#             Headline=clean_api_text(Headline)
#         
#             # remove words less than 4 characters long
#             Headline = ' '.join([wd for wd in Headline.split() if len(wd)>3])
#             
#             # set the row data
#             WriteThis=str(LABEL)+","+str(Date)+","+str(Source)+","+ str(Title) + "," + str(Headline) + "\n"
#             
#             # write the file to the csv file
#             MyFILE.write(WriteThis)
#             
#         # close the file
#         MyFILE.close()
# =============================================================================



# Get the absolute path of the target directory
output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'images', 'topic'))

# Create the 'images/sentiment' directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)



# GET THE DATA


# read csv to dataframe
raw_text_df=pd.read_csv(filename)

# get the labels and text from the dataframe
raw_text_df = raw_text_df[['LABEL', 'Headline']]









# CLEAN and EXPLORE THE DATA

# function to clean the data
def clean_data(text_df):
    # create the string for the raw text html
    text_df_html = f"""
        <div>
            <p>Number of rows: {len(text_df)}</p>
            <p>Number of duplicate rows: {len(text_df[text_df.duplicated()])}</p>
            <p>Number of rows with NA: {text_df.isnull().values.ravel().sum()}
    """
    # remove any rows with NaN in them
    text_df = text_df.dropna()
    
    # remove any duplicates
    text_df = text_df.drop_duplicates()
    
    # display number of unique rows
    text_df_html += f"""
            <p>Final Number of rows: {len(text_df)}</p>
        </div>
    
    """
    
    return text_df, text_df_html


# function to explore the cleaned data
def explore_data(text_df):
    # add the counts df to the html
    counts = text_df.groupby('LABEL').count()
    # name the column Counts
    counts['Counts'] = counts['Headline']
    # drop the Headline column
    counts.drop(['Headline'], axis=1)
    # insert the label column
    counts.insert(0, "LABEL", counts.index)
    # add the table to the html
    text_df_html = f"""
        <div width=250>
            {counts[['LABEL', 'Counts']].to_html(justify='center',
                                                  index=False, 
                                                  classes=['table-striped', 
                                                           'table-bordered', 
                                                           'table-sm'
                                                           ])}
        </div>
    """

    # create the countplot to the html
    label_countplot = sns.countplot(x=text_df.LABEL)
    plt.savefig("images/topic/Label_Countplot.png")

    text_df_html += """
        <div class="mt-5"  width=250>
            <img src='../images/topic/Label_Countplot.png' class='img-fluid'>
        </div>
    """

    # group the data by 'category' and get the first row of each group
    first_rows = text_df.groupby('LABEL').first()
    first_rows.insert(0, "LABEL", first_rows.index)

    # display 
    text_df_html += f"""
        <div class=mt-5>
            {first_rows.to_html(justify='center',
                index=False, 
                classes=['table-striped', 
                         'table-bordered', 
                         'table-sm'
                         ])}
        </div>
    """
    return text_df_html

# clean the data and get the html
raw_text_df, raw_text_df_html = clean_data(raw_text_df)

# explore the data and get the html
raw_text_df_html += explore_data(raw_text_df)









# FUNCTIONS TO PREPROCESS AND ANALYZE THE DATA

# function to vectorize the headlines
def preprocess_data(text_df, vect):
    
    # start the timer
    start = time.time()
    # print the vectorizer name
    print(f"Preprocessing with {type(vectorizer).__name__}")
    
    # get a list of the headlines
    headline_list=text_df["Headline"].values.tolist()
    # get a list of the labels
    label_list=text_df["LABEL"].values.tolist()
    # vectorize the data into a sparse matrix
    my_matrix = vect.fit_transform(headline_list)
    # get the column names
    column_names = vect.get_feature_names_out()
    # build the data frame
    vec_df=pd.DataFrame(my_matrix.toarray(),columns=column_names)
    # add labels to df
    vec_df.insert(0, "LABEL", pd.Series(label_list))
    # store the data in a csv
    #vec_df.to_csv(f"{str(type(vectorizer)).split('.')[-1][:-2]}_DF.csv")
    
    # end the time
    end = time.time()
    # print the time
    print(end - start)
    
    return vec_df, my_matrix


# function to create wordclouds for each topic
def create_wordclouds(df, vectorizer_name):
    
    # start the timer
    start = time.time()
    # print the vectorizer name
    print(f"Creating Wordclouds for {type(vectorizer).__name__}")
    
    # get the counts of each word
    temp_df = df.sum(axis=0,numeric_only=True)
    # create the wordcloud including all of the topics
    wordcloud = WordCloud().generate_from_frequencies(temp_df)
    # save the wordcloud
    wordcloud.to_file(f"images/topic/{vectorizer_name}_Wordcloud.png")
    
    
    # create a directory to store the images
    if not os.path.exists(f"images/topic/{vectorizer_name}_Wordclouds"): os.mkdir(f"images/topic/{vectorizer_name}_Wordclouds")
    
    # create an empty wordcloud list
    wordcloud_list = []
    
    # loop over each topic and generate the wordclouds
    for count, topic in enumerate(topics):
        # filter the data by topic
        temp_df = df[df['LABEL'] == topic]
        # get the counts of each word
        temp_df =temp_df.sum(axis=0,numeric_only=True)
        # create and store in a list the wordcloud OBJECTS
        wordcloud = WordCloud().generate_from_frequencies(temp_df)
        # add the wordcloud to the list
        wordcloud_list.append(wordcloud)
        # save the wordcloud
        wordcloud.to_file(f"images/topic/{vectorizer_name}_Wordclouds/{topic}_wordcloud.png")
        
    # end the time
    end = time.time()
    # print the time
    print(end - start)



# function to explore the vectorized data and create html output
def explore_vectorized_df(vect, df):
    # add the vectorizer html to the string
    df_html = f"""
    <h2>{type(vect).__name__} Data</h2>
    {df.head().to_html(index=False, classes="table table-striped")}
    <div width=250>
        <img src='../images/topic/{type(vect).__name__}_Wordcloud.png' class='img-fluid'>
    </div>
    """
    for count, filename in enumerate(os.listdir(f"images/topic/{type(vectorizer).__name__}_Wordclouds")):
        df_html += f"""
        <h3>{filename}</h3>
        <div width=250>
            <img src='../images/topic/{type(vect).__name__}_Wordclouds/{filename}' class='img-fluid'>
        </div>
        """
    return df_html


# function to complete lda analysis
def lda_analysis(vectorizer, data_matrix):
    
    # start the timer
    start = time.time()
    # print the vectorizer name
    print(f"LDA Analysis for {type(vectorizer).__name__}")
    
    # create the LDA model
    lda_model = LatentDirichletAllocation(n_components=5, max_iter=100, learning_method='online')
    lda_Z_DF = lda_model.fit_transform(data_matrix)
    
    # create html visualization
    #pyLDAvis.enable_notebook() ## not using notebook
    panel = LDAvis.prepare(lda_model, data_matrix, vectorizer,  mds='tsne')
    pyLDAvis.save_html(panel, f"images/topic/{type(vectorizer).__name__}_InTheNews.html")
    
    # end the time
    end = time.time()
    # print the time
    print(end - start)
    
    
def train_model(model, parameters, X_train, y_train):
    # start the timer
    start = time.time()
    # print the vectorizer name
    print(f"Training {type(model).__name__}")
    
    # train the grid search and append it to the list
    grid = GridSearchCV(estimator=model, 
                        param_grid=parameters, 
                        cv=3,
                        verbose=1).fit(X_train, y_train)
    # end the time
    end = time.time()
    # print the time
    print(end - start)
    
    return grid


# get frequency importance
def get_feature_importance(gridsearch_cv, vect):
    
    results_html = ""
    
    if type(gridsearch_cv.best_estimator_).__name__ == "MultinomialNB":
        
        # get the sorted log probabiltites for each class
        agriculture_prob_sorted = gridsearch_cv.best_estimator_.feature_log_prob_[0, :].argsort()[::-1]
        housing_prob_sorted = gridsearch_cv.best_estimator_.feature_log_prob_[1, :].argsort()[::-1]
        manufacturing_prob_sorted = gridsearch_cv.best_estimator_.feature_log_prob_[2, :].argsort()[::-1]
        stocks_prob_sorted = gridsearch_cv.best_estimator_.feature_log_prob_[3, :].argsort()[::-1]
        technology_prob_sorted = gridsearch_cv.best_estimator_.feature_log_prob_[4, :].argsort()[::-1]
        
        # get the top 10 words for each class
        agriculture_10 = np.take(vect.get_feature_names_out(), agriculture_prob_sorted[:10])
        housing_10 = np.take(vect.get_feature_names_out(), housing_prob_sorted[:10])
        manufacturing_10 = np.take(vect.get_feature_names_out(), manufacturing_prob_sorted[:10])
        stocks_10 = np.take(vect.get_feature_names_out(), stocks_prob_sorted[:10])
        technology_10 = np.take(vect.get_feature_names_out(), technology_prob_sorted[:10])
        
        # add the html
        results_html = f"""
            <p>Top agriculture words: {agriculture_10}</p>
            <p>Top housing words: {housing_10}</p>
            <p>Top manufacturing words: {manufacturing_10}</p>
            <p>Top stocks words: {stocks_10}</p>
            <p>Top technology words: {technology_10}</p>
        """
        
    elif type(gridsearch_cv.best_estimator_).__name__ == 'DecisionTreeClassifier':
        
        feature_importances = gridsearch_cv.best_estimator_.feature_importances_  
        indices = np.argsort(feature_importances)[::-1]
        
        results_html = "<p>Important Features</p>"
        
        ## print out the important features.....
        for f in range(len(indices)):
            if feature_importances[indices[f]] > 0:
                results_html += f"<p>{vect.get_feature_names_out()[indices[f]]}: {feature_importances[indices[f]]}</p>"
        
        
    return results_html



def decision_tree_visualization(gridsearch_cv, vect):
    dtree_html = ""
    # create decision tree images
    if type(model).__name__ == 'DecisionTreeClassifier':
        # Convert the decision tree to a DOT format
        dot = tree.export_graphviz(gridsearch_cv.best_estimator_, feature_names=vect.get_feature_names_out(),
                  class_names=['agriculture', 'housing', 'manufacturing', 'stocks', 'technology'],
                  filled=True,
                  rounded=True,
                  special_characters=True)
        
        graph = Source(dot)
        graph.format = 'png'
        graph.render(f'images/topic/{type(vect).__name__}_DecisionTree')
        
        # add the decision tree to the html
        dtree_html = f"""
            <h4>Decision Tree</h4>
            <img src='../images/topic/{type(vect).__name__}_DecisionTree.png' class='img-fluid'>
        """
        
    return dtree_html






# get the model results html
def get_model_results_html(X_test, y_test, test_prediction):
    
    # create the confusion matrix
    cm = ConfusionMatrixDisplay.from_estimator(grid.best_estimator_, X_test, y_test)
    # plot the confusion matrix
    cm.plot(xticks_rotation=30)
    # set tight layout
    plt.tight_layout()
    # save the image
    plt.savefig(f'images/topic/{type(vectorizer).__name__}_{type(model).__name__}_ConfusionMatrix.png')
    
    
    # classification report
    class_report = classification_report(y_test, 
                                         test_prediction, 
                                         target_names=topics.sort(), 
                                         output_dict=True)
    class_report_df = pd.DataFrame.from_dict(class_report).transpose()
    class_report_df_html = class_report_df.to_html(justify='center', 
                                                   index=True, 
                                                   classes=['table-striped', 
                                                            'table-bordered', 
                                                            'table-sm'])
    
    # add the confusion matrix html
    results_html = f"""
        <h3>{type(vectorizer).__name__}_{type(model).__name__}</h3>
        <div>
            <img src='../images/topic/{type(vectorizer).__name__}_{type(model).__name__}_ConfusionMatrix.png' class='img-fluid'>
        </div>
        {class_report_df_html}
    """
    
    # get feature importance
    results_html += get_feature_importance(grid, vectorizer)
    
    # get decision tree visualization
    results_html += decision_tree_visualization(grid, vectorizer)
    
    return results_html






# evaluate the model
def evaluate_model(grid, X_train, X_test, y_train, y_test):
    
    # predict on train and test
    train_prediction = grid.predict(X_train)
    test_prediction = grid.predict(X_test)
    
    # calculate accuracy for train and test
    train_accuracy = accuracy_score(y_train, train_prediction)
    test_accuracy = accuracy_score(y_test, test_prediction)
    
    # add the results to the list
    results_list = [type(vectorizer).__name__,
            type(grid.best_estimator_).__name__,
            grid.best_score_, 
            train_accuracy, 
            test_accuracy]
    
    # get the model html
    model_html = get_model_results_html(X_test, y_test, test_prediction)
    
    # return the metrics in a list with the name of the vecotrizer and model
    return (results_list, model_html)

















# SETTINGS FOR THE ANALYSIS

# add the topics to the stop english words
my_stop_words = list(text.ENGLISH_STOP_WORDS.union(topics))

# create the list of instantiated vectorizers
my_vectorizers = [
    CountVectorizer(
            input="content",
            lowercase=True,
            stop_words = my_stop_words,
            max_features=1500
            ),
    TfidfVectorizer(
            input="content",
            lowercase=True,
            stop_words = my_stop_words,
            max_features=1500
            )
    ]

# create the list of models
my_models = [
    (MultinomialNB(force_alpha=True),
     {'alpha': [0.1, 0.5, 1],
     }),
    (SVC(kernel='linear'),
      {'kernel': ['linear', 'rbf', 'poly'],
      'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}),
    (DecisionTreeClassifier(max_depth=20),
      {'criterion': ['gini', 'entropy', 'log_loss'],
      'min_samples_split': [10, 50, 100, 300],
      'random_state': [1,10,100]})
    ]



# create an empty string for the vectorizer html
cleaned_df_html = ""
# create an e,pty string for the model html
model_results_html = ""
# create an empty dataframe for the results
model_results_df = pd.DataFrame(columns=["Vectorizer",  
                                   "Model",
                                   "CV Accuracy",
                                   "Train Accuracy", 
                                   "Test Accuracy"])


# for loop runs the analysis on each vectorizer
for vectorizer in my_vectorizers:
     
    # vectorize the data into a dataframe, also get the matrix of text for lda
    cleaned_df, data_matrix = preprocess_data(raw_text_df, vectorizer)
    
    # create and save the wordclouds for the vectorizer
    create_wordclouds(cleaned_df, type(vectorizer).__name__)
    
    # create html output
    cleaned_df_html += explore_vectorized_df(vectorizer, cleaned_df)
    
    # create the LDA model
    lda_analysis(vectorizer, data_matrix)
    
    # train test split
    X_train, X_test, y_train, y_test = train_test_split(cleaned_df.drop("LABEL", axis=1), 
                                                        cleaned_df.LABEL, 
                                                        test_size=0.33, 
                                                        random_state=42)
    
    # for loop runs data through each model
    for model, parameters in my_models:
        
        # train the model
        grid = train_model(model, parameters, X_train, y_train)
        
        # evaluate the model and get a list of results
        model_results_list, results_html = evaluate_model(grid, X_train, X_test, y_train, y_test)
        
        # add the best model from the grid to the results dataframe along with details
        model_results_df.loc[len(model_results_df.index)] = model_results_list
        
        # append the model html
        model_results_html += results_html
        
        
        
    
# create the results dataframe
model_results_df_html = model_results_df.sort_values(
    by=["Test Accuracy"], 
    ascending=False).to_html(
        justify='center',
        index=False, 
        classes=['table-striped', 'table-bordered', 'table-sm'])




        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
# CREATE AN HTML REPORT

# to open/create a new html file in the write mode
f = open('reports/topic-report.html', 'w')
  
# the html code which will go in the file GFG.html
html = f"""
<!doctype html>
<html lang="en" data-bs-theme="dark">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Classification Report</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.1/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-4bw+/aepP/YC94hEpVNVgiZdgIC5+VKNBQNGCHeKRQN+PtmoHDEXuppvnDJzQIu9" crossorigin="anonymous">
  </head>
  <body>
      <main role="main" class="container">

        <div class="starter-template">
  
            <h1 class="display-1 text-center">Text Classification by Industry</h1>
            
            <div class="row mt-5 border">
                <h2>Raw Text Data</h2>
                {raw_text_df_html}
            </div>
            
            <div class="row mt-5 border">
                <h2>Model Results</h2>
                {model_results_df_html}
            </div>
            
            <div class="row mt-5 border">
                <h2>Vectorizers</h2>
                {cleaned_df_html}
            </div>
            
            <div class="row mt-5 border">
                <h2>Models</h2>
                {model_results_html}
            </div>
            
        </div>
    </main>
    
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.1/dist/js/bootstrap.bundle.min.js" integrity="sha384-HwwvtgBNo3bZJJLYd8oVXjrBZt8cqVSpeBNS5n7C8IVInixGAoxmnlMuBnhbgrkm" crossorigin="anonymous"></script>
    <script src="https://cdn.jsdelivr.net/npm/@popperjs/core@2.11.8/dist/umd/popper.min.js" integrity="sha384-I7E8VVD/ismYTF4hNIPjVp/Zjvgyol6VFvRkX/vR+Vc4jQkC+hVqc2pM8ODewa9r" crossorigin="anonymous"></script>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.1/dist/js/bootstrap.min.js" integrity="sha384-Rx+T1VzGupg4BHQYs2gCW9It+akI2MM/mndMCy36UVfodzcJcF0GGLxZIzObiEfa" crossorigin="anonymous"></script>
  </body>
</html>
"""
  
# writing the code into the file
f.write(html)
  
# close the file
f.close()