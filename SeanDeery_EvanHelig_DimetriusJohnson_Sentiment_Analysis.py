# -*- coding: utf-8 -*-
"""
Created on Sun Sep  3 22:15:21 2023

@author: sdeer
"""

# IMPORT LIBRARIES

# basic
import os
import numpy as np
import pandas as pd
import time
import warnings
warnings.filterwarnings("ignore")

# visualization
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import pyLDAvis.lda_model as LDAvis
import pyLDAvis
from graphviz import Source


# models
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.decomposition import LatentDirichletAllocation

# metrics
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, classification_report






# GET THE LABELED SENTIMENT DATA
raw_df = pd.read_csv('sentiment_data.csv')

# rename the columns to Text and Label
raw_df= raw_df.rename(columns={"Sentence": "Text", "Sentiment": "Label"})









# CLEAN AND EXPLORE SENTIMENT DATA

# function to remove duplicates and rows with missing values
def remove_duplicates_na(df):
    # create the string for the raw text html
    df_html = f"""
        <div>
            <p>Number of rows: {len(df)}</p>
            <p>Number of duplicate rows: {len(df[df.duplicated()])}</p>
            <p>Number of rows with NA: {df.isnull().values.ravel().sum()}
    """
    # remove any rows with NaN in them
    df = df.dropna()
    
    # remove any duplicates
    df = df.drop_duplicates()
    
    # display number of unique rows
    df_html += f"""
            <p>Final Number of rows: {len(df)}</p>
        </div>
    
    """
    
    return df, df_html


# function to return distribution data and visualizations as html
def explore_distribution(df, df_name):
    
    # get the counts of each label
    counts = df.groupby('Label').count()
    # name the column Counts
    counts['Counts'] = counts['Text']
    # drop the Headline column
    counts = counts.drop(['Text'], axis=1)
    # insert the label column
    counts.insert(0, "Label", counts.index)
    # add the table to the html
    df_html = f"""
        <div width=250>
            {counts.to_html(justify='center',
                            index=False, 
                            classes=['table-striped', 
                                     'table-bordered', 
                                     'table-sm'
                                     ])}
        </div>
    """
    
    print(len(df))
    
    # create the countplot to the html
    sns.countplot(x=df["Label"])
    plt.savefig(f"Sentiment_Images/{df_name}_Label_Countplot.png")
    plt.clf()

    df_html += f"""
        <div class="mt-5"  width=250>
            <img src='Sentiment_Images/{df_name}_Label_Countplot.png' class='img-fluid'>
        </div>
    """

    # group the data by 'category' and get the first row of each group
    first_rows = df.groupby('Label').first()
    first_rows.insert(0, "Label", first_rows.index)

    # display 
    df_html += f"""
        <div class=mt-5>
            {first_rows.to_html(justify='center',
                index=False, 
                classes=['table-striped', 
                         'table-bordered', 
                         'table-sm'
                         ])}
        </div>
    """
    return df_html


# function to balance the labels across the dataset
def balance_labels(df):
    
    # add the counts df to the html
    label_groups = df.groupby('Label')
    
    # sample each label the same amount
    balanced_df = label_groups.apply(lambda x: x.sample(label_groups.size().min()))
    
    # remove the index
    balanced_df = balanced_df.reset_index(drop=True)
    
    return balanced_df


# remove duplicates and rows with missing values
raw_text_df, raw_text_html = remove_duplicates_na(raw_df)

# display the distribution and get the html
raw_text_html += explore_distribution(raw_text_df, "BaseDF")

# balance the labels across the dataset
balanced_raw_text_df = balance_labels(raw_text_df)

# display the balanced data distribution
raw_text_html += explore_distribution(balanced_raw_text_df, "BalancedDF")

# create the list of dataframes
text_df_list = [
    (raw_text_df, "Base"),
    (balanced_raw_text_df, "Balanced")
    ]







# CREATE THE MODELS

# function to create wordclouds for the data as a whole and then for each label
def create_wordclouds(vect_df, vect_name):
    
    # start the timer
    start = time.time()
    # print the vectorizer name
    print(f"Creating Wordclouds for {vect_name}")
    
    # get the counts of each word
    temp_df = vect_df.sum(axis=0,numeric_only=True)
    # create the wordcloud including all of the topics
    wordcloud = WordCloud().generate_from_frequencies(temp_df)
    # save the wordcloud
    wordcloud.to_file(f"Sentiment_Images/{vect_name}_Wordcloud.png")
    
    wc_html = f"""
        <div width=250>
            <img src='Sentiment_Images/{vect_name}_Wordcloud.png' class='img-fluid'>
        </div>
    """
    
    # create a directory to store the images
    if not os.path.exists(f"Sentiment_Images/{vect_name}_Wordclouds"): 
        os.mkdir(f"Sentiment_Images/{vect_name}_Wordclouds")
    
    # loop over each topic and generate the wordclouds
    for count, label in enumerate(vect_df.Label.unique()):
        # filter the data by label
        temp_df = vect_df[vect_df['Label'] == label]
        # get the counts of each word
        temp_df = temp_df.sum(axis=0,numeric_only=True)
        
        # create and store in a list the wordcloud OBJECTS
        wordcloud = WordCloud().generate_from_frequencies(temp_df)

        # save the wordcloud
        wordcloud.to_file(f"Sentiment_Images/{vect_name}_Wordclouds/{label}_wordcloud.png")
        
        # add the image to the html
        wc_html += f"""
            <h3>{label}</h3>
            <div width=250>
                <img src='Sentiment_Images/{vect_name}_Wordclouds/{label}_wordcloud.png' class='img-fluid'>
            </div>
        """
        
    # end the time
    end = time.time()
    # print the time
    print(end - start)
    
    return wc_html




# function to vectorize the text
def vectorize_data(df, vect_name, vect):
    
    # start the timer
    start = time.time()
    
    # get the vectorizer name
    vect_name = type(vect).__name__
    
    # print the vectorizer name
    print(f"Vectorizing with {vect_name}")
    
    # get a list of the text
    text_list=df["Text"].values.tolist()
    # get a list of the labels
    label_list=df["Label"].values.tolist()
    
    # vectorize the data into a sparse matrix
    matrix = vect.fit_transform(text_list)
    
    # get the column names
    column_names = vect.get_feature_names_out()

    # build the data frame
    vect_df=pd.DataFrame(matrix.toarray(),columns=column_names)
    
    # remove any numeric tokens
    for col in vect_df.columns:
        if any(char.isdigit() for char in col) == True:
            vect_df = vect_df.drop([col], axis=1)
            
    # add labels to df
    vect_df.insert(0, "Label", pd.Series(label_list))

    # end the time
    end = time.time()
    # print the time
    print(f"{end - start} seconds")
    
    # add the vectorizer html to the string
    vect_html = f"""
    <h2>{vect_name} Data</h2>
    {vect_df.head().to_html(index=False, classes="table table-striped")}
    """
    
    vect_html += create_wordclouds(vect_df, vect_name)
    
    return vect, vect_df, matrix, vect_html





# function to complete lda analysis
def lda_analysis(vect, matrix, df_name):
    
    # start the timer
    start = time.time()
    
    # get the vectorizer name
    vect_name = type(vect).__name__
    
    # print the vectorizer name
    print(f"LDA Analysis for {vect_name}")
    
    # create the LDA model
    lda_model = LatentDirichletAllocation(n_components=5, max_iter=100, learning_method='online')
    lda_model.fit_transform(matrix)
    
    # create html visualization
    pyLDAvis.enable_notebook() ## not using notebook
    panel = LDAvis.prepare(lda_model, matrix, vect,  mds='tsne')
    pyLDAvis.save_html(panel, f"Sentiment_Images/{df_name}_{vect_name}_InTheNews.html")
    
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
                        verbose=1,
                        refit=True).fit(X_train, y_train)
    # end the time
    end = time.time()
    # print the time
    print(end - start)
    
    return grid


# get frequency importance
def get_feature_importance(gridsearch_cv, vect):
    
    results_html = ""
    
    # Naive Bayes log probabilities
    if type(gridsearch_cv.best_estimator_).__name__ == "MultinomialNB":
        
        # get the labels
        labels = gridsearch_cv.best_estimator_.classes_
        
        results_html = "<p>Important Features</p>"
        
        # get the log probabilities for each label and get the top 10 words
        for count, label in enumerate(labels):
            # get the sorted log probabiltites
            prob_sorted = gridsearch_cv.best_estimator_.feature_log_prob_[count, :].argsort()[::-1]
            # get the top 10 words for each class
            top_10 = np.take(gridsearch_cv.best_estimator_.feature_names_in_, prob_sorted[:10])
            # add the top words to html
            results_html += f"<p>Top {label} words: {top_10}</p>"
        
    # Decision Tress feature importance
    elif type(gridsearch_cv.best_estimator_).__name__ == 'DecisionTreeClassifier':
        
        # get the feature importances
        feature_importances = gridsearch_cv.best_estimator_.feature_importances_  
        # sort the feature importances
        indices = np.argsort(feature_importances)[::-1]
        
        results_html = "<p>Important Features</p>"
        
        # add feature importances to the html
        for f in range(len(indices)):
            if feature_importances[indices[f]] > 0:
                results_html += f"<p>{vect.get_feature_names_out()[indices[f]]}: {feature_importances[indices[f]]}</p>"
        
        
    return results_html



def decision_tree_visualization(gridsearch_cv, vect, vect_name):
    dtree_html = ""
    # create decision tree images
    if type(gridsearch_cv.best_estimator_).__name__ == 'DecisionTreeClassifier':
        # Convert the decision tree to a DOT format
        dot = tree.export_graphviz(gridsearch_cv.best_estimator_, feature_names=grid.feature_names_in_,
                  class_names=['agriculture', 'housing', 'manufacturing', 'stocks', 'technology'],
                  filled=True,
                  rounded=True,
                  special_characters=True)
        
        graph = Source(dot)
        graph.format = 'png'
        graph.render(f'Sentiment_Images/{vect_name}_DecisionTree')
        
        # add the decision tree to the html
        dtree_html = f"""
            <h4>Decision Tree</h4>
            <img src='Sentiment_Images/{vect_name}_DecisionTree.png' class='img-fluid'>
        """
        
    return dtree_html







# evaluate the model
def evaluate_model(grid, X_train, X_test, y_train, y_test, vect, vect_name):
        
    # get the model_name
    model_name = type(grid.best_estimator_).__name__
    
    # predict on train and test
    train_prediction = grid.predict(X_train)
    test_prediction = grid.predict(X_test)
    
    # calculate accuracy for train and test
    train_accuracy = accuracy_score(y_train, train_prediction)
    test_accuracy = accuracy_score(y_test, test_prediction)
    
    # add the results to the list
    results_list = [ 
                    vect,
                    vect_name,
                    grid,
                    model_name,
                    grid.best_score_, 
                    train_accuracy, 
                    test_accuracy
                    ]
    
    
    
    # create the confusion matrix
    cm = ConfusionMatrixDisplay.from_estimator(grid.best_estimator_, X_test, y_test)
    # plot the confusion matrix
    cm.plot()
    # save the image
    plt.savefig(f'Sentiment_Images/{vect_name}_{model_name}_ConfusionMatrix.png')
    
    
    
    # classification report
    class_report = classification_report(y_test, 
                                         test_prediction, 
                                         target_names=y_test.unique().sort(), 
                                         output_dict=True)
    
    # transform classification report to a dataframe
    class_report_df = pd.DataFrame.from_dict(class_report).transpose()
    
    # transform dataframe to html
    class_report_df_html = class_report_df.to_html(justify='center', 
                                                   index=True, 
                                                   classes=['table-striped', 
                                                            'table-bordered', 
                                                            'table-sm'])
    
    # add the confusion matrix html
    results_html = f"""
        <h3>{vect_name} {model_name}</h3>
        <div>
            <img src='Sentiment_Images/{vect_name}_{model_name}_ConfusionMatrix.png' class='img-fluid'>
        </div>
        {class_report_df_html}
    """
    
    # get feature importance
    results_html += get_feature_importance(grid, vectorizer)
    
    # get decision tree visualization
    results_html += decision_tree_visualization(grid, vectorizer, vect_name)
    
    # return the metrics in a list with the name of the vecotrizer and model
    return (results_list, results_html)



# define the vectorizers to use
vectorizer_list = [
    (raw_text_df, 
     'Base Countvectorizer', 
     CountVectorizer(
            input="content",
            lowercase=True,
            stop_words = "english",
            max_features=1500
            )),
    (balanced_raw_text_df, 
     'Balanced Countvectorizer', 
     CountVectorizer(
            input="content",
            lowercase=True,
            stop_words = "english",
            max_features=1500
            )),
    (raw_text_df, 
     'Base TfidfVectorizer',
     TfidfVectorizer(
            input="content",
            lowercase=True,
            stop_words = "english",
            max_features=1500
            )),
    (balanced_raw_text_df, 
     'Balanced TfidfVectorizer',
     TfidfVectorizer(
            input="content",
            lowercase=True,
            stop_words = "english",
            max_features=1500
            ))
]

# define the list of models
model_list = [
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

# define an empty string for the vectorizer_html
vectorizer_html = ""
# define an empty string for the model_html
model_html = ""
# create an empty dataframe for the results
model_results_df = pd.DataFrame(columns=[
                                        "Vectorizer",
                                        "Vectorizer_Name",
                                        "Grid",
                                        "Model",
                                        "CV Accuracy",
                                        "Train Accuracy", 
                                        "Test Accuracy"])


# create a directory to store the images
if not os.path.exists("Sentiment_Images"): os.mkdir("Sentiment_Images")



# vectorize the data with each vectorizer, and then train each of the models on each of the vectorized datasets
for text_df, vectorizer_name, vectorizer in vectorizer_list:
    
    # vectorize the data
    fit_vectorizer, vectorized_df, vectorized_matrix, vectorizer_html_results = vectorize_data(text_df, vectorizer_name, vectorizer)
    
    # add the html to the main html
    vectorizer_html += vectorizer_html_results
    
    # run LDA clustering
    #lda_analysis(vectorizer, vectorized_matrix, text_df_name)
    
    # train test split
    X_train, X_test, y_train, y_test = train_test_split(vectorized_df.drop("Label", axis=1), 
                                                        vectorized_df.Label, 
                                                        test_size=0.33, 
                                                        random_state=42)
    
    # train each model on the vectorized dataset
    for model, parameters in model_list:
        
        # train the model
        grid = train_model(model, parameters, X_train, y_train)
        
        # evaluate the model and get a list of results
        model_results_list, results_html = evaluate_model(grid, X_train, X_test, y_train, y_test, fit_vectorizer, vectorizer_name)
        
        # add the best model from the grid to the results dataframe along with details
        model_results_df.loc[len(model_results_df.index)] = model_results_list
        
        # append the model html
        model_html += results_html

# sort values by test accuracvy
model_results_df = model_results_df.sort_values(by=["Test Accuracy"], ascending=False)

# create the results dataframe
results_overview_html = model_results_df.drop(['Vectorizer', 'Grid'], axis=1).to_html(
        justify='center',
        index=False, 
        classes=['table-striped', 'table-bordered', 'table-sm'])



        
        
        
        
        
        

# GET THE NEWS ARTICLES LABELED BY TOPIC

# read csv to dataframe
new_text_df=pd.read_csv('NewHeadlines.csv')

# get the text from the dataframe
new_text_df = new_text_df[['Headline', 'LABEL']]

# set the name
new_text_df_name = "New Text"

# remove any rows with NaN in them
new_text_df = new_text_df.dropna()




# VECTORIZE THE NEW TEXT


# get the vectorizer used for the best model
vectorizer = model_results_df[(model_results_df.Vectorizer_Name=='Balanced TfidfVectorizer') & (model_results_df.Model=='SVC')].Vectorizer.values[0]

# get a list of the text
text_list = new_text_df["Headline"].values.tolist()

# get the column names
column_names = vectorizer.get_feature_names_out()
new_column_names = column_names


# vectorize the data into a sparse matrix
matrix = vectorizer.transform(text_list)



# build the data frame
new_vectorized_df = pd.DataFrame(matrix.toarray(),columns=column_names)

# remove any numeric tokens
for col in new_vectorized_df.columns:
    if any(char.isdigit() for char in col) == True:
        new_vectorized_df = new_vectorized_df.drop([col], axis=1)


print(len(new_vectorized_df.columns))





# PREDICT THE SENTIMENT OF EACH ARTICLE

# get the best model
grid_searchcv = model_results_df[(model_results_df.Vectorizer_Name=='Balanced TfidfVectorizer') & (model_results_df.Model=='SVC')].Grid.values[0]

print(len(grid_searchcv.feature_names_in_))

# predit on the best model
predictions = grid_searchcv.predict(new_vectorized_df)

# Create a new DataFrame with the predictions
results = pd.DataFrame(predictions, columns=['prediction'])

# combine the predictions and topic labels
topic_predictions_df = pd.concat([new_text_df.LABEL, results], axis=1)

plt.clf()
sns.countplot(data=topic_predictions_df[topic_predictions_df.prediction != "neutral"], x="LABEL", hue="prediction")
plt.savefig("Sentiment_Images/Label_Sentiment_Countplot.png")

predictions_html = """
                    <div width=250>
                        <img src='Sentiment_Images/Label_Sentiment_Countplot.png' class='img-fluid'>
                    </div>
                """




# CREATE HTML OUTPUT

# to open/create a new html file in the write mode
f = open('My_Sentiment_Report.html', 'w')
  
# the html code which will go in the file GFG.html
html = f"""
<!doctype html>
<html lang="en" data-bs-theme="dark">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Sentiment Report</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.1/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-4bw+/aepP/YC94hEpVNVgiZdgIC5+VKNBQNGCHeKRQN+PtmoHDEXuppvnDJzQIu9" crossorigin="anonymous">
  </head>
  <body>
      <main role="main" class="container">

        <div class="starter-template">
  
            <h1 class="display-1 text-center">Text Sentiment Classification by Industry</h1>
            
            <div class="row mt-5 border">
                <h2>Raw Text Data</h2>
                {raw_text_html}
            </div>
            
            <div class="row mt-5 border">
                <h2>Model Results</h2>
                {results_overview_html}
            </div>
            
            <div class="row mt-5 border">
                <h2>Vectorizers</h2>
                {vectorizer_html}
            </div>
            
            <div class="row mt-5 border">
                <h2>Models</h2>
                {model_html}
            </div>
            
            <div class="row mt-5 border">
                <h2>Sentiment Predictions</h2>
                {predictions_html}
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