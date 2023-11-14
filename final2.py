import pandas as pd
import numpy as np
import timeit
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from sklearn.ensemble import GradientBoostingClassifier


def barChart():
    df = pd.read_csv("NBA_Player_Stats_2.csv")
    df = df[df.MP >= 10]
    position_df =df[['Position', 'PTS', 'TRB', 'AST', 'STL', 'BLK']]
    # Melt the dataframe so that each row represents a single data point and the position is the x-axis
    melted_df = pd.melt(position_df, id_vars=['Position'], value_vars=['PTS', 'TRB', 'AST', 'STL', 'BLK'])
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Position', y='value', hue='variable', data=melted_df)
    plt.title('Bar Chart of Main Stats across all 5 Positions')
    plt.savefig("bar_chart.png")

def pairPlot():
    df = pd.read_csv("NBA_Player_Stats_2.csv")
    df = df[df.MP >= 10]
    sns_df = df[['PTS', 'TRB', 'AST', 'STL', 'BLK', 'Position']].head(300)
    sns_df = sns_df.reset_index()
    sns_df = sns_df.drop('index', axis=1)
    sns_plot = sns.pairplot(sns_df, hue='Position', size=2)
    sns_plot.savefig("pairplot.png")
    
def decisionTree():
    df = pd.read_csv("NBA_Player_Stats_2.csv")
    df = df[df.MP >= 10]
    df.fillna(0, inplace=True)
    # select the relevant columns for feature selection
    selected_features = ['PTS', 'TRB', 'AST','BLK','TOV','FT%','FG%','3P%','eFG%']

    X = df[selected_features]
    y = df['Position']

    y = pd.get_dummies(y)
    
    #Split X and Y into training and testing set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)

    scaler = StandardScaler()
    X_scaler = scaler.fit(X_train)
    X_train_scaled = X_scaler.transform(X_train)
    X_test_scaled = X_scaler.transform(X_test)
    
    decisionTreeImportance=DecisionTreeClassifier(random_state=1)
    decisionTreeImportance=decisionTreeImportance.fit(X_train_scaled, y_train)
    predictionImportance=decisionTreeImportance.predict(X_test_scaled)
    accI = accuracy_score(y_test, predictionImportance)
    print("Results with importance factors")
    print("Prediction:", predictionImportance)
    print("Accuracy:", accI)
    
    df = pd.read_csv("NBA_Player_Stats_2.csv")
    df = df[df.MP >= 10]
    df.fillna(0, inplace=True)
    # select the relevant columns for feature selection
    X = df[['PTS', 'TRB', 'AST', 'STL', 'BLK','TOV','FT%','FG%','FGA','3P','3PA','3P%','2P','2PA','2P%','eFG%','FT','FTA','ORB','DRB']]
    y = df['Position']

    y = pd.get_dummies(y)
    
    #Split X and Y into training and testing set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)

    scaler = StandardScaler()
    X_scaler = scaler.fit(X_train)
    X_train_scaled = X_scaler.transform(X_train)
    X_test_scaled = X_scaler.transform(X_test)
    
    decisionTreeAll=DecisionTreeClassifier(random_state=1)
    decisionTreeAll=decisionTreeAll.fit(X_train_scaled, y_train)
    predictionAll=decisionTreeAll.predict(X_test_scaled)
    accA= accuracy_score(y_test, predictionAll)
    print("Results after using all factors")
    print("Prediction:", predictionAll)
    print("Accuracy:", accA)
    
    
    
    
def randomForest():
    df = pd.read_csv("NBA_Player_Stats_2.csv")
    df = df[df.MP >= 10]
    df.fillna(0, inplace=True)
    # select the relevant columns for feature selection
    selected_features = ['PTS', 'TRB', 'AST','BLK','TOV','FT%','FG%','3P%','eFG%']

    X = df[selected_features]
    y = df['Position']

    y = pd.get_dummies(y)
    
    #Split X and Y into training and testing set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)

    scaler = StandardScaler()
    X_scaler = scaler.fit(X_train)
    X_train_scaled = X_scaler.transform(X_train)
    X_test_scaled = X_scaler.transform(X_test)
    
    randomForestImportance=RandomForestClassifier(n_estimators=650, random_state=50)
    randomForestImportance=randomForestImportance.fit(X_train_scaled, y_train)
    predictionImportance=randomForestImportance.predict(X_test_scaled)
    accI = accuracy_score(y_test, predictionImportance)
    print("Results with importance factors")
    print("Prediction:", predictionImportance)
    print("Accuracy:", accI)
    
    #Test with all of the categories
    df = pd.read_csv("NBA_Player_Stats_2.csv")
    df = df[df.MP >= 10]
    df.fillna(0, inplace=True)
    # select the relevant columns for feature selection
    X = df[['PTS', 'TRB', 'AST', 'STL', 'BLK','TOV','FT%','FG%','FGA','3P','3PA','3P%','2P','2PA','2P%','eFG%','FT','FTA','ORB','DRB']]
    y = df['Position']

    y = pd.get_dummies(y)
    
    #Split X and Y into training and testing set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)

    scaler = StandardScaler()
    X_scaler = scaler.fit(X_train)
    X_train_scaled = X_scaler.transform(X_train)
    X_test_scaled = X_scaler.transform(X_test)
    
    randomForestAll=RandomForestClassifier(n_estimators=650, random_state=50)
    randomForestAll=randomForestAll.fit(X_train_scaled, y_train)
    predictionAll=randomForestAll.predict(X_test_scaled)
    accA= accuracy_score(y_test, predictionAll)
    print("Results after using all factors")
    print("Prediction:", predictionAll)
    print("Accuracy:", accA)
    

def kNNforPos():
    # load your dataset
    df = pd.read_csv("NBA_Player_Stats_2.csv")

    selected_features = ['PTS', 'TRB', 'AST','BLK','TOV','FT%','FG%','3P%','eFG%']
    df = df.dropna()
    X = df[selected_features]
    y = df['Position']
   
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)


    # one hot enconding
    y = pd.get_dummies(y)

    # split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)

    # create the KNeighborsClassifier model
    knn = KNeighborsClassifier()

    # fit the model to the training data
    knn.fit(X_train, y_train)

    # evaluate the model on the test data
    accuracy = knn.score(X_test, y_test)
    print("Accuracy:", accuracy)
    
def gradientBoostTree():
    df = pd.read_csv("NBA_Player_Stats_2.csv")
    df = df[df.MP >= 10]
    df.fillna(0, inplace=True)

    selected_features = ['PTS', 'TRB', 'AST','BLK','TOV','FT%','FG%','3P%','eFG%']
    X = df[selected_features]
    y = df['Position']
   
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)


    # one hot enconding
    position_dictionary = {
    "PG": 1,
    "SG": 2,
    "SF": 3,
    "PF": 4,
    "C": 5}
    y = y.map(position_dictionary)

    # split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)
    scaler = StandardScaler()
    X_scaler = scaler.fit(X_train)
    X_train_scaled = X_scaler.transform(X_train)
    X_test_scaled = X_scaler.transform(X_test)
    
    gradientBoostTreeImportance=GradientBoostingClassifier(n_estimators=20, learning_rate=0.75, max_depth=3, random_state=50)
    gradientBoostTreeImportance=gradientBoostTreeImportance.fit(X_train_scaled,y_train)
    predictionImportance=gradientBoostTreeImportance.predict(X_test_scaled)
    
    accI = accuracy_score(y_test, predictionImportance)
    print("Results with importance factors")
    print("Prediction:", predictionImportance)
    print("Accuracy:", accI)
    
    #Test with all of the categories
    df = pd.read_csv("NBA_Player_Stats_2.csv")
    df = df[df.MP >= 10]
    df.fillna(0, inplace=True)
    X = df[['PTS', 'TRB', 'AST', 'STL', 'BLK','TOV','FT%','FG%','FGA','3P','3PA','3P%','2P','2PA','2P%','eFG%','FT','FTA','ORB','DRB']]
    y = df['Position']
   
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)


    # one hot enconding
    position_dictionary = {
    "PG": 1,
    "SG": 2,
    "SF": 3,
    "PF": 4,
    "C": 5}
    y = y.map(position_dictionary)
    
    # split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)
    scaler = StandardScaler()
    X_scaler = scaler.fit(X_train)
    X_train_scaled = X_scaler.transform(X_train)
    X_test_scaled = X_scaler.transform(X_test)
    
    
    
    gradientBoostTreeAll=GradientBoostingClassifier(n_estimators=20, learning_rate=0.75, max_depth=3, random_state=50)
    gradientBoostTreeAll=gradientBoostTreeAll.fit(X_train_scaled,y_train)
    predictionAll=gradientBoostTreeAll.predict(X_test_scaled)
    
    accA = accuracy_score(y_test, predictionAll)
    print("Results after using all factors")
    print("Prediction:", predictionAll)
    print("Accuracy:", accA)


    
    
    

    
    
def main():
    #print("GradientBoostTree results:")
    #gradientBoostTree()
    #print("RandomForest results:")
    #randomForest()
    #print("DecisionTree results:")
    #decisionTree()
    print("kNNforPos results:")
    kNNforPos()
    
    
       
main()