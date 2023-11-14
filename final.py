import pandas as pd
import numpy as np
import timeit
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




def correlationCo():
    # load your dataset
    df = pd.read_csv("NBA_Player_Stats_2.csv")
    df=pd.get_dummies(df, columns=['Position'])
    df.fillna(0, inplace=True)

    # select the relevant columns for feature selection
    X = df[['PTS', 'TRB', 'AST', 'STL', 'BLK','TOV','FT%','FG%','FGA','3P','3PA','3P%','2P','2PA','eFG%']]

    # loop through all one-hot encoded columns
    for column in df.columns:
        if 'Position' in column:
            y = df[column]
            corr_coefs = {}
            for col in X.columns:
                corr_coefs[col] = pearsonr(X[col], y)[0]
            print(f"Correlation coefficient for {column}")
            print(corr_coefs)

def featureImportance():
    # load your dataset
    df = pd.read_csv("NBA_Player_Stats_2.csv")
    df.fillna(0, inplace=True)
    df = pd.get_dummies(df, columns=['Position'])
    # select the relevant columns for feature selection
    X = df[['PTS', 'TRB', 'AST','BLK','TOV','FT%','FG%','3PA','3P%','eFG%']]

    # train a decision tree classifier for each position
    for position in ['Position_PG', 'Position_SG', 'Position_SF', 'Position_PF', 'Position_C']:
        y = df[position]
        dt = DecisionTreeClassifier()
        dt.fit(X, y)
        # extract the feature importance scores
        feature_importance = dt.feature_importances_
        print(f'Feature importance for {position}: {feature_importance}')
        
def randomForest():
    # load your dataset
    df = pd.read_csv("NBA_Player_Stats_2.csv")
    df = df[df.MP >= 10]
    # select the relevant columns for feature selection
    selected_features = ['PTS', 'TRB', 'AST','BLK','TOV','FT%','FG%','3P%','eFG%']
    X = df[selected_features]
    y = df['Position']
    
    # encode the positions using LabelEncoder
    le = LabelEncoder()
    y = le.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)
    
    scaler = StandardScaler()
    X_scaler = scaler.fit(X_train)
    X_train_scaled = X_scaler.transform(X_train)
    X_test_scaled = X_scaler.transform(X_test)
    
    rf = RandomForestClassifier()
    rf.fit(X_train_scaled, y_train_)
    #predict the test set
    y_pred = rf.predict(X_test)
    #evaluate the model
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print("Accuracy: ", acc)
    print("Precision: ", prec)
    print("Recall: ", recall_score(y_test, y_pred, average='weighted'))
    
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

def neuralNetwork():
    df = pd.read_csv("NBA_Player_Stats_2.csv")
    selected_features = ['PTS', 'TRB', 'AST','BLK','TOV','FT%','FG%','3P%','eFG%']
    df = df.dropna()
    X = df[selected_features]
    y = df['Position']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)

    # one hot enconding
    y_train = pd.get_dummies(y_train)
    y_test = pd.get_dummies(y_test)
    

    model = Sequential()
    model.add(Dense(10, input_shape=(9,), activation='relu'))
    model.add(Dense(6, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=50, batch_size=32)
    
    # Measure training time
    startTime = timeit.default_timer()
    hist = model.fit(X_train, y_train, epochs=50)
    elapsedTimeTraining = timeit.default_timer() - startTime
    
    # Measure testing time
    startTime = timeit.default_timer()
    evalResults = model.evaluate(X_test, y_test, verbose=1)
    elapsedTimeTesting = timeit.default_timer() - startTime
    
    print('TRAINING Time Elapsed: ' + str(elapsedTimeTraining) + ' seconds')
    print('TESTING Time Elapsed: ' + str(elapsedTimeTesting) + ' seconds')
    print(evalResults)
    model.summary()
    print("Number of parameters: ",model.count_params())

    # evaluate the model
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print('Test accuracy:', test_acc)
    
    
def main():
    correlationCo()
    featureImportance()
    randomForest()
    
main()