import streamlit as st
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
import numpy as np
import warnings
import csv
import pyttsx3
import re

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Load data (replace with your actual data loading logic)
training = pd.read_csv('E:\Projects\Health_Diagnosis_Web\healthcare-chatbot\Data\Training.csv')
testing = pd.read_csv('E:\Projects\Health_Diagnosis_Web\healthcare-chatbot\Data\Testing.csv')

# Your existing code for model training and evaluation
cols = training.columns
cols = cols[:-1]
x = training[cols]
y = training['prognosis']

dimensionality_reduction = training.groupby(training['prognosis']).max()
doc_dataset = pd.read_csv('E:\Projects\Health_Diagnosis_Web\healthcare-chatbot\Data\doctors_dataset.csv', names=['Name', 'Description'])

diseases = dimensionality_reduction.index
diseases = pd.DataFrame(diseases)

doctors = pd.DataFrame()
doctors['name'] = np.nan
doctors['link'] = np.nan
doctors['disease'] = np.nan

doctors['disease'] = diseases['prognosis']

doctors['name'] = doc_dataset['Name']
doctors['link'] = doc_dataset['Description']

record = doctors[doctors['disease'] == 'AIDS']
record['name']
record['link']

reduced_data = training.groupby(training['prognosis']).max()


le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
testx    = testing[cols]
testy    = testing['prognosis']  
testy    = le.transform(testy)
clf1 = DecisionTreeClassifier()
clf = clf1.fit(x_train, y_train)

print(clf.score(x_train,y_train))
print ("cross result========")
scores = cross_val_score(clf, x_test, y_test, cv=3)
# print (scores)
print (scores.mean())


model = SVC()
model.fit(x_train, y_train)
print("for svm: ")
print(model.score(x_test,y_test))

# Streamlit app
st.title("HealthCare ChatBot ")

# Display loaded training data
# st.subheader("Loaded Training Data:")
# st.write(training.head())

# Perform model evaluation and display confusion matrix
# st.subheader("Model Evaluation and Confusion Matrix")


y_pred = clf.predict(x_test)
dfs = []
# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)
# Calculate TP, TN, FP, FN values for each class
num_classes = len(le.classes_)
for i in range(num_classes):
    tn = cm[i, i]
    fn = sum(cm[i, :]) - 25
    fp = sum(cm[:, i]) - 25
    tp = sum(sum(cm)) - tn - fn - fp
    
    print(f"\nClass {le.classes_[i]}:")
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0


    data = {
        'Class': [le.classes_[i]],
        'True Positive (TP)': [tp],
        'True Negative (TN)': [tn],
        'False Positive (FP)': [fp],
        'False Negative (FN)': [fn],
        'Accuracy': [accuracy],
        'Precision': [precision],
        'Recall': [recall],
        'F1 Score': [f1_score]
    }
    
    # Create a DataFrame
    df = pd.DataFrame(data)
    # Display the DataFrame
    print(df)
    print("............................................")
    dfs.append(df)

# Concatenate all DataFrames in the list
result_df = pd.concat(dfs, ignore_index=True)

# Write the DataFrame to an Excel file
result_df.to_excel('confusion_matrix_results.xlsx', index=False)





importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
features = cols

def readn(nstr):
    engine = pyttsx3.init()

    engine.setProperty('voice', "english+f5")
    engine.setProperty('rate', 130)

    engine.say(nstr)
    engine.runAndWait()
    engine.stop()


severityDictionary = dict()
description_list = dict()
precautionDictionary = dict()

symptoms_dict = {}

for index, symptom in enumerate(x):
    symptoms_dict[symptom] = index


# def calc_condition(exp, days):
#     sum = 0
#     for item in exp:
#         sum = sum + severityDictionary[item]
#     if (sum * days) / (len(exp) + 1) > 13:
#         st.warning("You should take the consultation from a doctor.")
#     else:
#         st.success("It might not be that bad, but you should take precautions.")
        
def calc_condition(exp, days):
    sum = 0
    for item in exp:
        severity = severityDictionary.get(item, None)
        if severity is not None:
            sum += severity
        else:
            print(f"Warning: Severity not found for symptom '{item}'.")
    if ((sum * days) / (len(exp) + 1) > 13):
        print("You should take consultation from a doctor.")
        print()
    else:
        print("It might not be that bad, but you should take precautions.")
        print()



def getDescription():
    global description_list
    with open('E:\Projects\Health_Diagnosis_Web\healthcare-chatbot\MasterData\symptom_Description.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            _description = {row[0]: row[1]}
            description_list.update(_description)


def getSeverityDict():
    global severityDictionary
    with open('E:\Projects\Health_Diagnosis_Web\healthcare-chatbot\MasterData\Symptom_severity.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        try:
            for row in csv_reader:
                _diction = {row[0]: int(row[1])}
                severityDictionary.update(_diction)
        except:
            pass


def getprecautionDict():
    global precautionDictionary
    with open('E:\Projects\Health_Diagnosis_Web\healthcare-chatbot\MasterData\symptom_precaution.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            _prec = {row[0]: [row[1], row[2], row[3], row[4]]}
            precautionDictionary.update(_prec)


def getInfo():
    st.subheader("HealthCare ChatBot")
    st.write("-----------------------------------")
    st.write("\nYour Name?")
    name = st.text_input("")
    st.write("Hello, ", name)


def check_pattern(dis_list, inp):
    pred_list = []
    inp = inp.replace(' ', '_')
    patt = f"{inp}"
    regexp = re.compile(patt)
    pred_list = [item for item in dis_list if regexp.search(item)]
    if len(pred_list) > 0:
        return 1, pred_list
    else:
        return 0, []


def sec_predict(symptoms_exp):
    df = pd.read_csv('E:\Projects\Health_Diagnosis_Web\healthcare-chatbot\Data\Training.csv')
    X = df.iloc[:, :-1]
    y = df['prognosis']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20)
    rf_clf = DecisionTreeClassifier()
    rf_clf.fit(X_train, y_train)

    symptoms_dict = {symptom: index for index, symptom in enumerate(X)}
    input_vector = np.zeros(len(symptoms_dict))
    for item in symptoms_exp:
        input_vector[[symptoms_dict[item]]] = 1

    return rf_clf.predict([input_vector])


def print_disease(node):
    node = node[0]
    val = node.nonzero()
    disease = le.inverse_transform(val[0])
    return list(map(lambda x: x.strip(), list(disease)))


def tree_to_code(tree, feature_names, reduced_data):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    chk_dis = ",".join(feature_names).split(",")
    symptoms_present = []

    # Take symptom input from the user
    disease_input = st.text_input("Enter the symptom you are experiencing")

    conf, cnf_dis = check_pattern(chk_dis, disease_input)
    if conf == 1:
        st.write("Searches related to input:")
        for num, it in enumerate(cnf_dis):
            st.write(num, ")", it)
        if num != 0:
            conf_inp = st.number_input(f"Select the one you meant (0 - {num})", 0, num)
            disease_input = cnf_dis[conf_inp]
    else:
        st.warning("Enter valid symptom.")

    # Take the number of days from the user
    num_days = st.number_input("From how many days?", 1)

    def recurse(node, depth):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]

            if name == disease_input:
                val = 1
            else:
                val = 0
            if val <= threshold:
                recurse(tree_.children_left[node], depth + 1)
            else:
                symptoms_present.append(name)
                recurse(tree_.children_right[node], depth + 1)
        else:
            present_disease = print_disease(tree_.value[node])
            red_cols = reduced_data.columns
            symptoms_given = red_cols[reduced_data.loc[present_disease].values[0].nonzero()]
            dis_list = list(symptoms_present)
            if len(dis_list) != 0:
                st.write("Symptoms present:", str(list(symptoms_present)))

            st.write("Are you experiencing any")
            symptoms_exp = []
            for syms in list(symptoms_given):
                inp = st.selectbox(f"{syms}?", ["Yes", "No"])
                if inp == "Yes":
                    symptoms_exp.append(syms)

            second_prediction = sec_predict(symptoms_exp)
            calc_condition(symptoms_exp, num_days)

            if present_disease[0] == second_prediction[0]:
                st.write(f"You may have {present_disease[0]}")
                st.write(description_list[present_disease[0]])
            else:
                st.write(f"You may have {present_disease[0]} or {second_prediction[0]}")
                st.write(description_list[present_disease[0]])
                st.write(description_list[second_prediction[0]])

            precution_list = precautionDictionary[present_disease[0]]
            st.write("Take the following measures:")
            for i, j in enumerate(precution_list):
                st.write(f"{i+1})", j)

            confidence_level = (1.0 * len(symptoms_present)) / len(symptoms_given)
            # st.write(f"Confidence level is {confidence_level}")

            st.write('I would like to suggest you one of our respected doctors:')
            row = doctors[doctors['disease'] == present_disease[0]]
            st.write(f"Consult {str(row['name'].values)}")
            st.write(f"Visit {str(row['link'].values)}")

    recurse(0, 1)
    






    
    
    # def recurse(node, depth):
    #     if tree_.feature[node] != _tree.TREE_UNDEFINED:
    #         name = feature_name[node]
    #         threshold = tree_.threshold[node]

    #         if name == disease_input:
    #             val = 1
    #         else:
    #             val = 0
    #         if val <= threshold:
    #             recurse(tree_.children_left[node], depth + 1)
    #         else:
    #             symptoms_present.append(name)
    #             recurse(tree_.children_right[node], depth + 1)
    #     else:
    #         present_disease = print_disease(tree_.value[node])
    #         red_cols = reduced_data.columns
    #         symptoms_given = red_cols[reduced_data.loc[present_disease].values[0].nonzero()]
    #         dis_list = list(symptoms_present)
    #         if len(dis_list) != 0:
    #             st.write("Symptoms present:", str(list(symptoms_present)))

    #         st.write("Are you experiencing any")
    #         symptoms_exp = []
    #         for syms in list(symptoms_given):
    #             inp = st.selectbox(f"{syms}?", ["Select", "Yes", "No"])
    #             if inp == "Yes":
    #                 symptoms_exp.append(syms)
    #             elif inp =="Select":
    #                 break
    #             else:
    #                 continue

    #         second_prediction = sec_predict(symptoms_exp)
    #         calc_condition(symptoms_exp, num_days)

    #         if present_disease[0] == second_prediction[0]:
    #             st.write(f"You may have {present_disease[0]}")
    #             st.write(description_list[present_disease[0]])
    #         else:
    #             st.write(f"You may have {present_disease[0]} or {second_prediction[0]}")
    #             st.write(description_list[present_disease[0]])
    #             st.write(description_list[second_prediction[0]])

    #         precution_list = precautionDictionary[present_disease[0]]
    #         st.write("Take the following measures:")
    #         for i, j in enumerate(precution_list):
    #             st.write(f"{i+1})", j)

    #         confidence_level = (1.0 * len(symptoms_present)) / len(symptoms_given)
    #         # st.write(f"Confidence level is {confidence_level}")

    #         st.write('I would like to suggest you one of our respected doctors:')
    #         row = doctors[doctors['disease'] == present_disease[0]]
    #         st.write(f"Consult {str(row['name'].values)}")
    #         st.write(f"Visit {str(row['link'].values)}")

    # recurse(0, 1)
        
    

    
getSeverityDict()
getDescription()
getprecautionDict()
getInfo()
tree_to_code(clf, cols,reduced_data)
