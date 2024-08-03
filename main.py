from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

df = pd.read_csv("KaggleV2-May-2016.csv")
df = df.dropna()

features = ['Gender', 'Age', 'Scholarship', 'Hipertension', 'Diabetes', 'Alcoholism', 'Handcap', 'SMS_received']
extracted = df[features].copy()

scaler = StandardScaler()
encoder = LabelEncoder()

numericalFeatures = ['Age', 'Scholarship', 'Hipertension', 'Diabetes', 'Alcoholism', 'Handcap', 'SMS_received']
extracted["Gender"] = encoder.fit_transform(extracted["Gender"])
extracted[numericalFeatures] = scaler.fit_transform(extracted[numericalFeatures])

y = encoder.fit_transform(df["No-show"])
xTrain, xTemp, yTrain, yTemp = train_test_split(extracted, y, test_size=0.2, random_state=0)
xVal, xTest, yVal, yTest = train_test_split(xTemp, yTemp, test_size=0.5, random_state=0)

criteria = ["entropy", "gini"]
bestCriterion = None
bestAccuracy = 0

for criterion in criteria:
    classify = DecisionTreeClassifier(random_state=0, criterion=criterion)
    classify.fit(xTrain,yTrain)
    yPredict = classify.predict(xVal)
    accuracy = accuracy_score(yVal,yPredict)
    print(f"Accuracy with {criterion} criterion: {accuracy}")

    if accuracy > bestAccuracy:
        bestAccuracy = accuracy
        bestCriterion = criterion

print(f"Best criterion: {bestCriterion} with accuracy: {bestAccuracy}")

finalClassifier = DecisionTreeClassifier(random_state=0, criterion=bestCriterion)
finalClassifier.fit(xTrain, yTrain)

yPred = finalClassifier.predict(xTest)
testAcc = accuracy_score(yTest, yPred)
confusionMatrix = confusion_matrix(yTest,yPred)

print(f"Test Accuracy: {testAcc}")
print(confusionMatrix)

classifier = RandomForestClassifier(n_estimators=200, random_state=0, criterion=bestCriterion)
classifier.fit(xTrain, yTrain)
yValPredict = classifier.predict(xVal)
acc = accuracy_score(yVal, yValPredict)
precision = precision_score(yVal, yValPredict, average='weighted')

print(f"The accuracy of the test is {acc} and the precision is {precision}")