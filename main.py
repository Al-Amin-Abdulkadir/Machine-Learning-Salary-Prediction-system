import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import joblib

df = pd.read_csv("data/salary_data.csv")

print(df.head())
print(df.info())
print(df.describe())

df = df.drop_duplicates()
print(df.isnull().sum())

plt.scatter(df["experience"], df["salary"])
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.savefig("plot.png")
print("Plot displayed successfully")

X = df[["experience", "age"]]
y = df["salary"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


lr = LinearRegression()
lr.fit(X_train, y_train)

dt = DecisionTreeRegressor()
dt.fit(X_train, y_train)

rf = RandomForestRegressor(n_estimators=50)
rf.fit(X_train, y_train)

def evaluate(model):
    preds = model.predict(X_test)
    print("MAE:", mean_absolute_error(y_test, preds))
    print("R2:", r2_score(y_test, preds))
    print("-----")

print("Linear Regression")
evaluate(lr)

print("Decision Tree")
evaluate(dt)

print("Random Forest")
evaluate(rf)

print("Evaluation complete")

joblib.dump(rf, "model.pkl")
loaded_model = joblib.load("model.pkl")

prediction = loaded_model.predict([[5, 30]])

print(f"Predicted salary : {prediction}")