"""EXP 35: decision-tree classifier with a small reproducible dataset."""
from sklearn.tree import DecisionTreeClassifier
X=[[1,1],[1,2],[2,1],[6,6],[6,7],[7,6]]; y=['A','A','A','B','B','B']
m=DecisionTreeClassifier(random_state=0).fit(X,y)
print('Prediction for [6, 5]:',m.predict([[6,5]])[0]); print('Training accuracy:',round(m.score(X,y),2))
