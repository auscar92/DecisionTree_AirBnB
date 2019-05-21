y = df3
x = df
x = x.values
y = y.values

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = .3)

scaler = StandardScaler()
scaler.fit(xtrain)

xtrain = scaler.transform(xtrain)
xtest = scaler.transform(xtest)

classifier = DecisionTreeRegressor(random_state=0)
nnclf = MLPRegressor(hidden_layer_sizes = (128, 128, 128, 128, 128),
                    max_iter = 500,
                    activation = 'relu')
nnclf.fit(xtrain, ytrain)
nnpreds = nnclf.predict(xtest)
mean_absolute_error(ytest, nnpreds)

classifier.fit(xtrain, ytrain)

airpreds = classifier.predict(xtest)

mean_absolute_error(ytest, airpreds)

kfold = model_selection.KFold(n_splits=10, random_state=5)
bagging = BaggingClassifier(knnc, max_samples = .8, max_features = 5)
results = model_selection.cross_val_score(bagging, spamx, spamy, cv=kfold)

print(results.mean())

model = RandomForestClassifier(n_estimators = 4)
model.fit(xtrain, ytrain)
rfpred = model.predict(xtest)
print(accuracy_score(rfpred, ytest))
model1 = ExtraTreesClassifier(n_estimators=num_trees, max_features=max_features)


dot_data = tree.export_graphviz(classifier, out_file='graph.dot')
graph1 = graphviz.Source(dot_data)
check_call(['dot','-Tpng','graph.dot','-o','OutputFile.png'])

pylab.savefig( "graph1234.png")
plt.pyplot.savefig( "graph1234.png")
Graph(filename = "graph1234.png", name='graph1234', format = 'png')

estimator = model.estimators_[1]

from sklearn.tree import export_graphviz
# Export as dot file
export_graphviz(estimator, out_file='tree.dot',
               rounded = True, proportion = False,
               precision = 2, filled = True)

# Convert to png using system command (requires Graphviz)
from subprocess import call
call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])

# Display in jupyter notebook
from IPython.display import Image
Image(filename = 'tree.png')
