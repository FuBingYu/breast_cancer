import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score

def DataPreprocessing(data):
    data.drop("id", axis=1, inplace=True)
    replace1 = []
    for i in data['diagnosis']:
        replace1.append(1.0 if i == 'M' else 0.0)
    data['diagnosis'] = replace1
    features_mean = list(data.columns[1:11])
    features_se = list(data.columns[11:21])
    features_worst = list(data.columns[21:31])
    X = data[features_mean]
    y = data['diagnosis']

    fig, axes = plt.subplots(1, 10)
    color = dict(boxes='DarkGreen', whiskers='DarkOrange',
                 medians='DarkBlue', caps='Red')
    X.plot(kind='box', ax=axes, use_index=False, subplots=True,
           color=color, sym='r+')  # sym参数表示异常值标记的方式
    axes[0].set_xlabel('radius')
    axes[1].set_xlabel('texture')
    axes[2].set_xlabel('perimeter')
    axes[3].set_xlabel('area')
    axes[4].set_xlabel('smoothness')
    axes[5].set_xlabel('compactness')
    axes[6].set_xlabel('concavity')
    axes[7].set_xlabel('concave_points')
    axes[8].set_xlabel('symmetry')
    axes[9].set_xlabel('fractal_dimension')
    fig.subplots_adjust(wspace=10, hspace=10)  # 调整子图之间的间距
    fig.savefig(r"C:\Users\傅冰玉\Desktop\医学\傅冰玉--医学数据集\p1.png")
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # 采用Z-Score标准化数据
    ss = StandardScaler()
    X_train = ss.fit_transform(x_train)
    X_test = ss.transform(x_test)
    return features_mean,X_train,X_test,y_train,y_test

def ChooseParams(features_mean,X_train,X_test):
    sns.countplot(data['diagnosis'], label="Count")
    plt.savefig(r"C:\Users\傅冰玉\Desktop\医学\傅冰玉--医学数据集\p2.png")
    #plt.show()
    corr = data[features_mean].corr()
    plt.figure(figsize=(15, 15))
    sns.heatmap(corr, annot=True)
    plt.savefig(r"C:\Users\傅冰玉\Desktop\医学\傅冰玉--医学数据集\p3.png")
    #plt.show()
    features_remain = ['area_mean', 'texture_mean', 'smoothness_mean',
                       'concavity_mean', 'symmetry_mean', 'fractal_dimension_mean']
    features_numbers = [1, 3, 4, 6, 8, 9]
    X_train = X_train[:, features_numbers]
    X_test = X_test[:, features_numbers]
    return X_train,X_test

def ChooseModel(X_train,y_train,X_test,y_test):
    # 3.1 logistic回归
    lr_model = LogisticRegression()
    lr_model.fit(X_train, y_train)
    print('Logistic模型在测试集上的准确率为: ', lr_model.score(X_test, y_test))

    # 3.2 支持向量机（SVM）
    svm_model = svm.SVC()
    svm_model.fit(X_train, y_train)
    prediction = svm_model.predict(X_test)
    print('SVM模型在测试集上的准确率为: ', metrics.accuracy_score(prediction, y_test))

    # 3.3 k近邻
    knn = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto')
    knn.fit(X_train, y_train)
    print("KNN模型在测试集上的准确率为:", knn.score(X_test, y_test))

    # 3.4 决策树
    # 自动调参，确定最优参数。
    '''max_depth = range(4, 10, 1)
    min_samples_split = range(2, 12, 1)
    min_samples_leaf = range(2, 12, 1)
    parameters_dtc = {'max_depth': max_depth,
                      'min_samples_split': min_samples_split,
                      'min_samples_leaf': min_samples_leaf}
    DTC = DecisionTreeClassifier()
    grid_search = GridSearchCV(DTC, parameters_dtc, cv=10, n_jobs=1)
    grid_search.fit(X_train, y_train)
    print(grid_search.best_params_)'''
    # {'max_depth': 8, 'min_samples_leaf': 2, 'min_samples_split': 4}
    dtc = DecisionTreeClassifier(max_depth=8, min_samples_split=4, min_samples_leaf=2)
    dtc.fit(X_train, y_train)
    y_predict = dtc.predict(X_test)
    print("决策树模型在测试集上的准确率为：", metrics.accuracy_score(y_test, y_predict))

    # 3.5 随机森林
    # 自动调参，通过交叉验证确定最优参数。
    '''RF = RandomForestClassifier()
    # 可以通过定义树的各种参数，限制树的大小，防止出现过拟合现象
    parameters = {'n_estimators': [50, 100, 200],
                  'criterion': ['entropy', 'gini'],
                  'max_depth': [4, 5, 6],
                  'min_samples_split': [2, 4, 6, 8],
                  'min_samples_leaf': [2, 4, 6, 8, 10]
                  }
    grid_obj = GridSearchCV(RF, parameters, cv=10, n_jobs=1)
    grid_obj = grid_obj.fit(X_train, y_train)'''
    rf = RandomForestClassifier(criterion='entropy', max_depth=5, min_samples_leaf=2,
                                min_samples_split=2,n_estimators=50)
    rf.fit(X_train, y_train)
    predictions = rf.predict(X_test)
    print("随机森林模型在测试集上的准确率为：", accuracy_score(y_test, predictions))


if __name__ == '__main__':
    data = pd.read_csv(r"C:\Users\傅冰玉\Desktop\医学\傅冰玉--医学数据集\params.csv")
    names = data.keys()
    number = data.shape[0]  # 569
    #print(names)
    null_num = data.isnull().sum()
    #print(null_num)
    features_mean,X_train,X_test,y_train,y_test = DataPreprocessing(data)
    X_train, X_test = ChooseParams(features_mean,X_train,X_test)
    score = ChooseModel(X_train, y_train, X_test, y_test)

