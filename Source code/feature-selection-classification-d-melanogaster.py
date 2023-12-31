import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from niapy.problems import Problem
from niapy.task import Task
from niapy.algorithms.basic import ParticleSwarmOptimization
from sklearn import metrics

dataset = pd.read_csv("/kaggle/input/sequencedna/D.melanogaster(PreProcessed).csv")
X = dataset.drop(['label'],axis=1)
y = dataset['label']

X1 = X.to_numpy()
y1 = y.to_numpy()

feature_names = dataset.columns
feature_names = feature_names.to_numpy()
feature_names = feature_names[0:10]
feature_names

X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.2, stratify=y, random_state=1234)

from sklearn.ensemble import RandomForestClassifier
class Rf_FeatureSelection(Problem):
    def __init__(self, X_train, y_train, alpha=0.99):
        super().__init__(dimension=X_train.shape[1], lower=0, upper=1)
        self.X_train = X_train
        self.y_train = y_train
        self.alpha = alpha

    def _evaluate(self, x):
        selected = x > 0.5
        num_selected = selected.sum()
        if num_selected == 0:
            return 1.0
        accuracy = cross_val_score(RandomForestClassifier(), self.X_train[:, selected], self.y_train, cv=2, n_jobs=-1).mean()
        score = 1 - accuracy
        num_features = self.X_train.shape[1]
        return self.alpha * score + (1 - self.alpha) * (num_selected / num_features)

problem = Rf_FeatureSelection(X_train1, y_train1)
task = Task(problem, max_iters=20)
algorithm = ParticleSwarmOptimization(population_size=20, seed=1234)
best_features_rf, best_fitness_rf = algorithm.run(task)
selected_features_rf = best_features_rf > 0.5
print('Number of selected features for Random Forest:', selected_features_rf.sum())
print('Selected features:', ', '.join(feature_names[selected_features_rf].tolist()))

from sklearn.ensemble import GradientBoostingClassifier
class GB_FeatureSelection(Problem):
    def __init__(self, X_train, y_train, alpha=0.99):
        super().__init__(dimension=X_train.shape[1], lower=0, upper=1)
        self.X_train = X_train
        self.y_train = y_train
        self.alpha = alpha

    def _evaluate(self, x):
        selected = x > 0.5
        num_selected = selected.sum()
        if num_selected == 0:
            return 1.0
        accuracy = cross_val_score(GradientBoostingClassifier(), self.X_train[:, selected], self.y_train, cv=2, n_jobs=-1).mean()
        score = 1 - accuracy
        num_features = self.X_train.shape[1]
        return self.alpha * score + (1 - self.alpha) * (num_selected / num_features)

problem = GB_FeatureSelection(X_train1, y_train1)
task = Task(problem, max_iters=20)
algorithm = ParticleSwarmOptimization(population_size=20, seed=1234)
best_features_gb, best_fitness_gb = algorithm.run(task)
selected_features_gb = best_features_gb > 0.5
print('Number of selected features for Gradient Boosting:', selected_features_gb.sum())
print('Selected features:', ', '.join(feature_names[selected_features_gb].tolist()))

from sklearn.svm import SVC
class SVM_FeatureSelection(Problem):
    def __init__(self, X_train, y_train, alpha=0.99):
        super().__init__(dimension=X_train.shape[1], lower=0, upper=1)
        self.X_train = X_train
        self.y_train = y_train
        self.alpha = alpha

    def _evaluate(self, x):
        selected = x > 0.5
        num_selected = selected.sum()
        if num_selected == 0:
            return 1.0
        accuracy = cross_val_score(SVC(), self.X_train[:, selected], self.y_train, cv=2, n_jobs=-1).mean()
        score = 1 - accuracy
        num_features = self.X_train.shape[1]
        return self.alpha * score + (1 - self.alpha) * (num_selected / num_features)

problem = SVM_FeatureSelection(X_train1, y_train1)
task = Task(problem, max_iters=20)
algorithm = ParticleSwarmOptimization(population_size=20, seed=1234)
best_features_svm, best_fitness_svm = algorithm.run(task)
selected_features_svm = best_features_svm > 0.5
print('Number of selected features for Support Vector Machine:', selected_features_svm.sum())
print('Selected features:', ', '.join(feature_names[selected_features_svm].tolist()))

from sklearn.neighbors import KNeighborsClassifier
class KNN_FeatureSelection(Problem):
    def __init__(self, X_train, y_train, alpha=0.99):
        super().__init__(dimension=X_train.shape[1], lower=0, upper=1)
        self.X_train = X_train
        self.y_train = y_train
        self.alpha = alpha

    def _evaluate(self, x):
        selected = x > 0.5
        num_selected = selected.sum()
        if num_selected == 0:
            return 1.0
        accuracy = cross_val_score(KNeighborsClassifier(n_neighbors=7), self.X_train[:, selected], self.y_train, cv=2, n_jobs=-1).mean()
        score = 1 - accuracy
        num_features = self.X_train.shape[1]
        return self.alpha * score + (1 - self.alpha) * (num_selected / num_features)

problem = KNN_FeatureSelection(X_train1, y_train1)
task = Task(problem, max_iters=20)
algorithm = ParticleSwarmOptimization(population_size=20, seed=1234)
best_features_knn, best_fitness_knn = algorithm.run(task)
selected_features_knn = best_features_knn > 0.5
print('Number of selected features for K Nearest Neighbor:', selected_features_knn.sum())
print('Selected features:', ', '.join(feature_names[selected_features_knn].tolist()))

#1st layer
random_forest=RandomForestClassifier(random_state=30)
random_forest.fit(X_train1[:, selected_features_rf], y_train1)
rf_1=random_forest.predict(X_test1[:, selected_features_rf])
acc_rf_1=metrics.accuracy_score(y_test1,rf_1)*100

gb = GradientBoostingClassifier(random_state=10)
gb.fit(X_train1[:, selected_features_gb], y_train1)
gb_1=gb.predict(X_test1[:, selected_features_gb])
acc_gb_1=metrics.accuracy_score(y_test1,gb_1)*100

Support_vector = SVC()
Support_vector.fit(X_train1[:, selected_features_svm], y_train1)
svm_1=Support_vector.predict(X_test1[:, selected_features_svm])
acc_svm_1=metrics.accuracy_score(y_test1,svm_1)*100

X_train = pd.DataFrame(X_train1, columns=X.columns)
X_train['label'] = y_train1
X_test = pd.DataFrame(X_test1, columns=X.columns)

if acc_rf_1 >= acc_svm_1 and acc_rf_1 >= acc_gb_1 :
    X_test['label'] = y_test1
    X_test['rf_pre']=rf_1
    X_test['match'] = np.where(X_test['label'] == X_test['rf_pre'], 'True', 'False')

    FP_FN_RF_1=X_test[X_test['match']=='False']
    FP_FN_RF_1=FP_FN_RF_1.drop(['rf_pre','match'], axis=1)
    print("RF Provides best accuracy")
    print("Random_Forest_FP_FN: ",len(FP_FN_RF_1))
    
    TP_TN_RF_1=X_test[X_test['match']=='True']
    TP_TN_RF_1=TP_TN_RF_1.drop(['rf_pre','match'], axis=1)
    train2 = pd.concat([X_train, TP_TN_RF_1])
    print("Random_Forest_TN_TP: ",len(TP_TN_RF_1))

elif acc_svm_1>=acc_rf_1 and acc_svm_1>=acc_gb_1 :
    X_test['label'] = y_test1
    X_test['svm_pre']=svm_1     
    X_test['match'] = np.where(X_test['label'] == X_test['svm_pre'], 'True', 'False')

    FP_FN_SVM_1=X_test[X_test['match']=='False']
    FP_FN_SVM_1=FP_FN_SVM_1.drop(['svm_pre','match'], axis=1)
    print("SVM Provides best accuracy")
    print("SVM_FP_FN: ",len(FP_FN_SVM_1))
    
    TP_TN_SVM_1=X_test[X_test['match']=='True']
    TP_TN_SVM_1=TP_TN_SVM_1.drop(['svm_pre','match'], axis=1)
    train2 = pd.concat([X_train, TP_TN_SVM_1])
    print("SVM_TN_TP: ",len(TP_TN_SVM_1))
         
else:
    X_test['label'] = y_test1
    X_test['gb_pre']=gb_1
    X_test['match'] = np.where(X_test['label'] ==  X_test['gb_pre'], 'True', 'False')
    FP_FN_GB_1= X_test[X_test['match']=='False']
    FP_FN_GB_1=FP_FN_GB_1.drop(['gb_pre','match'], axis=1)
    print("GB Provides best accuracy: ")
    print("Gradient_Boosting_FP_FN: ",len(FP_FN_GB_1))
    
    TP_TN_GB_1= X_test[X_test['match']=='True']
    TP_TN_GB_1=TP_TN_GB_1.drop(['gb_pre','match'], axis=1)
    train2 = pd.concat([X_train, TP_TN_GB_1])
    print("Gradient_Boosting_TN_TP: ",len(TP_TN_GB_1))

#2nd layer
random_forest=RandomForestClassifier(random_state=30)
random_forest.fit(train2[feature_names[selected_features_rf]], train2['label'])
rf_2=random_forest.predict(FP_FN_SVM_1[feature_names[selected_features_rf]])
acc_rf_2=metrics.accuracy_score(FP_FN_SVM_1['label'],rf_2)*100

gb = GradientBoostingClassifier(random_state=10)
gb.fit(train2[feature_names[selected_features_gb]], train2['label'])
gb_2=gb.predict(FP_FN_SVM_1[feature_names[selected_features_gb]])
acc_gb_2=metrics.accuracy_score(FP_FN_SVM_1['label'],gb_2)*100

Support_vector = SVC()
Support_vector.fit(train2[feature_names[selected_features_svm]], train2['label'])
svm_2=Support_vector.predict(FP_FN_SVM_1[feature_names[selected_features_svm]])
acc_svm_2=metrics.accuracy_score(FP_FN_SVM_1['label'],svm_2)*100

if acc_rf_2 >= acc_svm_2 and acc_rf_2 >= acc_gb_2 :
    FP_FN_SVM_1['label'] = FP_FN_SVM_1['label']
    FP_FN_SVM_1['rf_pre']=rf_2
    FP_FN_SVM_1['match'] = np.where(FP_FN_SVM_1['label'] == FP_FN_SVM_1['rf_pre'], 'True', 'False')

    FP_FN_RF_2=FP_FN_SVM_1[FP_FN_SVM_1['match']=='False']
    FP_FN_RF_2=FP_FN_RF_2.drop(['rf_pre','match'], axis=1)
    print("RF Provides best accuracy")
    print("Random_Forest_FP_FN: ",len(FP_FN_RF_2))
    
    TP_TN_RF_2=FP_FN_SVM_1[FP_FN_SVM_1['match']=='True']
    TP_TN_RF_2=TP_TN_RF_2.drop(['rf_pre','match'], axis=1)
    train3 = pd.concat([train2, TP_TN_RF_2])
    print("Random_Forest_TN_TP: ",len(TP_TN_RF_2))

elif acc_svm_2>=acc_rf_2 and acc_svm_2>=acc_gb_2 :
    FP_FN_SVM_1['label'] = FP_FN_SVM_1['label']
    FP_FN_SVM_1['svm_pre']=svm_2     
    FP_FN_SVM_1['match'] = np.where(FP_FN_SVM_1['label'] == FP_FN_SVM_1['svm_pre'], 'True', 'False')

    FP_FN_SVM_2=FP_FN_SVM_1[FP_FN_SVM_1['match']=='False']
    FP_FN_SVM_2=FP_FN_SVM_2.drop(['svm_pre','match'], axis=1)
    print("SVM Provides best accuracy")
    print("SVM_FP_FN: ",len(FP_FN_SVM_2))
    
    TP_TN_SVM_2=FP_FN_SVM_1[FP_FN_SVM_1['match']=='True']
    TP_TN_SVM_2=TP_TN_SVM_2.drop(['svm_pre','match'], axis=1)
    train3 = pd.concat([train2, TP_TN_SVM_2])
    print("SVM_TN_TP: ",len(TP_TN_SVM_2))
         
else:
    FP_FN_SVM_1['label'] = FP_FN_SVM_1['label']
    FP_FN_SVM_1['gb_pre']=gb_2
    FP_FN_SVM_1['match'] = np.where(FP_FN_SVM_1['label'] ==  FP_FN_SVM_1['gb_pre'], 'True', 'False')
    
    FP_FN_GB_2= FP_FN_SVM_1[FP_FN_SVM_1['match']=='False']
    FP_FN_GB_2=FP_FN_GB_2.drop(['gb_pre','match'], axis=1)
    print("GB Provides best accuracy: ")
    print("Gradient_Boosting_FP_FN: ",len(FP_FN_GB_2))
    
    TP_TN_GB_2= FP_FN_SVM_1[FP_FN_SVM_1['match']=='True']
    TP_TN_GB_2=TP_TN_GB_2.drop(['gb_pre','match'], axis=1)
    train3 = pd.concat([train2, TP_TN_GB_2])
    print("Gradient_Boosting_TN_TP: ",len(TP_TN_GB_2))

#3rd layer
random_forest=RandomForestClassifier(random_state=30)
random_forest.fit(train3[feature_names[selected_features_rf]], train3['label'])
rf_3=random_forest.predict(FP_FN_SVM_2[feature_names[selected_features_rf]])
acc_rf_3=metrics.accuracy_score(FP_FN_SVM_2['label'],rf_3)*100

gb = GradientBoostingClassifier(random_state=10)
gb.fit(train3[feature_names[selected_features_gb]], train3['label'])
gb_3=gb.predict(FP_FN_SVM_2[feature_names[selected_features_gb]])
acc_gb_3=metrics.accuracy_score(FP_FN_SVM_2['label'],gb_3)*100

Support_vector = SVC()
Support_vector.fit(train3[feature_names[selected_features_svm]], train3['label'])
svm_3=Support_vector.predict(FP_FN_SVM_2[feature_names[selected_features_svm]])
acc_svm_3=metrics.accuracy_score(FP_FN_SVM_2['label'],svm_3)*100

if acc_rf_3 >= acc_svm_3 and acc_rf_3 >= acc_gb_3 :
    FP_FN_SVM_2['label'] = FP_FN_SVM_2['label']
    FP_FN_SVM_2['rf_pre']=rf_3
    FP_FN_SVM_2['match'] = np.where(FP_FN_SVM_2['label'] == FP_FN_SVM_2['rf_pre'], 'True', 'False')

    FP_FN_RF_3=FP_FN_SVM_2[FP_FN_SVM_2['match']=='False']
    FP_FN_RF_3=FP_FN_RF_3.drop(['rf_pre','match'], axis=1)
    print("RF Provides best accuracy")
    print("Random_Forest_FP_FN: ",len(FP_FN_RF_3))
    
    TP_TN_RF_3=FP_FN_SVM_2[FP_FN_SVM_2['match']=='True']
    TP_TN_RF_3=TP_TN_RF_3.drop(['rf_pre','match'], axis=1)
    train4 = pd.concat([train3, TP_TN_RF_3])
    print("Random_Forest_TN_TP: ",len(TP_TN_RF_3))

elif acc_svm_3>=acc_rf_3 and acc_svm_3>=acc_gb_3 :
    FP_FN_SVM_2['label'] = FP_FN_SVM_2['label']
    FP_FN_SVM_2['svm_pre']=svm_3     
    FP_FN_SVM_2['match'] = np.where(FP_FN_SVM_2['label'] == FP_FN_SVM_2['svm_pre'], 'True', 'False')

    FP_FN_SVM_3=FP_FN_SVM_2[FP_FN_SVM_2['match']=='False']
    FP_FN_SVM_3=FP_FN_SVM_3.drop(['svm_pre','match'], axis=1)
    print("SVM Provides best accuracy")
    print("SVM_FP_FN: ",len(FP_FN_SVM_3))
    
    TP_TN_SVM_3=FP_FN_SVM_2[FP_FN_SVM_2['match']=='True']
    TP_TN_SVM_3=TP_TN_SVM_3.drop(['svm_pre','match'], axis=1)
    train4 = pd.concat([train3, TP_TN_SVM_3])
    print("SVM_TN_TP: ",len(TP_TN_SVM_3))
         
else:
    FP_FN_SVM_2['label'] = FP_FN_SVM_2['label']
    FP_FN_SVM_2['gb_pre']=gb_3
    FP_FN_SVM_2['match'] = np.where(FP_FN_SVM_2['label'] ==  FP_FN_SVM_2['gb_pre'], 'True', 'False')

    FP_FN_GB_3= FP_FN_SVM_2[FP_FN_SVM_2['match']=='False']
    FP_FN_GB_3=FP_FN_GB_3.drop(['gb_pre','match'], axis=1)
    print("GB Provides best accuracy: ")
    print("Gradient_Boosting_FP_FN: ",len(FP_FN_GB_3))
    
    TP_TN_GB_3= FP_FN_SVM_2[FP_FN_SVM_2['match']=='True']
    TP_TN_GB_3=TP_TN_GB_3.drop(['gb_pre','match'], axis=1)
    train4 = pd.concat([train3, TP_TN_GB_3])
    print("Gradient_Boosting_TN_TP: ",len(TP_TN_GB_3))

#4th layer
random_forest=RandomForestClassifier(random_state=40)
random_forest.fit(train4[feature_names[selected_features_rf]], train4['label'])
rf_4=random_forest.predict(FP_FN_GB_3[feature_names[selected_features_rf]])
acc_rf_4=metrics.accuracy_score(FP_FN_GB_3['label'],rf_4)*100

gb = GradientBoostingClassifier(random_state=10)
gb.fit(train4[feature_names[selected_features_gb]], train4['label'])
gb_4=gb.predict(FP_FN_GB_3[feature_names[selected_features_gb]])
acc_gb_4=metrics.accuracy_score(FP_FN_GB_3['label'],gb_4)*100

Support_vector = SVC()
Support_vector.fit(train4[feature_names[selected_features_svm]], train4['label'])
svm_4=Support_vector.predict(FP_FN_GB_3[feature_names[selected_features_svm]])
acc_svm_4=metrics.accuracy_score(FP_FN_GB_3['label'],svm_4)*100

if acc_rf_4 >= acc_svm_4 and acc_rf_4 >= acc_gb_4 :
    FP_FN_GB_3['label'] = FP_FN_GB_3['label']
    FP_FN_GB_3['rf_pre']=rf_4
    FP_FN_GB_3['match'] = np.where(FP_FN_GB_3['label'] == FP_FN_GB_3['rf_pre'], 'True', 'False')

    FP_FN_RF_4=FP_FN_GB_3[FP_FN_GB_3['match']=='False']
    FP_FN_RF_4=FP_FN_GB_4.drop(['rf_pre','match'], axis=1)
    print("RF Provides best accuracy")
    print("Random_Forest_FP_FN: ",len(FP_FN_RF_4))
    
    TP_TN_RF_4=FP_FN_GB_3[FP_FN_GB_3['match']=='True']
    TP_TN_RF_4=TP_TN_RF_4.drop(['rf_pre','match'], axis=1)
    train5 = pd.concat([train4, TP_TN_RF_4])
    print("Random_Forest_TN_TP: ",len(TP_TN_RF_4))

elif acc_svm_4>=acc_rf_4 and acc_svm_4>=acc_gb_4 :
    FP_FN_GB_3['label'] = FP_FN_GB_3['label']
    FP_FN_GB_3['svm_pre']=svm_4     
    FP_FN_GB_3['match'] = np.where(FP_FN_GB_3['label'] == FP_FN_GB_3['svm_pre'], 'True', 'False')

    FP_FN_SVM_4=FP_FN_GB_3[FP_FN_GB_3['match']=='False']
    FP_FN_SVM_4=FP_FN_SVM_4.drop(['svm_pre','match'], axis=1)
    print("SVM Provides best accuracy")
    print("SVM_FP_FN: ",len(FP_FN_SVM_4))
    
    TP_TN_SVM_4=FP_FN_GB_3[FP_FN_GB_3['match']=='True']
    TP_TN_SVM_4=TP_TN_SVM_4.drop(['svm_pre','match'], axis=1)
    train5 = pd.concat([train4, TP_TN_SVM_4])
    print("SVM_TN_TP: ",len(TP_TN_SVM_4))
         
else:
    FP_FN_GB_3['label'] = FP_FN_GB_3['label']
    FP_FN_GB_3['gb_pre']=gb_4
    FP_FN_GB_3['match'] = np.where(FP_FN_GB_3['label'] ==  FP_FN_GB_3['gb_pre'], 'True', 'False')

    FP_FN_GB_4= FP_FN_GB_3[FP_FN_GB_3['match']=='False']
    FP_FN_GB_4=FP_FN_GB_4.drop(['gb_pre','match'], axis=1)
    print("GB Provides best accuracy: ")
    print("Gradient_Boosting_FP_FN: ",len(FP_FN_GB_4))
    
    TP_TN_GB_4= FP_FN_GB_3[FP_FN_GB_3['match']=='True']
    TP_TN_GB_4=TP_TN_GB_4.drop(['gb_pre','match'], axis=1)
    train5 = pd.concat([train4, TP_TN_GB_4])
    print("Gradient_Boosting_TN_TP: ",len(TP_TN_GB_4))

#5th layer
random_forest=RandomForestClassifier(random_state=50)
random_forest.fit(train5[feature_names[selected_features_rf]], train5['label'])
rf_5=random_forest.predict(FP_FN_RF_4[feature_names[selected_features_rf]])
acc_rf_5=metrics.accuracy_score(FP_RF_GB_4['label'],rf_5)*100

gb = GradientBoostingClassifier(random_state=10)
gb.fit(train5[feature_names[selected_features_gb]], train5['label'])
gb_5=gb.predict(FP_FN_RF_4[feature_names[selected_features_gb]])
acc_gb_5=metrics.accuracy_score(FP_FN_RF_4['label'],gb_5)*100

Support_vector = SVC()
Support_vector.fit(train5[feature_names[selected_features_svm]], train5['label'])
svm_5=Support_vector.predict(FP_FN_RF_4[feature_names[selected_features_svm]])
acc_svm_5=metrics.accuracy_score(FP_FN_RF_4['label'],svm_5)*100

if acc_rf_5 >= acc_svm_5 and acc_rf_5 >= acc_gb_5 :
    FP_FN_RF_2['label'] = FP_FN_RF_2['label']
    FP_FN_RF_2['rf_pre']=rf_5
    FP_FN_RF_2['match'] = np.where(FP_FN_RF_2['label'] == FP_FN_RF_2['rf_pre'], 'True', 'False')

    FP_FN_RF_5=FP_FN_RF_2[FP_FN_RF_2['match']=='False']
    FP_FN_RF_5=FP_FN_RF_5.drop(['rf_pre','match'], axis=1)
    print("RF Provides best accuracy")
    print("Random_Forest_FP_FN: ",len(FP_FN_RF_5))
    
    TP_TN_RF_5=FP_FN_RF_2[FP_FN_RF_2['match']=='True']
    TP_TN_RF_5=TP_TN_RF_5.drop(['rf_pre','match'], axis=1)
    train6 = pd.concat([train5, TP_TN_RF_5])
    print("Random_Forest_TN_TP: ",len(TP_TN_RF_5))

elif acc_svm_5>=acc_rf_5 and acc_svm_5>=acc_gb_5 :
    FP_FN_RF_2['label'] = FP_FN_RF_2['label']
    FP_FN_RF_2['svm_pre']=svm_5     
    FP_FN_RF_2['match'] = np.where(FP_FN_RF_2['label'] == FP_FN_RF_2['svm_pre'], 'True', 'False')

    FP_FN_SVM_5=FP_FN_RF_2[FP_FN_RF_2['match']=='False']
    FP_FN_SVM_5=FP_FN_SVM_5.drop(['svm_pre','match'], axis=1)
    print("SVM Provides best accuracy")
    print("SVM_FP_FN: ",len(FP_FN_SVM_4))
    
    TP_TN_SVM_5=FP_FN_RF_2[FP_FN_RF_2['match']=='True']
    TP_TN_SVM_5=TP_TN_SVM_5.drop(['svm_pre','match'], axis=1)
    train6 = pd.concat([train6, TP_TN_SVM_5])
    print("SVM_TN_TP: ",len(TP_TN_SVM_5))
         
else:
    FP_FN_RF_2['label'] = FP_FN_RF_2['label']
    FP_FN_RF_2['gb_pre']=gb_5
    FP_FN_RF_2['match'] = np.where(FP_FN_RF_2['label'] ==  FP_FN_RF_2['gb_pre'], 'True', 'False')

    FP_FN_GB_5= FP_FN_RF_2[FP_FN_RF_2['match']=='False']
    FP_FN_GB_5=FP_FN_GB_5.drop(['gb_pre','match'], axis=1)
    print("GB Provides best accuracy: ")
    print("Gradient_Boosting_FP_FN: ",len(FP_FN_GB_5))
    
    TP_TN_GB_5= FP_FN_RF_2[FP_FN_RF_2['match']=='True']
    TP_TN_GB_5=TP_TN_GB_5.drop(['gb_pre','match'], axis=1)
    train6 = pd.concat([train5, TP_TN_GB_5])
    print("Gradient_Boosting_TN_TP: ",len(TP_TN_GB_5))

#6th layer
random_forest=RandomForestClassifier(random_state=60)
random_forest.fit(train6[feature_names[selected_features_rf]], train6['label'])
rf_6=random_forest.predict(FP_FN_RF_5[feature_names[selected_features_rf]])
acc_rf_6=metrics.accuracy_score(FP_FN_RF_5['label'],rf_6)*100

gb = GradientBoostingClassifier(random_state=10)
gb.fit(train6[feature_names[selected_features_gb]], train6['label'])
gb_6=gb.predict(FP_FN_RF_5[feature_names[selected_features_gb]])
acc_gb_6=metrics.accuracy_score(FP_FN_RF_5['label'],gb_6)*100

Support_vector = SVC()
Support_vector.fit(train6[feature_names[selected_features_svm]], train6['label'])
svm_6=Support_vector.predict(FP_FN_RF_5[feature_names[selected_features_svm]])
acc_svm_6=metrics.accuracy_score(FP_FN_RF_5['label'],svm_6)*10

if acc_rf_6 >= acc_svm_6 and acc_rf_6 >= acc_gb_6 :
    FP_FN_RF_5['label'] = FP_FN_RF_5['label']
    FP_FN_RF_5['rf_pre']=rf_6
    FP_FN_RF_5['match'] = np.where(FP_FN_SVM_5['label'] == FP_FN_RF_5['rf_pre'], 'True', 'False')

    FP_FN_RF_6=FP_FN_SVM_5[FP_FN_SVM_5['match']=='False']
    FP_FN_RF_6=FP_FN_SVM_6.drop(['rf_pre','match'], axis=1)
    print("RF Provides best accuracy")
    print("Random_Forest_FP_FN: ",len(FP_FN_RF_5))
    
    TP_TN_RF_6=FP_FN_SVM_5[FP_FN_SVM_5['match']=='True']
    TP_TN_RF_6=TP_TN_RF_6.drop(['rf_pre','match'], axis=1)
    train7 = pd.concat([train6, TP_TN_RF_6])
    print("Random_Forest_TN_TP: ",len(TP_TN_RF_6))

elif acc_svm_6>=acc_rf_6 and acc_svm_6>=acc_gb_6 :
    FP_FN_RF_5['label'] = FP_FN_RF_5['label']
    FP_FN_RF_5['svm_pre']=svm_6     
    FP_FN_RF_5['match'] = np.where(FP_FN_SVM_5['label'] == FP_FN_RF_5['svm_pre'], 'True', 'False')

    FP_FN_SVM_6=FP_FN_SVM_5[FP_FN_SVM_5['match']=='False']
    FP_FN_SVM_6=FP_FN_SVM_6.drop(['svm_pre','match'], axis=1)
    print("SVM Provides best accuracy")
    print("SVM_FP_FN: ",len(FP_FN_SVM_5))
    
    TP_TN_SVM_6=FP_FN_SVM_5[FP_FN_SVM_5['match']=='True']
    TP_TN_SVM_6=TP_TN_SVM_6.drop(['svm_pre','match'], axis=1)
    train7 = pd.concat([train6, TP_TN_SVM_6])
    print("SVM_TN_TP: ",len(TP_TN_SVM_6))
         
else:
    FP_FN_RF_5['label'] = FP_FN_RF_5['label']
    FP_FN_RF_5['gb_pre']=gb_6
    FP_FN_RF_5['match'] = np.where(FP_FN_SVM_5['label'] ==  FP_FN_RF_5['gb_pre'], 'True', 'False')

    FP_FN_GB_6= FP_FN_RF_5[FP_FN_SVM_5['match']=='False']
    FP_FN_GB_6=FP_FN_GB_6.drop(['gb_pre','match'], axis=1)
    print("GB Provides best accuracy: ")
    print("Gradient_Boosting_FP_FN: ",len(FP_FN_GB_6))
    
    TP_TN_GB_6= FP_FN_RF_5[FP_FN_SVM_5['match']=='True']
    TP_TN_GB_6=TP_TN_GB_6.drop(['gb_pre','match'], axis=1)
    train7 = pd.concat([train6, TP_TN_GB_6])
    print("Gradient_Boosting_TN_TP: ",len(TP_TN_GB_6))

#7th layer
random_forest=RandomForestClassifier(random_state=70)
random_forest.fit(train7[feature_names[selected_features_rf]], train7['label'])
rf_7=random_forest.predict(FP_FN_GB_6[feature_names[selected_features_rf]])
acc_rf_7=metrics.accuracy_score(FP_FN_GB_6['label'],rf_7)*100

gb = GradientBoostingClassifier(random_state=10)
gb.fit(train7[feature_names[selected_features_gb]], train7['label'])
gb_7=gb.predict(FP_FN_GB_6[feature_names[selected_features_gb]])
acc_gb_7=metrics.accuracy_score(FP_FN_GB_6['label'],gb_7)*100

Support_vector = SVC()
Support_vector.fit(train7[feature_names[selected_features_svm]], train7['label'])
svm_7=Support_vector.predict(FP_FN_GB_6[feature_names[selected_features_svm]])
acc_svm_7=metrics.accuracy_score(FP_FN_GB_6['label'],svm_7)*100

if acc_rf_7 >= acc_svm_7 and acc_rf_7 >= acc_gb_7 :
    FP_FN_GB_6['label'] = FP_FN_GB_6['label']
    FP_FN_GB_6['rf_pre']=rf_7
    FP_FN_GB_6['match'] = np.where(FP_FN_GB_6['label'] == FP_FN_GB_6['rf_pre'], 'True', 'False')

    FP_FN_RF_7=FP_FN_GB_6[FP_FN_GB_6['match']=='False']
    FP_FN_RF_7=FP_FN_RF_7.drop(['rf_pre','match'], axis=1)
    print("RF Provides best accuracy")
    print("Random_Forest_FP_FN: ",len(FP_FN_RF_5))
    
    TP_TN_RF_7=FP_FN_GB_6[FP_FN_GB_6['match']=='True']
    TP_TN_RF_7=TP_TN_RF_7.drop(['rf_pre','match'], axis=1)
    train8 = pd.concat([train7, TP_TN_RF_7])
    print("Random_Forest_TN_TP: ",len(TP_TN_RF_7))

elif acc_svm_7>=acc_rf_7 and acc_svm_7>=acc_gb_7 :
    FP_FN_GB_6['label'] = FP_FN_GB_6['label']
    FP_FN_GB_6['svm_pre']=svm_7     
    FP_FN_GB_6['match'] = np.where(FP_FN_GB_6['label'] == FP_FN_GB_6['svm_pre'], 'True', 'False')

    FP_FN_SVM_7=FP_FN_GB_6[FP_FN_GB_6['match']=='False']
    FP_FN_SVM_7=FP_FN_SVM_7.drop(['svm_pre','match'], axis=1)
    print("SVM Provides best accuracy")
    print("SVM_FP_FN: ",len(FP_FN_GB_6))
    
    TP_TN_SVM_7=FP_FN_GB_6[FP_FN_GB_6['match']=='True']
    TP_TN_SVM_7=TP_TN_SVM_7.drop(['svm_pre','match'], axis=1)
    train8 = pd.concat([train7, TP_TN_SVM_7])
    print("SVM_TN_TP: ",len(TP_TN_SVM_7))
         
else:
    FP_FN_GB_6['label'] = FP_FN_GB_6['label']
    FP_FN_GB_6['gb_pre']=gb_7
    FP_FN_GB_6['match'] = np.where(FP_FN_GB_6['label'] ==  FP_FN_GB_6['gb_pre'], 'True', 'False')

    FP_FN_GB_7= FP_FN_GB_6[FP_FN_GB_6['match']=='False']
    FP_FN_GB_7=FP_FN_GB_7.drop(['gb_pre','match'], axis=1)
    print("GB Provides best accuracy: ")
    print("Gradient_Boosting_FP_FN: ",len(FP_FN_GB_7))
    
    TP_TN_GB_7= FP_FN_GB_6[FP_FN_GB_6['match']=='True']
    TP_TN_GB_7=TP_TN_GB_7.drop(['gb_pre','match'], axis=1)
    train8 = pd.concat([train7, TP_TN_GB_7])
    print("Gradient_Boosting_TN_TP: ",len(TP_TN_GB_7))

#8th layer
random_forest=RandomForestClassifier(random_state=80)
random_forest.fit(train8[feature_names[selected_features_rf]], train8['label'])
rf_8=random_forest.predict(FP_FN_SVM_7[feature_names[selected_features_rf]])
acc_rf_8=metrics.accuracy_score(FP_FN_SVM_7['label'],rf_8)*100

gb = GradientBoostingClassifier(random_state=10)
gb.fit(train8[feature_names[selected_features_gb]], train8['label'])
gb_8=gb.predict(FP_FN_SVM_7[feature_names[selected_features_gb]])
acc_gb_8=metrics.accuracy_score(FP_FN_SVM_7['label'],gb_8)*100

Support_vector = SVC()
Support_vector.fit(train8[feature_names[selected_features_svm]], train8['label'])
svm_8=Support_vector.predict(FP_FN_SVM_7[feature_names[selected_features_svm]])
acc_svm_8=metrics.accuracy_score(FP_FN_SVM_7['label'],svm_8)*100

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=7)
knn.fit(train8[feature_names[selected_features_knn]], train8['label'])
knn_8=knn.predict(FP_FN_SVM_7[feature_names[selected_features_knn]])
acc_knn_8=metrics.accuracy_score(FP_FN_SVM_7['label'],knn_8)*100

FP_FN_SVM_7['label'] = FP_FN_SVM_7['label']
FP_FN_SVM_7['knn_pre']=knn_8
FP_FN_SVM_7['match'] = np.where(FP_FN_SVM_7['label'] ==  FP_FN_SVM_7['knn_pre'], 'True', 'False')

FP_FN_KNN_8= FP_FN_SVM_7[FP_FN_SVM_7['match']=='False']
FP_FN_KNN_8=FP_FN_KNN_8.drop(['knn_pre','match'], axis=1)
print("K Nearest Neighbor_FP_FN: ",len(FP_FN_KNN_8))
    
TP_TN_KNN_8= FP_FN_SVM_7[FP_FN_SVM_7['match']=='True']
TP_TN_KNN_8=TP_TN_KNN_8.drop(['knn_pre','match'], axis=1)
print("K Nearest Neighbor_TN_TP: ",len(TP_TN_KNN_8))

#Total Accuracy
print("Accuracy: ",(len(TP_TN_SVM_1)+len(TP_TN_SVM_2)+len(TP_TN_GB_3)+len(TP_TN_RF_4)+
                    len(TP_TN_RF_5)+len(TP_TN_GB_6)+len(TP_TN_SVM_7)+len(TP_TN_KNN_8))/len(X_test))
