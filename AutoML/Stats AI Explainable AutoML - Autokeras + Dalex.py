#!/usr/bin/env python
# coding: utf-8

# ## Explainable AutoML - Titanic Survival Classification Demo

# In[1]:


# Author Hussain Abbas
# Copyright © Stats AI 2021. All Rights Reserved

import tensorflow as tf
import autokeras as ak
from tensorflow.keras import backend as K
import keras_tuner
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, fbeta_score, roc_curve
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm


# In[2]:


# Verify GPU is detected and working
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


# In[3]:


TRAIN_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/train.csv"
TEST_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/eval.csv"

#datasets located in C:/Users/USER/.keras/datasets

train_file_path = tf.keras.utils.get_file("train.csv", TRAIN_DATA_URL)
test_file_path = tf.keras.utils.get_file("eval.csv", TEST_DATA_URL)

train_data = pd.read_csv(train_file_path)
test_data = pd.read_csv(test_file_path)

df = pd.concat([train_data, test_data])
df.drop_duplicates(inplace=True)
df = df.reset_index()
df = df.drop(['index'], axis=1)

print('All Data Summary')
print(df.describe())
print('\n')

print('Train Data Summary')
print(train_data.describe())
print('\n')

print('Test Data Summary')
print(test_data.describe())


# In[4]:


print('Train Data')
train_data.head()


# In[5]:


print('Test Data')
test_data.head()


# In[6]:


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f_beta_score(y_true, y_pred):
    
    a = 0.5 ** 2
    b = 1 + a
    
    precision = precision_m(y_true, y_pred)
    
    recall = recall_m(y_true, y_pred)
    
    return b*((precision*recall)/(a*precision+recall+K.epsilon()))


def ak_predict(model, data):
    
    pred_input = data.astype(np.compat.unicode)
    predicted = model.predict(pred_input).flatten()
    pred_result = predicted
    
    #cut_off = 0.5

    #pred_result = [1 if x > cut_off else 0 for x in predicted]
    
    return pred_result

def jdl(y_true, y_pred, smooth=100):
    """
    Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))
    
    The jaccard distance loss is usefull for unbalanced datasets. This has been
    shifted so it converges on 0 and is smoothed to avoid exploding or disapearing
    gradient.
    
    Ref: https://en.wikipedia.org/wiki/Jaccard_index
    
    @url: https://gist.github.com/wassname/f1452b748efcbeb4cb9b1d059dce6f96
    @author: wassname
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth


# In[7]:


from tensorflow.keras.utils import CustomObjectScope
from sklearn.utils import class_weight

with CustomObjectScope({'f_beta_score': f_beta_score, 
                       'jdl': jdl, }):

    results = []

    # number of times we partition the data into training/test set
    outer_loop_folds = 1

    # number of times we partition the training data into training/validation set
    inner_loop_folds = 1

    #max_trials: Default= 100. The max num of different models to try
    num_trials = 20

    #epochs: If unspecified, we use epochs equal to 1000 and early stopping with patience equal to 30
    
    Early_Stopping = tf.keras.callbacks.EarlyStopping(monitor='val_f_beta_score', patience=101)


    for j in tqdm(range(outer_loop_folds)):

        #Randomly split df into 80% train, 20% test
        
        x_train, x_test, y_train, y_test = train_test_split(df.drop('survived', axis=1), 
                                                    df.survived, test_size=0.2, 
                                                   stratify = df.survived)

        for i in tqdm(range(inner_loop_folds)):

            # Further randomly split the 80% train into 64% train and 16% validation
            
            x_inner_train, x_inner_val, y_inner_train, y_inner_val = train_test_split(x_train,
                                                    y_train, test_size=0.2, 
                                                   stratify = y_train)   
            
            w = y_inner_train.value_counts(normalize = True)[0]/y_inner_train.value_counts(normalize = True)[1]
            cw = {0: 1., 1: w}
            #cw = {0: 1., 1: 0.5}


            # Try max_trial different models
            clf = ak.StructuredDataClassifier(
                overwrite=True, 
                max_trials = num_trials,
                
                #tuner = 'random',
                #tuner = 'hyperband',
                tuner = 'bayesian', 

                metrics=[jdl,
                        'binary_crossentropy', 
                         tf.keras.metrics.AUC(name='auc'), 
                        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                         tf.keras.metrics.Precision(name='precision'),                     
                         tf.keras.metrics.Recall(name='recall'),
                         f_beta_score],

                objective=keras_tuner.Objective('val_f_beta_score', direction='max'),
                #objective=keras_tuner.Objective('val_jdl', direction='min'),
                
                #loss = jdl,

            )

            try:
                # Fit the best model
                clf.fit(x_inner_train, y_inner_train, 
                        validation_data = (x_inner_val, y_inner_val),
                        #class_weight = cw
                        epochs = 3000,
                        callbacks = [Early_Stopping]
                       )

                # Predict with the best model
                x = clf.evaluate(x_test, y_test)
                x_test_loss, x_jdl, x_bc, x_auc, x_accuracy, x_precision, x_recall, x_f_beta_score= x

                # Save the results 
                model_name = 'model_autokeras_' + str(j) + '_'+ str(i)
                
                results.append([model_name, j, i, 
                                x_test_loss, x_jdl, x_bc,
                                x_auc, x_accuracy, 
                                x_precision, x_recall, 
                                x_f_beta_score]) 

            except:
                print("Issue training model")    

            try: 
                # Save the model after each j, i iteration
                model = clf.export_model()
                model.save(model_name, save_format="tf")    

            except: 
                print("Issue saving model")

                
results = pd.DataFrame(results, columns = ['model_name', 'j', 'i', 'Test_loss','Loss:JDL', 'Loss:Binary Cross Entropy',
                                           'AUC', 'Accuracy', 'Precision', 'Recall', 'F_Beta_Score'])
    


# In[8]:


get_ipython().run_cell_magic('capture', 'cap --no-stderr', 'clf.tuner.results_summary()')


# In[9]:


with open('model_val_info.txt', 'w') as f:
    f.write(str(cap))

z = open('model_val_info.txt').read()
z = re.findall(r'Score: ([^/]+)', z)
z = np.array([x.split()[0] for x in z]).astype(np.float)


# In[10]:


results.describe()


# In[11]:


results['F1_Beta_Val'] = z.max()


# In[12]:


results


# In[13]:


#best_model = results.loc[np.argmax(results.test_accuracy)].model_name

best_model = results.loc[np.argmax(results.F_Beta_Score)].model_name


best_model


# In[14]:


from tensorflow.keras.models import load_model

my_custom_objects={'f_beta_score': f_beta_score, 
                       'jdl': jdl, }
my_custom_objects.update(ak.CUSTOM_OBJECTS)

model_ak = load_model(best_model, custom_objects=my_custom_objects)


# In[15]:


model_ak.summary()


# In[16]:


# type: pandas.core.frame.DataFrame
pred_input = x_test.astype(np.compat.unicode)

# type: numpy.ndarray
predicted = model_ak.predict(pred_input).flatten()

cut_off = 0.5

pred_result = [1 if x > cut_off else 0 for x in predicted]
pred_result = np.array(pred_result)

actual = y_test.to_numpy()
actual = actual.flatten()

cm = tf.math.confusion_matrix(actual, pred_result)
cm = cm/cm.numpy().sum(axis=1)[:, tf.newaxis]

sns.heatmap(
    cm, annot=True,
    xticklabels=['no', 'yes'],
    yticklabels=['no', 'yes'])
plt.xlabel("Predicted")
plt.ylabel("True")


'''
https://towardsdatascience.com/accuracy-precision-recall-or-f1-331fb37c5cb9
https://stackoverflow.com/questions/44172162/f1-score-vs-roc-auc

- Maximize Precision when False Positives are of concern 
- Maximize Recall when False Negatives are of concern
- Maximize F1 Score when both are important and classes are unbalanced

'''

auc_score = roc_auc_score(actual, pred_result)
precision = precision_score(actual, pred_result)
recall = recall_score(actual, pred_result)
f_beta = fbeta_score(actual, pred_result, beta = 0.5)

print("Cut-Off:", cut_off)
print("ROC-AUC-Score:", auc_score)
print('Precision: ' + str(precision))
print('Recall: ' + str(recall))      
print('F_Beta: ' + str(f_beta))

y_test_classes = list(set(y_test))

# print Confusion Matrix from Sklearn
cm = confusion_matrix(actual, pred_result, labels = y_test_classes)

#cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels = y_test_classes)
disp.plot(); 


# In[17]:


# compute true positive rate and false positive rate
false_positive_rate, true_positive_rate, thresholds = roc_curve(actual, pred_result)

# plotting them against each other
def plot_roc_curve(false_positive_rate, true_positive_rate, label=None):
    plt.plot(false_positive_rate, true_positive_rate, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'r', linewidth=4)
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate (FPR)', fontsize=16)
    plt.ylabel('True Positive Rate (TPR)', fontsize=16)

plt.figure(figsize=(14, 7))
plot_roc_curve(false_positive_rate, true_positive_rate)
plt.show()

# https://towardsdatascience.com/predicting-the-survival-of-titanic-passengers-30870ccc7e8
# https://stackoverflow.com/questions/44172162/f1-score-vs-roc-auc


# In[18]:


# Naive Random Coin Flip Classifier Performance

predicted = np.random.randint(0,2, size = len(y_test))
pred_result = predicted.flatten()

actual = y_test.to_numpy()
actual = actual.flatten()

cm = tf.math.confusion_matrix(actual, pred_result)
cm = cm/cm.numpy().sum(axis=1)[:, tf.newaxis]

sns.heatmap(
    cm, annot=True,
    xticklabels=['no', 'yes'],
    yticklabels=['no', 'yes'])
plt.xlabel("Predicted")
plt.ylabel("True")

#true_positives = tf.math.count_nonzero(pred_result * actual)
#true_negatives = tf.math.count_nonzero((pred_result - 1) * (actual - 1))
#false_positives = tf.math.count_nonzero(pred_result * (actual - 1))
#false_negatives = tf.math.count_nonzero((pred_result - 1) * actual)

#precision = true_positives / (true_positives + false_positives)
#recall = true_positives / (true_positives + false_negatives)
#f1 = 2 * precision * recall / (precision + recall)

#print("Precision: " + str(np.array(precision).flatten()[0]))
#print("Recall: " + str(np.array(recall).flatten()[0])) 
#print("F1: " + str(np.array(f1).flatten()[0]))
#print('')

'''
https://towardsdatascience.com/accuracy-precision-recall-or-f1-331fb37c5cb9
https://stackoverflow.com/questions/44172162/f1-score-vs-roc-auc

- Maximize Precision when False Positives are of concern 
- Maximize Recall when False Negatives are of concern
- Maximize F1 Score when both are important and classes are unbalanced

'''
auc_score = roc_auc_score(actual, pred_result)
precision = precision_score(actual, pred_result)
recall = recall_score(actual, pred_result)
f_beta = fbeta_score(actual, pred_result, beta = 1)

print("ROC-AUC-Score:", auc_score)
print('Precision: ' + str(precision))
print('Recall: ' + str(recall))      
print('F_Beta: ' + str(f_beta))

y_test_classes = list(set(y_test))

# print Confusion Matrix from Sklearn
cm = confusion_matrix(actual, pred_result, labels = y_test_classes)

#cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels = y_test_classes)
disp.plot(); 


# ## Explainable AI using Dalex

# In[19]:


import dalex as dx


# In[20]:


X, y = df.drop('survived', axis=1), df.survived
n, p = X.shape


# In[21]:


explainer_keras = dx.Explainer(model_ak, 
                               data = X, 
                               y = y,
                               predict_function = ak_predict,
                               label = 'autokeras',
                               #predict_function = dx._explainer.yhat.yhat_tf_classification,
                               model_type = 'classification'
                              )


# In[22]:


explainer_keras.model_performance()


# In[23]:


explainer_keras.model_diagnostics().result


# In[24]:


explainer_keras.model_parts().plot()


# In[25]:


explainer_keras.model_profile().plot(variables=['sex', 'class', 'fare', 'deck', 
                                                'n_siblings_spouses', 'embark_town', 
                                                'parch', 'alone', 'age'])


# In[26]:


explainer_keras.predict_parts(X.loc[0], type='shap').plot()


# In[27]:


explainer_keras.predict_parts(X.loc[1], type='shap').plot()


# In[28]:


X_one_hot = pd.get_dummies(X, drop_first=True)

X_one_hot


# In[29]:


from sklearn import tree

clf = tree.DecisionTreeClassifier(max_features = 5, max_depth = 3)

X_one_hot = pd.get_dummies(X, drop_first=True)

clf = clf.fit(X_one_hot, y)


# In[30]:


df[df.sex == 'male'].survived.value_counts()


# In[31]:


df[df.sex == 'female'].survived.value_counts()


# In[32]:


df.survived.value_counts()


# In[33]:


#clf.classes_


# In[34]:


fn = list(X_one_hot.columns)

cn = ['did not survive', 'survived']

#cn = ['survived', 'did not survive']


fig, axes = plt.subplots(nrows = 1,
                         ncols = 1,
                         figsize = (30,30))
                         #dpi=500)

tree.plot_tree(clf,
               feature_names = fn, 
               class_names=cn,
               filled = True, fontsize = 15)

#fig.savefig('imagename.png')


# ## working version

# In[35]:


import dalex as dx

data = pd.read_csv("https://raw.githubusercontent.com/pbiecek/xai-happiness/main/happiness.csv", index_col=0)
data.head()


# In[36]:


X, y = data.drop('score', axis=1), data.score
n, p = X.shape

X


# In[37]:


y


# In[38]:


#tf.random.set_seed(11)

normalizer  = tf.keras.layers.experimental.preprocessing.Normalization(input_shape=[p,])
normalizer.adapt(X.to_numpy())

model = tf.keras.Sequential([
    normalizer,
    tf.keras.Input(shape=(p,)),
    tf.keras.layers.Dense(p*2, activation='relu'),
    tf.keras.layers.Dense(p*3, activation='relu'),
    tf.keras.layers.Dense(p*2, activation='relu'),
    tf.keras.layers.Dense(p, activation='relu'),
    tf.keras.layers.Dense(1, activation='linear')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.mae
)

model.fit(X, y, batch_size=int(n/10), epochs=2000, verbose=False)


# In[39]:


#type(model)


# In[40]:


#model.output_shape


# In[41]:


explainer = dx.Explainer(model, X, y, label='happiness')


# In[42]:


#explainer_new = dx.Explainer(model, X, y, label='happiness', 
#                            predict_function = dx._explainer.yhat.yhat_tf_regression)


# In[43]:


#explainer.predict_function


# In[44]:


#dx._explainer.yhat.yhat_tf_regression(model, X)


# In[45]:


#explainer.residual_function


# In[46]:


#dx._explainer.checks.check_residual_function.residual_function

#def rf(_model, _data, _y):
#    return _y - dx._explainer.yhat.yhat_tf_regression(model, X)

#rf(model, X, y)


# In[47]:


explainer.model_performance()


# In[48]:


explainer.model_parts().plot()


# In[49]:


explainer.model_profile().plot(variables=['social_support', 'healthy_life_expectancy',
                                          'gdp_per_capita', 'freedom_to_make_life_choices'])


# In[50]:


explainer.model_diagnostics().plot(variable='social_support', yvariable="abs_residuals", marker_size=5, line_width=3)


# In[51]:


explainer.model_diagnostics().result


# In[52]:


explainer.predict_parts(X.loc['Poland'], type='shap').plot()


# In[53]:


pp_list = []

for country in ['Afghanistan', 'Belgium', 'China', 'Denmark', 'Ethiopia']:
    pp = explainer.predict_parts(X.loc[country], type='break_down')
    pp.result.label = country
    pp_list += [pp]

pp_list[0].plot(pp_list[1::], min_max=[2.5, 8.5])


# In[54]:


lime_explanation = explainer.predict_surrogate(X.loc['United States'], mode='regression')

lime_explanation.plot()


# In[55]:


lime_explanation.result


# In[56]:


surrogate_model = explainer.model_surrogate(max_vars=4, max_depth=3)
surrogate_model.performance


# In[57]:


surrogate_model.plot()

