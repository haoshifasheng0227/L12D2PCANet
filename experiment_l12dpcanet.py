import re
import seaborn as sns;sns.set()
from l12dpcanet import *
from sklearn import svm
import os

# to change training set and test set:
# change the directory path

# DIRECTORY_TRAIN = "yale-train-test-mask/yale-train-mask-10"
# DIRECTORY_TEST = "yale-train-test-mask/yale-test-mask-10"
DIRECTORY_TRAIN = "yale-train-test/yale2/train"
DIRECTORY_TEST = "yale-train-test/yale2/test"

def list_files(directory, contains):
    return list(f for f in listdir(directory) if contains in f)
def find_files(directory, pattern):
    return [f for f in os.listdir(directory) if fnmatch.fnmatch(f, pattern)]

# not used
def makeY(DIRECTORY):
    patter_y = r'\d+'
    y=[]
    filenames = pd.DataFrame(list_files(DIRECTORY, "subject"))
    # generate split
    df = filenames[0].str.split(".", expand=True)
    df["filename"] = filenames
    df = df.rename(columns = {0:"subject", 1:"category"})
    df['subject'] = df.subject.str.replace('subject' , '')
    df.apply(pd.to_numeric, errors='coerce').dropna()
    df['subject'] = pd.to_numeric(df["subject"])
    df['subject'].unique()
    y = df['subject'].tolist()

    return y

def makeX(DIRECTORY):
    patter_y = r'\d+'
    y = []
    pattern = 'subject*'  # 替换为你要查找的特定字符
    files = find_files(DIRECTORY+'/',pattern)
    img_list = []
    for file in files:
        pixels = plt.imread(DIRECTORY+'/'+file)
        pixels = cv2.cvtColor(pixels, cv2.COLOR_BGR2GRAY)
        match = re.search(patter_y, file)
        if match:
            # 将匹配到的数字部分转换为整数并添加到列表中
            y.append(int(match.group()))
            img_list.append(pixels)
    X = np.stack(img_list)
    return X,y

# make label for dataset
x_train,y_train = makeX(DIRECTORY_TRAIN)
x_test,y_test = makeX(DIRECTORY_TEST)

# apply l12dpcanet_features
w_train_pca=get_l12dpcanet_filters(x_train)
x_train_pca=get_l12dpcanet_features(x_train,w_train_pca)
#print(x_train_pca.shape)
w_test_pca=get_l12dpcanet_filters(x_test)
x_test_pca=get_l12dpcanet_features(x_test,w_test_pca)

# use svm to recongize face and calculate accuracy
#lib_svm_model=svm_train(y_train,x_train_pca)
#y_pred=svm_predict(y_test,x_test_pca,lib_svm_model)


#Create a svm Classifier
clf = svm.SVC(kernel='rbf') # Linear Kernel

#Train the model using the training sets
clf.fit(x_train_pca, y_train)

#Predict the response for test dataset
y_pred = clf.predict(x_test_pca)
print(np.sum(y_test == y_pred)/len(y_test))
