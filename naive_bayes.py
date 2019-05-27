import numpy as np
import pandas as pd

train_df = pd.read_csv('./car_bayes/train')
train_df.columns = ['First', 'Second', 'Third', 'Fourth', 'Fifth', 'Sixth', 'Label']
classes = list(train_df.groupby(['Label']).groups.keys())
print('classes: ' + str(classes))


def predict(predictors):
    results = []
    pr_options = []
    for predictor_index in range(len(predictors)):
        pr_options.append(len(train_df[train_df.columns[predictor_index]].unique()))

    for car_class in classes:
        class_group = train_df[train_df["Label"] == car_class]
        class_probability = (class_group.shape[0] / train_df.shape[0]) # / class_group.shape[0]

        result = class_probability  # initial value to pick up the probabilities product
        for predictor_index in range(len(predictors)):
            col_name = train_df.columns[predictor_index]
            pr_probability = (class_group[class_group[col_name] == predictors[predictor_index]].shape[0] + 1) /\
                             (class_group.shape[0] + pr_options[predictor_index])
            # smoothing

            result *= pr_probability
        results.append(result)

    class_label = classes[results.index(max(results))]

    print(class_label)
    return class_label


# validate precision on test dataframe
test_df = pd.read_csv('./car_bayes/test')
test_df.columns = ['First', 'Second', 'Third', 'Fourth', 'Fifth', 'Sixth', 'Label']

correct = 0
for idx, row in test_df.iterrows():
    if predict(row[:6]) == row[6]:
        correct += 1

print('Total classification accuracy: ' + str((correct / test_df.shape[0]) * 100))

while True:
    print('Enter your example (format: vhigh,vhigh,2,4,small,low): ')
    vector = input().split(',')
    predict(vector)