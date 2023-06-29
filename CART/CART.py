import pandas as pd  # Data maniuplation
import numpy as np  # Data manipulation

# Splitting data into train and test samples
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, mean_absolute_error, mean_squared_log_error, mean_squared_error, mean_absolute_percentage_error, median_absolute_error, max_error  # for model evaluation metrics
from sklearn import tree  # Decision metrics

import plotly.express as px  # Data visualization
import plotly.graph_objects as go  # Data visualization
import graphviz  # Plotting decision tree graphs


import os
import sys


class CART:
    def __init__(self, csvFilePath: str) -> None:
        pd.options.display.max_columns = 50

        self.file = pd.read_csv(csvFilePath, encoding='utf-8')

        # print(self.file)

    def execution(self) -> None:

        print('=====================================')
        print('CLASSIFICADOR')
        print('=====================================')
        input = self.file[['qPa', 'Pulso', 'Resp']]
        output = self.file['Risco'].values

        input_test, input_train, output_test, output_train, classification = self.fitting(
            input, output, "entropy", 10)

    def fitting(self, input, output, criterion, mdepth):

        # Create training and testing samples
        input_train, input_test, output_train, output_test = train_test_split(
            input, output, test_size=0.30)

        # Fit the model
        model = tree.DecisionTreeClassifier(
            criterion=criterion, max_depth=mdepth)

        classification = model.fit(input_train, output_train)

        # Predict class labels on training data
        pred_labels_tr = model.predict(input_train)

        # Predict class labels on a test data
        pred_labels_te = model.predict(input_test)

        # Tree summary and model evaluation metrics
        print('*************** Tree Summary ***************')
        print('Classes: ', classification.classes_)
        print('Tree Depth: ', classification.tree_.max_depth)
        print('No. of leaves: ', classification.tree_.n_leaves)
        print('No. of features: ', classification.n_features_in_)
        print('--------------------------------------------------------')
        print("")

        print('*************** Evaluation on Test Data ***************')
        score_te = model.score(input_test, output_test)
        print('Accuracy Score: ', score_te)

        # Look at classification report to evaluate the model
        print(classification_report(output_test, pred_labels_te, zero_division=0))
        print('--------------------------------------------------------')
        print("")

        print('*************** Evaluation on Training Data ***************')
        score_tr = model.score(input_train, output_train)
        print('Accuracy Score: ', score_tr)
        # Look at classification report to evaluate the model
        print(classification_report(output_train, pred_labels_tr, zero_division=0))
        print('--------------------------------------------------------')
        print("")

        # with open("resultados_treino.txt", "w") as output_file:
        #     output_file.write(f'Risco' + '\n')
        #     for row in output_train:
        #         output_file.write(str(row) + '\n')

        filedata2 = pd.read_csv(
            '../teste_cego.txt', encoding='utf-8')
        x = filedata2[['qPa', 'Pulso', 'Resp']]
        # y = output_test

        pred_labels_new = model.predict(x)

        with open("resultados_treino.txt", "w") as output_file:
            output_file.write(f'Risco' + '\n')
            for row in pred_labels_new:
                output_file.write(str(row) + '\n')

        print(pred_labels_new)

        # print('*************** Evaluation on New Data ***************')
        # score_new = model.score(x, y)
        # print('Accuracy Score: ', score_new)
        # # Look at classification report to evaluate the model
        # print(classification_report(y, pred_labels_new,zero_division=0))
        # print("f-measure (weighted):", f1_score(y,
        #       pred_labels_new, average="weighted"))

        # dot_data = tree.export_graphviz(classification, out_file=None,
        #                                 feature_names=input.columns,
        #                                 class_names=[str(list(classification.classes_)[0]), str(
        #                                     list(classification.classes_)[1])],
        #                                 filled=True,
        #                                 rounded=True,
        #                                 # rotate=True,
        #                                 )
        # graph = graphviz.Source(dot_data)

        return input_train, input_test, output_train, output_test, classification

    # def Plot_3D(X, X_test, y_test, clf, x1, x2, mesh_size, margin):

    #     # Specify a size of the mesh to be used
    #     mesh_size = mesh_size
    #     margin = margin

    #     # Create a mesh grid on which we will run our model
    #     x_min, x_max = X.iloc[:, 0].fillna(X.mean()).min(
    #     ) - margin, X.iloc[:, 0].fillna(X.mean()).max() + margin
    #     y_min, y_max = X.iloc[:, 1].fillna(X.mean()).min(
    #     ) - margin, X.iloc[:, 1].fillna(X.mean()).max() + margin
    #     xrange = np.arange(x_min, x_max, mesh_size)
    #     yrange = np.arange(y_min, y_max, mesh_size)
    #     xx, yy = np.meshgrid(xrange, yrange)

    #     # Calculate predictions on grid
    #     Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    #     Z = Z.reshape(xx.shape)

    #     # Create a 3D scatter plot with predictions
    #     fig = px.scatter_3d(x=X_test[x1], y=X_test[x2], z=y_test,
    #                         opacity=0.8, color_discrete_sequence=['black'])

    #     # Set figure title and colors
    #     fig.update_layout(  # title_text="Scatter 3D Plot with CART Prediction Surface",
    #         paper_bgcolor='white',
    #         scene=dict(xaxis=dict(title=x1,
    #                             backgroundcolor='white',
    #                             color='black',
    #                             gridcolor='#f0f0f0'),
    #                 yaxis=dict(title=x2,
    #                             backgroundcolor='white',
    #                             color='black',
    #                             gridcolor='#f0f0f0'
    #                             ),
    #                 zaxis=dict(title='Probability of Rain Tomorrow',
    #                             backgroundcolor='lightgrey',
    #                             color='black',
    #                             gridcolor='#f0f0f0',
    #                             )))

    #     # Update marker size
    #     fig.update_traces(marker=dict(size=1))

    #     # Add prediction plane
    #     fig.add_traces(go.Surface(x=xrange, y=yrange, z=Z, name='CART Prediction',
    #                             colorscale='Jet',
    #                             reversescale=True,
    #                             showscale=False,
    #                             contours={"z": {"show": True, "start": 0.5, "end": 0.9, "size": 0.5}}))
    #     fig.show()
    #     return fig

    def manualParsing(self, filePath):
        fileContent = open(filePath, 'r')

        fileData = []
        # Puts it in a matrix
        for line in fileContent:
            fileData.append(line.split(','))

        # Removes '\n' from the last collumn
        rowLength = len(fileData[0]) - 1
        for row in range(len(fileData)):
            number = fileData[row][rowLength]
            fileData[row][rowLength] = number[:-1]

        # Casts inputs to float
        for colIterator in range(len(fileData[0])-1):
            for rowIterator in range(len(fileData)):
                try:
                    fileData[rowIterator][colIterator] = float(
                        fileData[rowIterator][colIterator])
                except:
                    pass
        # Casts outputs to int
        for rowIterator in range(len(fileData)):
            try:
                fileData[rowIterator][rowLength] = int(
                    fileData[rowIterator][rowLength])
            except:
                pass

        return fileData
