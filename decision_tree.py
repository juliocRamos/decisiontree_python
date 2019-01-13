#/*
#   Codificado por Júlio César Ramos - R.A 87389
#   Utilizei como base as video aulas: https://www.youtube.com/watch?v=y6DmpG_PtN0&list=PLPOTBrypY74xS3WD0G_uzqPjCQfU6IRK-
#   Obs. Os trechos de código que estão comentados são utilizados para plotar os gráficos da divisão da base
#   e realizar testes no decorrer da execução
#   
#   Árvore de Decisão - Status: Validando acertividade do algoritmo e possiveis erros de código. Versão: Alpha.
#*/


#Imports
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline

import random
from pprint import pprint

#Importando e preparando base de dados (Iris)
df = pd.read_csv("Iris.csv") #// Para mudar a base, coloque o arquivo na mesma pasta e mude o nome aqui.
df = df.drop("Id", axis=1)   #// Remove o eixo id da base
df = df.rename(columns={"species":"label"}) #// Utiliza as especies como label
#Fim import Iris

#Importando e preparando base de dados (Titanic)
#df = pd.read_csv("Titanic.csv")
#df["label"] = df.Survived
#df = df.drop(["PassengerId", "Survived", "Name", "Ticket", "Cabin"], axis=1)

# Tratando valores faltantes
#median_age = df.Age.median()
#mode_embarked = df.Embarked.mode()[0]
#df = df.fillna({"Age": median_age, "Embarked": mode_embarked})
#Fim import Titanic

#Determina se os dados são categóricos ou contínuos
def determine_type_of_feature(df):
    feature_types = []
    n_unique_values_treshold = 15
    for feature in df.columns:
        if feature != "label":
            unique_values = df[feature].unique()
            example_value = unique_values[0]

            if (isinstance(example_value, str)) or (len(unique_values) <= n_unique_values_treshold):
                feature_types.append("categorical")
            else:
                feature_types.append("continuous")
    return feature_types

#Caso você queira testar o feature type. 
#Classifica os dados em categóricos ou continuos
#feature_types = determine_type_of_feature(df)
#i=0
#for column in df.columns:
#    print(column, "-", feature_types[i])
#    i += 1

global COLUMN_HEADERS,FEATURE_TYPES
COLUMN_HEADERS = df.columns
FEATURE_TYPES = determine_type_of_feature(df)

#Treino de divisão
def train_test_split(df, test_size):
    if isinstance(test_size, float):
        test_size = round(test_size * len(df))

    indices = df.index.tolist()
    test_indices = random.sample(population=indices, k=test_size)

    test_df = df.loc[test_indices]
    train_df = df.drop(test_indices)    
    return train_df, test_df

random.seed(0)
train_df, test_df = train_test_split(df, test_size=20)

data = train_df.values
data[:5]

#O dado é puro?
def check_purity(data):
    label_column = data[:, -1]
    unique_classes = np.unique(label_column)
    if len(unique_classes) == 1:
        return True
    else:
        return False

#Classificação de dados
def classify_data(data):
    label_column = data[:, -1]
    unique_classes, count_unique_classes = np.unique(label_column, return_counts=True)

    index = count_unique_classes.argmax()
    classification = unique_classes[index]
    return classification

#Potenciais divisões
def get_potential_splits(data):
    potential_splits = {}
    _, n_columns = data.shape
    for column_index in range(n_columns - 1):          # Exclui a coluna Label 
        values = data[:, column_index]                 # utilizada para classificaçao
        unique_values = np.unique(values)
        
        type_of_feature = FEATURE_TYPES[column_index]
        if type_of_feature == "continuous":
            potential_splits[column_index] = []
            for index in range(len(unique_values)):
                if index != 0:
                    current_value = unique_values[index]
                    previous_value = unique_values[index - 1]
                    potential_split = (current_value + previous_value) / 2

                    potential_splits[column_index].append(potential_split)
        
        # caso o tipo de dado for categórico       
        else:
            potential_splits[column_index] = unique_values
    return potential_splits

potential_splits = get_potential_splits(train_df.values)

# Por favor, se quiser ver este gráfico para ambas as bases, será
# necessário trocar os eixos x e y para valores que estão presentes na base 
#sns.lmplot(data=train_df, x="petal_width", y="petal_length", hue="label",  
#    fit_reg=False, height=6, aspect=1.5)                     

# Descomente estas 2 linhas para ver as potenciais divisões
#plt.vlines(x=potential_splits[3], ymin=1, ymax=7)           
#plt.hlines(y=potential_splits[2], xmin=0, xmax=2.5)
#plt.show() # Caso queira ver o gráfico da distribuição dos dados

#Divide a base
def split_data(data, split_column, split_value):
    
    split_column_values = data[:, split_column]

    type_of_feature = FEATURE_TYPES[split_column]
    if type_of_feature == "continuous":
        data_below = data[split_column_values <= split_value]
        data_above = data[split_column_values >  split_value]
    
    # Caso o tipo de dado é categórico   
    else:
        data_below = data[split_column_values == split_value]
        data_above = data[split_column_values != split_value]
    
    return data_below, data_above

#data_below, data_above = split_data(train_df.values, split_column=1, split_value="male")
#pprint(df.head)

split_column = 3
split_value = 0.8
#data_below, data_above = split_data(data, split_column, split_value)
#plotting_df = pd.DataFrame(data, columns=df.columns)
#sns.lmplot(data=plotting_df, x="petal_width", y="petal_length", fit_reg=False, aspect=1.5)
#plt.vlines(x=split_value, ymin=1, ymax=7)
#plt.xlim(0, 2.6)
#plt.show() #//Caso queira ver a divisão da base

#Menor entropia geral
def calculate_entropy(data):
    label_column = data[:, -1]
    _, counts = np.unique(label_column, return_counts=True)
    probabilities = counts/counts.sum()
    entropy = sum(probabilities * -np.log2(probabilities))
    return entropy

#print(calculate_entropy(data_above))

#Calcula a média geral da entropia
def calculate_overall_entropy(data_below, data_above):
    n_data_points = len(data_below) + len(data_above)
    p_data_below = len(data_below) / n_data_points
    p_data_above = len(data_above) / n_data_points

    overall_entropy = (p_data_below * calculate_entropy(data_below) + p_data_above * calculate_entropy(data_above))
    return overall_entropy

#print(calculate_overall_entropy(data_below, data_above))

#Determina a melhor divisão dos dados
def determine_best_split(data, potential_splits):
    overall_entropy = 9999
    for column_index in potential_splits:
        for value in potential_splits[column_index]:
            data_below, data_above = split_data(data, split_column = column_index, split_value = value)
            current_overall_entropy = calculate_overall_entropy(data_below, data_above)

            if current_overall_entropy <= overall_entropy:
                overall_entropy = current_overall_entropy
                best_split_column = column_index
                best_split_value = value
    return best_split_column, best_split_value

#potential_splits = get_potential_splits(data)
#test = determine_best_split(data, potential_splits)
#print(test)

#Algoritmo da arvore de decisão
def decision_tree_algorithm(df, counter=0, min_samples=2, max_depth=5):
    #Preparação dos dados
    if counter == 0:
        data = df.values
    else:
        data = df 
    
    #caso base
    if (check_purity(data)) or (len(data) < min_samples) or (counter == max_depth):        
        classification = classify_data(data)
        return classification
    #recursivo
    else:
        counter += 1

        #funções de suporte
        potential_splits = get_potential_splits(data)
        split_column, split_value = determine_best_split(data, potential_splits)
        data_below, data_above = split_data(data, split_column, split_value)

        #instancia sub-arvore
        feature_name = COLUMN_HEADERS[split_column]
        type_of_feature = FEATURE_TYPES[split_column]
        if type_of_feature == "continuous":
            question = "{} <= {}".format(feature_name, split_value)
        else:
            question = "{} == {}".format(feature_name, split_value)
        sub_tree = {question: []}

        #encontra as respostas (recursivo)
        positive_answer = decision_tree_algorithm(data_below, counter, min_samples, max_depth)
        negative_answer = decision_tree_algorithm(data_above, counter, min_samples, max_depth)
        
        if positive_answer == negative_answer:
            sub_tree = positive_answer
        else:
            sub_tree[question].append(positive_answer)
            sub_tree[question].append(negative_answer)
        return sub_tree

tree = decision_tree_algorithm(train_df, max_depth=3)

#Função de Classificação
example = test_df.iloc[0]
def classify_example(example, tree):
    question = list(tree.keys())[0]
    feature_name, comparison_operator, value = question.split(" ")

    # ask question
    if comparison_operator == "<=":
        if example[feature_name] <= float(value):
            answer = tree[question][0]
        else:
            answer = tree[question][1]
    
    # Caso o dado for categorico
    else:
        if str(example[feature_name]) == value:
            answer = tree[question][0]
        else:
            answer = tree[question][1]

    # base case
    if not isinstance(answer, dict):
        return answer
    
    # recursive part
    else:
        residual_tree = answer
        return classify_example(example, residual_tree)

classify_example(example, tree)

#Calcula acurácia
def calculate_accuracy(df, tree):
    df["classification"] = df.apply(classify_example, axis=1, args=(tree,))
    df["classification_correct"] = df.classification == df.label
    
    accuracy = df.classification_correct.mean()
    return accuracy

#Busca erros de classificação
def show_classification_errors(test_df):
    print(test_df)
    count = 0
    # Validar se é categórico ou contínuo
    # Não tive muito tempo para ajustar esta função para tratar
    # dados categóricos, estarei trabalhando no algoritmo no decorrer do ano :)
    type_of_feature = FEATURE_TYPES[split_column]
    if type_of_feature == "continuous":
        is_false = test_df[test_df['classification_correct'] == False]
        if not is_false.empty:
            print("\nClassification Errors: #########################################################  \n")
            pprint(is_false)
            print("\n ################################################################")
        else:
            print("\nNo errors detected: #########################################################  \n")
    else:
        is_false = test_df[test_df['classification_correct'] == 0]
        if not is_false.empty:
            print("\nClassification Errors: #########################################################  \n")
            pprint(is_false)
            print("\n################################################################")
        else:
            print("\nNo errors detected: #########################################################  \n")


# Se um dado foi classificado incorretamente
# a função abaixo permite analise

#Resultados da arvore
train_df, test_df = train_test_split(df, test_size=0.2)
tree = decision_tree_algorithm(train_df, max_depth=3)
#show_classification_errors(df) #Faltou uma correção minha.
print("\nArvore ")
pprint(tree, width=50)
print("\nAcurácia: "+ str(calculate_accuracy(test_df, tree)))
print("################ Fim da execução ################")


print("\n\nEste foi um algoritmo está em fase de validação.")
print("\n\nMay the force be with you!!")


