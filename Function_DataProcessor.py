'''
    Funções essencias para análise de dados faltantes
'''

# Importações:
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def count_na(dataframe):
    '''
        Método para contar os valores NaN
    que existem no dataframe. O método não recebe
    nenhuma entrada e devolve como saida a informação
    da coluna e sua contagem de NaN em numeros absolutos.
    '''
    
    # Contagem:
    enulo  = dataframe.isnull().sum()
    lista_var_nan = []
    # Construindo a informação:
    for variavel, valor in enulo.items():
        if valor != float(0):
            lista_var_nan.append((variavel,valor))
            print(str(variavel)+ '-> '+str(valor))
            
    return lista_var_nan 

# Funções:
def contagem_faltantes(df, lista, categoria='nan', valor_significativo=2, intervalo=None):
    '''
        Dado a necessidade de se entender a dinâmica dos dados faltantes para
    os casos de variáveis respondidas pelo entrevistador, o objetivo dessa função
    é entender quais variável possuem alguma categoria faltante e quantos porcento.

    Entrada:
        1. objeto pandas - DataFrame com os dados;
        2. lista - Uma lista contendo o conjunto das variáveis a serem contadas;
        3. string - A categoria considerada faltante que se queira analisar;
        4. booleano - Indicando se quer o valor normalizado ou não;
        5. float - Valor significativo para definir a porcentagem;
        6. lista - intervalo de frequência que se considere relevante.

    Saída:
        1. lista - Uma lista com o nome da variável que possui a respectiva
        categoria e a proporção da categoria com com relação ao total.
    '''
    
    # Configurações prévias:
    lista_resultados = [] # Lista com o resultado final;
    df_utilizado = df.copy()# Definindo uma cópia para não gerar conflito de informações;
    df_utilizado.fillna('nan', inplace=True) # Definindo NaN como uma string para ela aparercer nas contagens.
    
    # Tratamento de erros do parâmetro intervalo:
    if intervalo != None:
        try:
            #intervalo = list(intervalo)
            if len(intervalo) == 2:
                pass
            
            else:
                print('''
                    Intervalo inadequado, o aparâmetro intervalo é um fatiamento
                da faixa de frequência que se queira analisar, sendo assim,
                respeita o mesmo principio de ponto inicia e final.
                ''')
                return
            
        except:
            print('''
                A proposta do "intervalo" é estabelecer uma espécie de fatiamento para
            as frequências da informação de frequências, sendo assim, você precisa introduzir
            a informação como lista. Você pode fatiar de um ponto ao máximo ou ao minimo, ou seja,
            estabelecer só uma informação na lista, mas a entrada precisa ser uma lista.
            ''')
            
            return
    
    
    # Caso defaut (NaN):
    if intervalo == None:
        for col in lista.tolist():
            if categoria in df_utilizado[col].unique().tolist():
                contagem_categoria = df_utilizado[df_utilizado[col] == categoria][col].value_counts()
                frequencia = round(contagem_categoria/len(df_utilizado),valor_significativo)
                lista_resultados.append((col, categoria, frequencia))
    else:
        for col in lista.tolist():
            if categoria in df_utilizado[col].unique().tolist():
                contagem_categoria = df_utilizado[df_utilizado[col] == categoria][col].value_counts()
                frequencia = round(contagem_categoria/len(df_utilizado),valor_significativo)
                if float(frequencia) >= float(intervalo[0]) and float(frequencia) <= float(intervalo[1]):
                    lista_resultados.append((col, categoria, frequencia))

    # Resultados:
    if lista_resultados == []:
        return f"Nenhuma informação sobre '{categoria}' nesse conjunto de variáveis\n"

    print(f'Quantidade de variáveis: {len(lista_resultados)}')
    
    return lista_resultados


def binari_codification(dataframe, var, cat_faltante = 'nan', inplace = False):
    '''
        Função que normaliza o conjunto de dados segundo
    a literatura de dados faltantes. Ou seja, zero para os dados
    observáveis e 1 para os dados faltantes.
    
        Note que, para analisar corretamente, você precisa entender o
    que de fato significa 
    
    Entrada:
        1. objeto pandas - Dataframe contendo as variáveis;
        2. string - Nome da vaiável(coluna) a ser nomalizada;
        3. string - Categoria que será considerada como faltante;
        4. booleano - True or False para definir se essa normalização
        ira alterar no dataframe original ou não.
        
    Saída:
        1.  Uma alguma coisa contendo os dados normalizados.        
    '''
    
    df_utilizado = dataframe.copy()[var]
    
    #Casos defaut (NaN):
    if cat_faltante == 'nan':
        # Normalizando:
        df_utilizado.fillna(1, inplace = True)
        categorias = dataframe[var].unique()
        for cat in categorias:
            if cat != 1:
                df_utilizado.replace(cat,0,inplace=True)
    
        return df_utilizado
    
    # Normalizando:
    categorias = dataframe[var].unique()
    df_utilizado.replace(cat_faltante, 1, inplace = True)
    for cat in categorias:
        if cat != cat_faltante:
            df_utilizado.replace(cat,0,inplace=True)
            
    if inplace == True:
        dataframe.drop(var,axis=1)
        dataframe[var] = df_utilizado
        return dataframe

    return df_utilizado


def maiuscula(palavra):
    palavra = list(palavra)
    palavra[0] = palavra[0].upper()
    palavra = ''.join(palavra)
    return palavra

def  heat_cor(dataframe, variaveis,categoria_alvo,title=None,metodo='pearson',algarismo_significativo=2, drop=None, savefig=False,cmap=None,rotation=90):
    '''
        Função para checar visualmente a correlação dos dados faltantes entre si.
    '''
    
    if not isinstance(variaveis, (list, tuple)):
        variaveis = list(variaveis)
        variaveis = [''.join(variaveis)]

    df = dataframe.copy()[variaveis]
    
    for var in variaveis:
        categorias = df[var].unique()
        df[var].replace(categoria_alvo,1,inplace=True)
        if drop in categorias:
            print(f'Contém {drop}')
            df.drop(df[df[var] == drop].index, inplace=True)
            print(df[var].value_counts())
            
        for cat in categorias:
            if cat != categoria_alvo:
                df[var].replace(cat,0,inplace=True)
    
    # Gráfico de Correlação de Pearson:
    corr_pearson = df.corr(method=metodo)

    # Gráfico de Correlação de Pearson:
    plt.figure(figsize=(10,10))
    if title == None:
        plt.title(f'Gráfico de Correlação de {maiuscula(metodo)}')
    else:    
        plt.title(title)
    mask = np.triu(np.ones_like(corr_pearson, dtype=bool))
    if cmap == None:
        sns.heatmap(corr_pearson, annot=True, fmt=f'.{int(algarismo_significativo)}f',mask=mask,linewidth=.5)
    sns.heatmap(corr_pearson, annot=True, fmt=f'.{int(algarismo_significativo)}f',mask=mask,linewidth=.5,cmap=cmap)
    plt.xticks(rotation=rotation)
    if savefig == True:
        plt.savefig('Gráfico de Correlação de Pearson.png', bbox_inches='tight')
    plt.show()


def feature_importance(dataframe, var,title='Feature Importance',color='red',plot=True,metrics=True):
    '''
        Função criada para executar o método de feature importance
    e métricas relevantes para a análise de confiabilidade do modelo.
    
    Entrada:
        1. O dataframe;
        2. A variável algo.
        
    Saídas:
        1. As métricas relevantes
        2. A plotagem:
        
    OBS:
        Note que a função não possui o método plot, ou seja,
    funcionando individualmente, ela funciona sem problemas, porém,
    caso queria utilizi-la em uma estrutura de loop, o plt.show()
    torna-se necessário para a plotagem.
    '''
    
    dataframe = dataframe.copy()
    
    # XGboot:
    import xgboost as xgb
    from xgboost import plot_importance
    from sklearn.model_selection import train_test_split
    
    # Separando o alvo:
    y = dataframe[var]
    X = dataframe.drop([var], axis=1)
    
    # Treinando o modelo:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Estimando:
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000, max_depth=5, learning_rate=0.1)
    xgb_model.fit(X_train, y_train)
    
    from sklearn.metrics import explained_variance_score
    
    y_pred = xgb_model.predict(X_test)
    evs = explained_variance_score(y_test, y_pred)
    
    from sklearn.model_selection import GridSearchCV
    
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.1, 0.01, 0.001],
        'n_estimators': [100, 500, 1000]
    }
    
    grid_search = GridSearchCV(xgb_model, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    
    # Plotando:
    if metrics == True:
        print('Ponto de Variância Explicada:', evs)
        print('Melhores parâmetros:', grid_search.best_params_)
        print('MSE:', -grid_search.best_score_)
    if plot == True:
        plot_importance(xgb_model, max_num_features=10,color=color, title=title)
        plt.show()
        
    importance_dict = xgb_model.get_booster().get_score(importance_type='weight')
    importance_list = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    top_features = [f[0] for f in importance_list[:10]]
    
    return top_features

def plot_qtd_faltantes(dataframe, valor_significativo=2, figsize=None, title='Gráfico de Barras', xlabel='Variáveis', ylabel='Quantidade', plot=True, real_size=True, grid=True, absolute_value=True, critical_value=None):
    '''
        Função que plota a quantidade de valores faltantes no
        respectivo dataframe.
    '''
    dataframe = dataframe.copy()

    # calcula a soma das variáveis
    soma_variaveis = dataframe.sum()

    # cria um dicionário para armazenar a frequência de cada soma
    freq_somas = {}

    # percorre as somas de variáveis e armazena a frequência de cada soma no dicionário
    for i, v in enumerate(soma_variaveis):
        valor = v / len(dataframe)
        freq_somas[dataframe.columns[i]] = round(valor, valor_significativo)

    # ordena o dicionário pelas values (frequência)
    freq_somas = {k: v for k, v in sorted(freq_somas.items(), key=lambda x: x[1], reverse=True)}

    # cria uma lista de tuplas com o nome das variáveis e suas frequências
    var_freq = [(k, v) for k, v in freq_somas.items()]

    # Plotagem:
    if plot == True:
        if real_size == True:
            # Define o tamanho do gráfico
            if figsize != None:
                fig, ax = plt.subplots(figsize=figsize, dpi=100)
            else:
                fig, ax = plt.subplots(figsize=(10, 5), dpi=100)

            # Define o comprimento do eixo X como a quantidade de linhas no dataframe
            ax.set_xlim(0, len(dataframe))

            # Cria o gráfico de barras
            sns.barplot(x=[soma_variaveis[var[0]] for var in var_freq], y=[var[0] for var in var_freq], ax=ax)

            # Adiciona o texto da frequência em cada barra
            for i, var in enumerate(var_freq):
                ax.text(soma_variaveis[var[0]] + 0.5, i, str(var[1]), color='blue', fontweight='bold')

            if absolute_value == True:
                # Adiciona rótulos nas barras
                for i, var in enumerate(var_freq):
                    ax.text(soma_variaveis[var[0]] / 2, i, str(int(soma_variaveis[var[0]])),
                            color='black', ha='center', va='center', fontweight='bold')
            else:
                pass

            # Adiciona linhas de grade
            ax.grid(grid)

            if critical_value != None:
                # Adiciona uma linha de referência
                ax.axvline(critical_value, color='red', linestyle='--', linewidth=2)
            else:
                pass

            # Adiciona título e labels dos eixos
            plt.title(title)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)

            # Exibe o gráfico
            plt.show()

        elif real_size == False:
            # cria a plotagem do gráfico de barras
            ax = sns.barplot(x=[soma_variaveis[var[0]] for var in var_freq], y=[var[0] for var in var_freq])

            # adiciona o texto da frequência em cada barra
            for i, var in enumerate(var_freq):
                ax.text(soma_variaveis[var[0]] + 0.5, i, str(var[1]), color='blue', fontweight='bold')

            if absolute_value == True:
                # Adiciona rótulos nas barras
                for i, var in enumerate(var_freq):
                    ax.text(soma_variaveis[var[0]] / 2, i, str(int(soma_variaveis[var[0]])),
                            color='black', ha='center', va='center', fontweight='bold')
            else:
                pass

            # adiciona título e labels dos eixos
            plt.title(title)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)

            # exibe o gráfico
            plt.show()
        else:
            print('''
                A entrada do real_size é um booleano, sendo assim, apenas aceita
            True or False como parâmetro.
            ''')
            return freq_somas

    return freq_somas

def bootstrap(coluna, estatistic='mean', n_bootstrap = 1000):
    '''
        Função que realiza
    '''
    from sklearn.utils import resample
    import numpy as np
    
    vector = coluna.fillna('nan')
    
    for i, cat in enumerate(vector.unique()):
        valor = i+1
        print(f'categoria "{cat}" definida como valor {valor}')
        vector.replace(cat,valor,inplace=True)
    
    vector = vector.sort_values()
    if estatistic == 'mean' or estatistic == 'media':
        # Gerando amostras bootstrap e calculando a média para cada amostra
        bootstrap_means = []
        for i in range(n_bootstrap):
            sample = resample(vector, replace=True, n_samples=len(vector))
            bootstrap_means.append(np.mean(sample))
    
        # Calculando o intervalo de confiança de 95% para a média
        lower_ci = np.percentile(bootstrap_means, 2.5)
        upper_ci = np.percentile(bootstrap_means, 97.5)
        mean = np.mean(bootstrap_means)
    
        print("Média estimada: ", mean)
        print("Intervalo de confiança (95%): [{:.2f}, {:.2f}]".format(lower_ci, upper_ci))
        
    elif estatistic == 'median' or estatistic == 'mediana':
        # Gerando amostras bootstrap e calculando a mediana para cada amostra
        bootstrap_medians = []
        vector_sorted = vector.sort_values()
        for i in range(n_bootstrap):
            sample = resample(vector_sorted, replace=True, n_samples=len(vector_sorted))
            bootstrap_medians.append(np.median(sample))
        
        # Calculando o intervalo de confiança de 95% para a mediana
        lower_ci = np.percentile(bootstrap_medians, 2.5)
        upper_ci = np.percentile(bootstrap_medians, 97.5)
        median = np.median(bootstrap_medians)
        
        print("Mediana estimada: ", median)
        print("Intervalo de confiança (95%): [{:.2f}, {:.2f}]".format(lower_ci, upper_ci))
        
    else:
        print('''
            Parece que você não indicou a estatística correta, a função
        trabalha com a media ou mediana, digite uma das duas e realize o
        bootstrapping adequado para a sua situação.
        ''')