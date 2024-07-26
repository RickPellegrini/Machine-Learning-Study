# %%
import pandas as pd
df = pd.read_excel("../data/dados_frutas.xlsx")
df

# %%

filtro_redondada = df['Arredondada'] == 1
filtro_suculenta = df['Suculenta'] == 1
filtro_vermelha = df['Vermelha'] == 1
filtro_doce = df['Doce'] == 1

df[filtro_redondada & filtro_suculenta & filtro_vermelha & filtro_doce]

# %%
from sklearn import tree

features = ['Arredondada', 'Suculenta','Vermelha', 'Doce' ]
target = 'Fruta'

X = df[features]
y = df[target]


# %%
arvore = tree.DecisionTreeClassifier()
arvore.fit(X,y)

# %%
import matplotlib.pyplot as plt

plt.figure(dpi=600)
tree.plot_tree(arvore, 
               class_names=arvore.classes_, 
               feature_names= features,
               filled= True)

# %%
## ['Arredondada', 'Suculenta','Vermelha', 'Doce' ]
arvore.predict([[1,1,1,1]])
# %%

probas = arvore.predict_proba([[1,1,1,1]])[0]
pd.Series(probas,index=arvore.classes_)
# %%
