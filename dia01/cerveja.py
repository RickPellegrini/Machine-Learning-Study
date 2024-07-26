# %%
import pandas as pd

df = pd.read_excel("../data/dados_cerveja.xlsx")
df

# %%
## Como podemos fazer a máquina aprender?

Features = ['temperatura','copo','espuma','cor']
Target = 'classe'

X = df[Features]
y = df[Target]
# %%

X = X.replace({
    'mud':1,'pint':0,
    'sim':1,'não':0,
    'escura':1, 'clara':0,
})



from sklearn import tree

arvore = tree.DecisionTreeClassifier()
arvore.fit(X,y)

# %%
import matplotlib.pyplot as plt

plt.figure(dpi=600)

tree.plot_tree(arvore, 
               class_names=arvore.classes_,
               feature_names=Features,
               filled=True)
# %%

probas = arvore.predict_proba([[1, 1, 0, 1]])[0]

pd.Series(probas, index=arvore.classes_)
# %%
