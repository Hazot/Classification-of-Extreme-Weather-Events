Il faut commencer par importer les données qu'on veut utiliser à l'aide de la librairie pandas. On choisit ensuite les features qu'on veut enlever en utilisant la fonction drop de pandas. Pour utiliser facilement les données, il faut transformer les données en array de numpy.

La séparation des données en ensemble d'entrainement et de séparation doit être fait par l'utilisateur. Il doit aussi séparer ses labels et features.

Pour faire la régression softmax, il faut commencer par créer un objet SoftmaxRegression() et lui donner en argument les labels et les features. Par exemple:
	sm = SoftmaxRegression(train_labels, train_features)
Les features ne doivent pas être donne en format onehot, se traitement se fera dans la fonction lorsque nécessaire.

Pour entrainer le modèle, donc trouver le w et le b optimale, il faut utiliser SoftmaxRegression.train(eta, lam, nb_iterations). eta est le "learning rate", lam est le terme de régularisation et nb_itérations est le nombre d'itérations qu'on veut utiliser.

Pour ce qui est des k plus proches voisins, la librairie sklearn.neighbors a été utilisé. Voici comment elle fonctionne:
On crée notre objet avec KNeighborsClassifier
	knn = KNeighborsClassifier(n_neighbors=k)
le k choisi sera le nombre de voisins. Ensuite, on entraine le modèle avec knn.fit():
	knn.fit(x_train, labels)
Finalement, on calcule nos prédictions de la façon suivante:
	predictions_knn = [knn.predict([x]) for x in z]
où z est l'ensemble qu'on souhaite prédire.

Les parties laissés en commentaire au début et à la fin du fichier sont simplement des tests fait sur les modèles choisis.


Pour XGBoost, il ne suffit que de run le notebook avec les bonnes importations présentes sur l'environnement. Il est faut nommer les variables X_train, y_train, X_valid et y_valid et X_test pour pouvoir run le code.