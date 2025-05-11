# 📚 Partie 1 : RAG - Retrieval-Augmented Generation

Ce dépôt contient la première partie d'un projet en deux volets, portant sur deux méthodes indépendantes d'optimisation de la génération de texte par modèles de langage.
Cette première partie implémente un système **RAG (Retrieval-Augmented Generation)**, qui permet de répondre à des questions à partir d’un document PDF, en combinant recherche sémantique et génération de texte.

La seconde partie du projet portera sur la méthode **RLHF (Reinforcement Learning from Human Feedback)**, qui sera étudiée séparément, avec d'autres modèles, d'autres données, et une approche distincte.


## 🌟 Objectif de cette partie (RAG)

L’objectif est de permettre à un système de génération automatique de répondre de manière fiable à des questions, en s’appuyant sur une base documentaire externe (ici : un fichier PDF), selon les étapes suivantes :

1. Extraction de texte depuis un PDF.
2. Découpage du texte en passages.
3. Embedding des passages avec un modèle `SentenceTransformer`.
4. Indexation vectorielle via FAISS.
5. Recherche sémantique des passages les plus pertinents.
6. Génération de réponse avec un modèle de type T5.



## 🧰 Fonctionnalités

* 📄 Lecture et traitement automatique de fichiers PDF.
* ✂️ Segmentation du texte en paragraphes courts.
* 🔍 Recherche sémantique à l’aide de vecteurs d’embeddings.
* 🧠 Génération de réponses à partir du contexte sélectionné.



## ⚙️ Utilisation

### 1. Installation des dépendances

```bash
pip install fitz numpy faiss-cpu sentence-transformers transformers
```

Remarque : `fitz` correspond à `PyMuPDF`.



### 2. ⚙️ Exécution du script

Placez votre fichier PDF dans le répertoire du projet, par exemple :
`definitions_generales.pdf`

Ensuite, exécutez le fichier Python principal avec la commande suivante :

```bash
python RAG_Project.py
```

Le script effectuera toutes les étapes nécessaires, de l'extraction du texte jusqu'à la génération de la réponse.


### 3. ✅ Résultat attendu

Le script effectuera les opérations suivantes :

* 📁 **Extrait** le texte depuis le fichier PDF.
* ✂️ **Découpe** le texte en paragraphes courts (\~200 caractères max).
* 🔢 **Crée** des embeddings vectoriels à partir de ces paragraphes.
* 🔎 **Effectue** une recherche sémantique à partir d'une question utilisateur.
* 🧠 **Génère** une réponse à partir du contexte le plus pertinent.

**Lors de l'exécution, les sorties suivantes s'affichent dans la console :**

* ✅ **Texte extrait et découpé** : affichage des premiers paragraphes extraits.
* ✅ **Passages pertinents** : afichage des passages les plus proches de la question.
* ✅ **Contexte utilisé** : affichage du contenu utilisé pour générer la réponse.
* ✅ **Réponse générée** : texte produit par le modèle.



### 🧪 Exemple de question

Lors de l'exécution du script, vous pouvez poser une question comme :

```text
C'est quoi l'innovation ?
```


### 🧱 Composants utilisés

* **Embedding** : `all-MiniLM-L6-v2` (modèle préentrainé de `sentence-transformers`)
* **Indexation** : `FAISS` (recherche vectorielle rapide)
* **Génération** : `google/flan-t5-base` (via `transformers`, pipeline `"text2text-generation"`)

# 📚 Partie 2 : RLHF - Reinforcement Learning from Human Feedback

Ce dépôt contient la seconde partie du projet, axée sur la méthode RLHF (Reinforcement Learning from Human Feedback), une approche d’apprentissage par renforcement guidée par des retours humains. Cette démonstration simplifiée illustre le fonctionnement conceptuel de RLHF sans modèles complexes, en se concentrant sur l’interaction entre un agent, un retour humain, et l’évolution de ses comportements en fonction des récompenses.

## Objectif de cette partie (RLHF)

L’objectif est de simuler comment un agent peut apprendre à produire de meilleures réponses à partir de feedbacks humains simulés, selon les étapes suivantes :

1.Génération d’une réponse aléatoire à une entrée fixe.

2.Évaluation de cette réponse par une fonction de feedback humain.

3.Récompense attribuée selon la qualité perçue de la réponse.

4.Affichage de la réponse et de sa récompense à chaque itération.

## 🧰 Fonctionnalités
🧠 Simulation d’un modèle de réponse simple
👤 Évaluation manuelle simulée des réponses
🔁 Boucle d’entraînement avec retour de récompense
📊 Affichage des performances par itération

##  Exécution du script
 
            python RLHF_Project.py

## ✅ Résultat attendu    

Le script exécute 10 itérations d’un entraînement simulé :

📥 Le "modèle" reçoit une entrée texte (ex. "Salut")

🤖 Il génère une réponse aléatoire parmi des phrases définies

🧑 Le système simule un retour humain pour chaque réponse

📈 Chaque réponse est accompagnée de sa récompense (0 ou 1)

 Sortie exemple :

python-repl
Copier
Modifier
Epoch 1: Réponse: Très bien. | Récompense: 1
Epoch 2: Réponse: Je ne sais pas. | Récompense: 0

## Exemple d’entrée
Aucune interaction requise : l’entrée simulée est "Salut".

## 🧱 Composants utilisés
Modèle simulé : fonction model() qui génère une réponse aléatoire

Feedback humain : fonction human_feedback() qui attribue une récompense

Boucle d'entraînement : train_model() répète l’évaluation 10 fois