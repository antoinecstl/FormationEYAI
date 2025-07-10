# 🤖 Formation RAG - EY AI

**Formation complète sur le RAG (Retrieval-Augmented Generation) dispensée sur Google Colab**

Une formation pratique pour apprendre à construire un assistant IA capable d'interroger vos documents PDF, entièrement réalisable dans votre navigateur grâce à Google Colab.

## 🎯 Objectifs Pédagogiques

À la fin de cette formation, vous serez capable de :
- ✅ Comprendre les composants essentiels d'un système RAG
- ✅ Manipuler du code Python sur Google Colab
- ✅ Installer et utiliser **Ollama** pour faire tourner des modèles LLM localement
- ✅ Tester un prototype RAG sur un document PDF
- ✅ Déployer une application RAG complète avec interface web

## 📚 Structure de la Formation

### 📓 Partie 1 : Concepts Fondamentaux du RAG
**Fichier:** `Formation-RAG_Partie-1.ipynb`

**Séquences couvertes :**
- 🔍 **Bases du RAG** : Embeddings & similarité sémantique
- 📐 **Chunking** : Découpage intelligent de documents PDF
- 📊 **Index vectoriel FAISS** : Recherche rapide dans les documents
- 🧮 **Algorithme MMR** : Sélection de passages pertinents et diversifiés  
- 🧑‍🎤 **Prompt engineering** : Construction de prompts efficaces
- 🏗️ **Prototype RAG** : Assemblage d'un système complet

### 📓 Partie 2 : Déploiement et Mise en Production
**Fichier:** `Formation-RAG_Partie-2.ipynb`

**Séquences couvertes :**
- 🛠️ **Installation d'Ollama** : Configuration de l'environnement local
- 🔗 **Bootstrap Colab** : Connexion Drive et clonage du projet
- 🏎️ **Déploiement** : Lancement de l'application Streamlit avec LocalTunnel

## 🚀 Comment Suivre la Formation

### 📋 Prérequis
- Un compte Google (pour accéder à Google Colab)
- Un navigateur web moderne
- Connexion Internet stable

### 🎓 Instructions de Formation

#### Étape 1 : Accéder aux Notebooks
1. Ouvrir Google Colab : [colab.research.google.com](https://colab.research.google.com)
2. Cliquer sur "GitHub" dans l'onglet d'ouverture
3. Saisir l'URL du repository : `https://github.com/antoinecstl/FormationEYAI`
4. Sélectionner le notebook désiré

#### Étape 2 : Commencer par la Partie 1
1. Ouvrir `Formation-RAG_Partie-1.ipynb`
2. Exécuter les cellules **dans l'ordre** (Shift + Enter)
3. Suivre les explications pédagogiques dans chaque cellule
4. Expérimenter avec les exemples fournis

#### Étape 3 : Continuer avec la Partie 2  
1. Ouvrir `Formation-RAG_Partie-2.ipynb`
2. Suivre les instructions d'installation d'Ollama
3. Déployer l'application complète
4. Tester l'interface web générée

## 🛠️ Technologies Utilisées

### 🤖 Modèles IA
- **llama3.2:3B** - Modèle de génération de texte (Meta)
- **nomic-embed-text** - Modèle d'embeddings
- **bge-m3** - Modèle d'embeddings alternatif

### 🔧 Outils et Bibliothèques
- **Ollama** - Exécution locale de modèles LLM
- **FAISS** - Recherche vectorielle rapide
- **Streamlit** - Interface web interactive
- **LocalTunnel** - Accès web depuis Colab
- **LangChain** - Découpage intelligent de documents
- **PyPDF2** - Extraction de texte PDF

## 📖 Concepts Clés Abordés

### 🧠 RAG (Retrieval-Augmented Generation)
Technique combinant :
- **Retrieval** : Recherche d'informations pertinentes dans une base de documents
- **Augmentation** : Enrichissement du prompt avec le contexte trouvé
- **Generation** : Production d'une réponse par le modèle LLM

### 📊 Embeddings
Transformation de texte en vecteurs numériques capturant le sens sémantique, permettant :
- Calcul de similarité entre textes
- Recherche sémantique dans les documents
- Indexation vectorielle efficace

### 🧮 MMR (Maximal Marginal Relevance)
Algorithme équilibrant :
- **Pertinence** : Similarité avec la question
- **Diversité** : Variété des passages sélectionnés

## 🔍 Document d'Exemple

La formation utilise `Anonymized_Rapport.pdf` comme document de test pour :
- Démontrer l'extraction et le chunking de PDF
- Tester les requêtes RAG
- Valider la qualité des réponses générées

## 💡 Conseils pour la Formation

### ✏️ Expérimentation Encouragée
- Modifiez les questions dans les exemples
- Testez différents paramètres (temperature, top_k)
- Uploadez vos propres documents PDF

### 🐛 Résolution de Problèmes Courants

**Problème** : Ollama ne s'installe pas
**Solution** : Relancer la cellule d'installation, vérifier la connexion Internet

**Problème** : Modèles ne se téléchargent pas
**Solution** : Vérifier l'espace disque disponible dans Colab

**Problème** : LocalTunnel ne fonctionne pas
**Solution** : Utiliser l'IP affichée comme mot de passe dans le tunnel

### 📱 Accès à l'Application
Une fois la Partie 2 terminée, vous obtiendrez :
- Un lien LocalTunnel pour accéder à l'application
- Une interface web Streamlit interactive
- La possibilité d'uploader vos propres PDF

## 🎯 Cas d'Usage

Cette formation vous prépare à construire des assistants IA pour :
- 📋 Analyse de rapports d'entreprise
- 📚 Recherche dans la documentation technique
- 🔍 Extraction d'informations de contrats
- 📊 Synthèse de documents réglementaires

## 🤝 Support

Pour toute question pendant la formation :
- Consultez les explications détaillées dans chaque cellule
- Vérifiez les messages d'erreur dans la console
- Relancez les cellules d'installation si nécessaire

---

**🚀 Commencez maintenant : Ouvrez `Formation-RAG_Partie-1.ipynb` dans Google Colab !**
