"""
Script d'installation pour Google Colab
Installe Ollama et configure l'environnement pour la formation RAG
"""

import subprocess
import time
import os
import sys

def install_ollama():
    """Installation d'Ollama sur Google Colab"""
    print("🚀 Installation d'Ollama...")
    
    try:
        # Téléchargement et installation d'Ollama
        subprocess.run([
            "curl", "-fsSL", "https://ollama.ai/install.sh", "-o", "/tmp/install.sh"
        ], check=True)
        
        subprocess.run(["chmod", "+x", "/tmp/install.sh"], check=True)
        subprocess.run(["/tmp/install.sh"], check=True)
        
        print("✅ Ollama installé avec succès")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Erreur lors de l'installation d'Ollama: {e}")
        return False

def start_ollama_service():
    """Démarre le service Ollama en arrière-plan"""
    print("🔄 Démarrage du service Ollama...")
    
    try:
        # Démarre Ollama en arrière-plan
        process = subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid
        )
        
        # Attendre que le service démarre
        time.sleep(10)
        
        # Vérifier si le service fonctionne
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            print("✅ Service Ollama démarré")
            return process
        else:
            print("❌ Échec du démarrage du service")
            return None
            
    except Exception as e:
        print(f"❌ Erreur lors du démarrage: {e}")
        return None

def pull_models():
    """Télécharge les modèles nécessaires"""
    models = ["llama3.2:3b", "nomic-embed-text:latest"]
    
    for model in models:
        print(f"📥 Téléchargement du modèle {model}...")
        try:
            result = subprocess.run(
                ["ollama", "pull", model],
                capture_output=True,
                text=True,
                timeout=600  # 10 minutes max par modèle
            )
            
            if result.returncode == 0:
                print(f"✅ Modèle {model} téléchargé")
            else:
                print(f"❌ Erreur pour {model}: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            print(f"⏱️ Timeout pour le modèle {model}")
        except Exception as e:
            print(f"❌ Erreur pour {model}: {e}")

def install_python_dependencies():
    """Installe les dépendances Python"""
    print("📦 Installation des dépendances Python...")
    
    dependencies = [
        "streamlit>=1.28.0",
        "faiss-cpu>=1.7.4",
        "numpy>=1.24.0",
        "PyPDF2>=3.0.1",
        "langchain>=0.1.0",
        "scikit-learn>=1.3.0",
        "ollama-python>=0.1.0"
    ]
    
    for dep in dependencies:
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", dep], check=True)
            print(f"✅ {dep.split('>=')[0]} installé")
        except subprocess.CalledProcessError as e:
            print(f"❌ Erreur pour {dep}: {e}")

def setup_colab_environment():
    """Configuration complète de l'environnement Colab"""
    print("🎯 Configuration de l'environnement Google Colab pour RAG PDF Chat")
    print("=" * 60)
    
    # 1. Installation des dépendances Python
    install_python_dependencies()
    
    # 2. Installation d'Ollama
    if not install_ollama():
        print("❌ Impossible de continuer sans Ollama")
        return False
    
    # 3. Démarrage du service
    ollama_process = start_ollama_service()
    if not ollama_process:
        print("❌ Impossible de démarrer le service Ollama")
        return False
    
    # 4. Téléchargement des modèles
    pull_models()
    
    print("=" * 60)
    print("🎉 Configuration terminée !")
    print("📝 Instructions:")
    print("   1. Exécutez les cellules du notebook dans l'ordre")
    print("   2. Uploadez vos fichiers PDF via l'interface")
    print("   3. Posez vos questions au chatbot")
    print("⚠️  Note: Le service Ollama tourne en arrière-plan")
    
    return True

if __name__ == "__main__":
    setup_colab_environment()
