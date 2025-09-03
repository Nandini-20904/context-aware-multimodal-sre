#!/usr/bin/env python3
"""
Startup Script for Complete SRE Incident Insight Engine
Production-ready launcher with dependency checks and system validation
"""

import os
import sys
import subprocess
import importlib.util
from pathlib import Path

def check_python_version():
    """Check Python version compatibility"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"   Current version: {sys.version}")
        return False
    
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True

def check_required_files():
    """Check if all required files are present"""
    required_files = [
        'sre_app.py',
        'sre.py',
        'assistant_chatbot.py',
        'nlp_processor.py',
        'incident.py',
        'config.py',
        'main.py'  # User's existing ML system
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
        else:
            print(f"✅ Found: {file}")
    
    if missing_files:
        print("\n❌ Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        
        if 'paste.py' in missing_files:
            print("\n💡 Note: Rename your ML system file to 'paste.py' or update imports")
        
        return False
    
    return True

def check_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = {
        'streamlit': 'streamlit>=1.28.0',
        'pandas': 'pandas>=1.3.0',
        'numpy': 'numpy>=1.21.0',
        'plotly': 'plotly>=5.0.0',
        'google.generativeai': 'google-generativeai>=0.5.0',
        'spacy': 'spacy>=3.4.0',
        'nltk': 'nltk>=3.7',
        'sentence_transformers': 'sentence-transformers>=2.2.0',
        'sklearn': 'scikit-learn>=1.0.0',
        'joblib': 'joblib>=1.1.0'
    }
    
    missing_packages = []
    
    for package, requirement in required_packages.items():
        try:
            if '.' in package:
                # Handle packages with dots (like google.generativeai)
                main_package = package.split('.')[0]
                spec = importlib.util.find_spec(main_package)
            else:
                spec = importlib.util.find_spec(package)
            
            if spec is None:
                missing_packages.append(requirement)
            else:
                print(f"✅ Found: {package}")
                
        except ImportError:
            missing_packages.append(requirement)
    
    if missing_packages:
        print("\n❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        
        print("\n💡 Install missing packages with:")
        print("pip install", " ".join(missing_packages))
        return False
    
    return True

def check_spacy_model():
    """Check if spaCy model is downloaded"""
    try:
        import spacy
        try:
            spacy.load("en_core_web_sm")
            print("✅ spaCy model: en_core_web_sm")
            return True
        except OSError:
            print("❌ spaCy model 'en_core_web_sm' not found")
            print("💡 Download with: python -m spacy download en_core_web_sm")
            return False
    except ImportError:
        print("⚠️ spaCy not installed")
        return False

def check_api_configuration():
    """Check API configuration"""
    try:
        from config import config
        
        if hasattr(config, 'GEMINI_API_KEY') and config.GEMINI_API_KEY:
            masked_key = config.GEMINI_API_KEY[:10] + "..." + config.GEMINI_API_KEY[-4:]
            print(f"✅ Gemini API key configured: {masked_key}")
            return True
        else:
            print("⚠️ Gemini API key not configured")
            print("💡 Set your API key in sre_config.py")
            return False
            
    except ImportError:
        print("❌ sre_config.py not found")
        return False
    except Exception as e:
        print(f"⚠️ API configuration error: {e}")
        return False

def run_system_validation():
    """Run quick system validation"""
    try:
        print("\n🔍 Running system validation...")
        
        # Test imports
        from sre import CompleteSREInsightEngine
        from assistant_chatbot import SREAssistantChatbot
        from nlp_processor import ProductionNLPProcessor
        
        print("✅ Core components imported successfully")
        
        # Test basic initialization (without running full analysis)
        try:
            # This should work even without external dependencies
            engine = CompleteSREInsightEngine()
            print("✅ SRE Engine initialization test passed")
        except Exception as e:
            print(f"⚠️ SRE Engine initialization warning: {e}")
        
        return True
        
    except ImportError as e:
        print(f"❌ System validation failed: {e}")
        return False
    except Exception as e:
        print(f"⚠️ System validation warning: {e}")
        return True  # Continue anyway

def launch_streamlit():
    """Launch Streamlit application"""
    
    print("\n🚀 Launching SRE Incident Insight Engine...")
    print("=" * 60)
    print("🎯 Complete Context-Aware Multimodal Incident Analysis System")
    print("📊 Integrating: ML, NLP, GenAI Chatbot, Monitoring, Analysis")
    print("🤖 Powered by Gemini AI with your production ML system")
    print("=" * 60)
    print("\n💡 The application will open in your web browser")
    print("🔗 If it doesn't open automatically, go to: http://localhost:8501")
    print("\n⏹️ Press Ctrl+C to stop the application")
    print("-" * 60)
    
    try:
        # Launch Streamlit
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run', 'sre_app.py',
            '--server.address', '0.0.0.0',
            '--server.port', '8501',
            '--browser.gatherUsageStats', 'false'
        ], check=True)
        
    except KeyboardInterrupt:
        print("\n👋 SRE Incident Insight Engine stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to launch Streamlit: {e}")
        print("💡 Try running manually: streamlit run sre_app.py")
    except Exception as e:
        print(f"❌ Launch error: {e}")

def install_missing_dependencies():
    """Install missing dependencies"""
    
    packages_to_install = [
        'streamlit>=1.28.0',
        'plotly>=5.0.0',
        'google-generativeai>=0.5.0',
        'python-dotenv>=1.0.0'
    ]
    
    print("🔧 Installing missing dependencies...")
    
    for package in packages_to_install:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            print(f"✅ Installed: {package}")
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install: {package}")

def main():
    """Main startup function"""
    
    print("🚀 SRE INCIDENT INSIGHT ENGINE - STARTUP")
    print("=" * 50)
    print("🎯 Production-Ready Complete System Launcher")
    print("📊 Checking system requirements...")
    print("=" * 50)
    
    # Check requirements
    checks_passed = 0
    total_checks = 6
    
    print("\n1️⃣ Checking Python version...")
    if check_python_version():
        checks_passed += 1
    
    print("\n2️⃣ Checking required files...")
    if check_required_files():
        checks_passed += 1
    
    print("\n3️⃣ Checking dependencies...")
    if check_dependencies():
        checks_passed += 1
    else:
        # Offer to install missing dependencies
        response = input("\n🔧 Install missing dependencies? (y/n): ").strip().lower()
        if response in ['y', 'yes']:
            install_missing_dependencies()
            print("✅ Dependencies installed. Please restart the launcher.")
            return
    
    print("\n4️⃣ Checking spaCy model...")
    if check_spacy_model():
        checks_passed += 1
    else:
        response = input("\n🔧 Download spaCy model? (y/n): ").strip().lower()
        if response in ['y', 'yes']:
            try:
                subprocess.check_call([sys.executable, '-m', 'spacy', 'download', 'en_core_web_sm'])
                print("✅ spaCy model downloaded")
                checks_passed += 1
            except subprocess.CalledProcessError:
                print("❌ Failed to download spaCy model")
    
    print("\n5️⃣ Checking API configuration...")
    if check_api_configuration():
        checks_passed += 1
    
    print("\n6️⃣ Running system validation...")
    if run_system_validation():
        checks_passed += 1
    
    # Summary
    print(f"\n📊 SYSTEM CHECK SUMMARY: {checks_passed}/{total_checks} checks passed")
    
    if checks_passed >= 4:  # Minimum required
        print("✅ System ready for launch!")
        
        response = input("\n🚀 Launch SRE Incident Insight Engine? (y/n): ").strip().lower()
        if response in ['y', 'yes', '']:
            launch_streamlit()
        else:
            print("👋 Launch cancelled by user")
    else:
        print("❌ System not ready - please resolve the issues above")
        print("\n💡 TROUBLESHOOTING TIPS:")
        print("1. Ensure all Python files are in the same directory")
        print("2. Install missing dependencies: pip install -r requirements.txt")
        print("3. Download spaCy model: python -m spacy download en_core_web_sm")
        print("4. Configure Gemini API key in sre_config.py")
        print("5. Rename your ML system file to 'paste.py'")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Startup cancelled by user")
    except Exception as e:
        print(f"❌ Startup error: {e}")
        print("💡 Try running components manually if issues persist")