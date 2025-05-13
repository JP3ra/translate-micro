#!/bin/bash

echo "🔧 Starting Render-friendly setup..."

# Exit on any error
set -e

# 1. Clone IndicTrans2 if not already present
if [ ! -d "IndicTrans2" ]; then
  echo "📥 Cloning IndicTrans2..."
  git clone https://github.com/AI4Bharat/IndicTrans2.git
else
  echo "📁 IndicTrans2 already exists. Skipping clone."
fi

# 2. Install Python dependencies for IndicTrans2
cd IndicTrans2/huggingface_interface
echo "📦 Installing Python dependencies for IndicTrans2..."
pip install nltk sacremoses pandas regex mock 'transformers>=4.33.2' mosestokenizer
python -c "import nltk; nltk.download('punkt')"
pip install bitsandbytes scipy accelerate datasets sentencepiece
cd ../../

# 3. Clone IndicTransToolkit if not already present
if [ ! -d "IndicTransToolkit" ]; then
  echo "📥 Cloning IndicTransToolkit..."
  git clone https://github.com/VarunGumma/IndicTransToolkit.git
else
  echo "📁 IndicTransToolkit already exists. Skipping clone."
fi

# 4. Install Python dependencies for IndicTransToolkit
cd IndicTransToolkit
pip install --editable ./
cd ..

pip install flask