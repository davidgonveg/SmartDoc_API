#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🔧 Setting up SmartDoc Test Environment${NC}"
echo "=" * 50

# Check if we're in the right directory
if [ ! -f "requirements.txt" ]; then
    echo -e "${RED}❌ Error: requirements.txt not found. Are you in the agent-api directory?${NC}"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo -e "${BLUE}🐍 Python version: $PYTHON_VERSION${NC}"

if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 11) else 1)"; then
    echo -e "${YELLOW}⚠️  Warning: Python 3.11+ recommended, you have $PYTHON_VERSION${NC}"
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}📦 Creating virtual environment...${NC}"
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo -e "${RED}❌ Failed to create virtual environment${NC}"
        exit 1
    fi
    echo -e "${GREEN}✅ Virtual environment created${NC}"
else
    echo -e "${GREEN}✅ Virtual environment already exists${NC}"
fi

# Activate virtual environment
echo -e "${YELLOW}🔄 Activating virtual environment...${NC}"
source venv/bin/activate

# Verify activation
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo -e "${GREEN}✅ Virtual environment activated: $VIRTUAL_ENV${NC}"
else
    echo -e "${RED}❌ Failed to activate virtual environment${NC}"
    exit 1
fi

# Upgrade pip
echo -e "${YELLOW}📦 Upgrading pip...${NC}"
python3 -m pip install --upgrade pip --quiet

# Install main dependencies
echo -e "${YELLOW}📦 Installing main dependencies...${NC}"
python3 -m pip install -r requirements.txt --quiet
if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Failed to install main dependencies${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Main dependencies installed${NC}"

# Install testing dependencies
echo -e "${YELLOW}🧪 Installing testing dependencies...${NC}"
python3 -m pip install pytest==7.4.3 pytest-asyncio==0.21.1 pytest-cov==4.1.0 pytest-mock==3.12.0 responses==0.23.3 aioresponses==0.7.4 factory-boy==3.3.0 faker==20.1.0 --quiet
if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Failed to install testing dependencies${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Testing dependencies installed${NC}"

# Create scripts directory if it doesn't exist
mkdir -p scripts

# Make scripts executable
chmod +x scripts/*.py scripts/*.sh 2>/dev/null || true

# Verify imports
echo -e "${YELLOW}🔍 Verifying imports...${NC}"
python3 scripts/verify_imports.py

if [ $? -eq 0 ]; then
    echo -e "${GREEN}🎉 Test environment setup complete!${NC}"
    echo ""
    echo -e "${BLUE}Next steps:${NC}"
    echo "1. Keep virtual environment activated: source venv/bin/activate"
    echo "2. Run import verification anytime: python3 scripts/verify_imports.py"
    echo "3. Ready to create and run tests!"
    echo ""
    echo -e "${YELLOW}Current environment:${NC}"
    echo "- Python: $(python3 --version)"
    echo "- Pip: $(python3 -m pip --version)"
    echo "- Virtual env: $VIRTUAL_ENV"
else
    echo -e "${RED}⚠️  Some imports failed. Check the output above for details.${NC}"
    exit 1
fi