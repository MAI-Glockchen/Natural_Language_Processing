#!/bin/bash

# ----------------------
# Eval environment setup script
# ----------------------

set -e

GREEN="\033[0;32m"
NC="\033[0m"

echo -e "${GREEN}Starting evaluation environment...${NC}"

# 1. Start Postgres container
echo -e "${GREEN}Step 1: Starting Postgres container...${NC}"
docker compose up -d postgres

# 2. Install dependencies with Poetry
echo -e "${GREEN}Step 2: Installing dependencies with Poetry...${NC}"
poetry install

# 3. Activate Poetry .venv
echo -e "${GREEN}Step 3: Activating Poetry virtual environment...${NC}"
# Get the path to Poetry .venv
VENV_PATH=$(poetry env info -p)
# Activate it
source "$VENV_PATH/bin/activate"

# 4. Restore latest DB backup
echo -e "${GREEN}Step 4: Restoring latest database backup...${NC}"
python -m db-backup-tools restore latest

# 5. Optional: interactive psql shell
echo -e "${GREEN}Step 5: Opening interactive psql shell (Ctrl+D to exit)...${NC}"
docker exec -it citation-postures psql -U postgres -d wiki psql

echo -e "${GREEN}Eval environment ready.${NC}"