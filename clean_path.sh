#!/bin/bash

# Create a clean PATH without duplicates and pyenv
NEW_PATH=$(echo $PATH | \
  tr ':' '\n' | \
  grep -v "pyenv" | \
  awk '!seen[$0]++' | \
  tr '\n' ':' | \
  sed 's/:$//')

echo "export PATH=\"$NEW_PATH\"" > ~/.clean_path

echo "Clean PATH has been written to ~/.clean_path"
echo "To apply it, run: source ~/.clean_path"