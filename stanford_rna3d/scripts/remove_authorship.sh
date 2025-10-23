#!/bin/bash

# Script para remover informações de autoria e licença MIT incorretamente adicionadas
# aos arquivos do projeto Stanford RNA 3D Folding

set -e

PROJECT_ROOT="/home/test/Downloads/Github/kaggle/Stanford-RNA-3D-Folding"
cd "$PROJECT_ROOT"

echo "Iniciando remoção de informações de autoria e licença MIT..."

# Encontrar todos os arquivos que contêm as informações de autoria
echo "Identificando arquivos com informações de autoria..."

# Arquivos Python, Markdown, Shell scripts, etc. (excluindo backups e .venv)
FILES_TO_CLEAN=$(find . -type f \( -name "*.py" -o -name "*.md" -o -name "*.sh" -o -name "*.ipynb" \) \
    ! -path "./.venv/*" \
    ! -path "./backup*/*" \
    ! -name "*.backup" \
    ! -name "*.authorship_backup" \
    ! -name "clean_project_text.sh" \
    ! -name "remove_authorship.sh" \
    -exec grep -l "Mauro Risonho de Paula Assumpção\|MIT License.*2025" {} \;)

echo "Arquivos encontrados para limpeza:"
echo "$FILES_TO_CLEAN"

# Função para limpar um arquivo
clean_authorship_info() {
    local file="$1"
    echo "Limpando: $file"
    
    # Criar backup
    cp "$file" "${file}.authorship_backup"
    
    # Remover seções de autoria e licença MIT
    # Usando sed para remover linhas específicas de autoria
    
    # Para arquivos Python (.py)
    if [[ "$file" =~ \.py$ ]]; then
        # Remover linhas de autoria em comentários Python
        sed -i '/^# Author: Mauro Risonho de Paula Assumpção/d' "$file"
        sed -i '/^# Created:/d' "$file"
        sed -i '/^# License: MIT License/d' "$file"
        sed -i '/^# Kaggle Competition:/d' "$file"
        
        # Remover bloco completo de licença MIT em comentários
        sed -i '/^# MIT License$/,/^# THE SOFTWARE\.$/d' "$file"
        sed -i '/^# ---$/d' "$file"
    fi
    
    # Para arquivos Markdown (.md)
    if [[ "$file" =~ \.md$ ]]; then
        # Remover linhas de autoria em Markdown
        sed -i '/^\*\*Author\*\*: Mauro Risonho de Paula Assumpção/d' "$file"
        sed -i '/^\*\*Created\*\*:/d' "$file"
        sed -i '/^\*\*License\*\*: MIT License/d' "$file"
        sed -i '/^\*\*Kaggle Competition\*\*:/d' "$file"
        
        # Remover bloco de licença MIT completo
        sed -i '/^MIT License$/,/^THE SOFTWARE\.$/d' "$file"
        sed -i '/^---$/d' "$file"
        
        # Remover linha de copyright
        sed -i '/^Copyright (c) 2025 Mauro Risonho de Paula Assumpção/d' "$file"
    fi
    
    # Para arquivos Jupyter Notebook (.ipynb)
    if [[ "$file" =~ \.ipynb$ ]]; then
        # Para notebooks, usar uma abordagem mais cuidadosa
        # Remover linhas que contêm informações de autoria
        sed -i '/"Author": Mauro Risonho de Paula Assumpção/d' "$file"
        sed -i '/"Created":/d' "$file" 
        sed -i '/"License": MIT License/d' "$file"
        sed -i '/"Kaggle Competition":/d' "$file"
        
        # Remover blocos de licença MIT
        sed -i '/MIT License/,/THE SOFTWARE\./d' "$file"
        sed -i '/Copyright (c) 2025 Mauro Risonho de Paula Assumpção/d' "$file"
    fi
    
    # Para arquivos Shell (.sh)
    if [[ "$file" =~ \.sh$ ]]; then
        sed -i '/^# Author: Mauro Risonho de Paula Assumpção/d' "$file"
        sed -i '/^# Created:/d' "$file"
        sed -i '/^# License: MIT License/d' "$file"
        sed -i '/^# Kaggle Competition:/d' "$file"
        
        # Remover bloco de licença MIT
        sed -i '/^# MIT License$/,/^# THE SOFTWARE\.$/d' "$file"
    fi
    
    echo "  ✓ $file limpo"
}

# Processar cada arquivo encontrado
if [[ -n "$FILES_TO_CLEAN" ]]; then
    while IFS= read -r file; do
        if [[ -f "$file" ]]; then
            clean_authorship_info "$file"
        fi
    done <<< "$FILES_TO_CLEAN"
else
    echo "Nenhum arquivo encontrado com informações de autoria para limpar."
fi

echo ""
echo "LIMPEZA DE AUTORIA CONCLUÍDA!"
echo ""
echo "Resumo das alterações:"
echo "- Informações de autoria removidas"
echo "- Blocos de licença MIT removidos" 
echo "- Backups criados com extensão .authorship_backup"
echo ""
echo "Para reverter as mudanças:"
echo "for f in *.authorship_backup; do mv \"\$f\" \"\${f%.authorship_backup}\"; done"
echo ""
echo "NOTA: O arquivo LICENSE principal foi mantido intacto."
