# Relatório de Análise de PII (Informações Pessoais Identificáveis)

**Data da Análise**: 19 de outubro de 2025  
**Repositório**: Stanford-RNA-3D-Folding  
**Escopo**: Análise completa excluindo diretórios `.venv`  

## Resumo Executivo

Este relatório apresenta os resultados da varredura automatizada por informações pessoais identificáveis (PII) no repositório. A análise focou em identificar emails, chaves privadas, tokens, CPFs, telefones e outros dados sensíveis.

### Principais Achados

- **200+ emails encontrados**: Principalmente em arquivos PDB_RNA/*.cif (dados públicos do PDB)
- **0 chaves privadas reais** no código do projeto
- **0 credenciais AWS/tokens** no código principal
- **Email do autor** presente em documentação e notebooks (intencional)
- **Números suspeitos** apenas em datasets CSV científicos (coordenadas/IDs)

## Classificação de Achados por Prioridade

### 🔴 ALTA PRIORIDADE (Ação Imediata Recomendada)

#### 1. Diretórios de Ambiente Virtual
**Localização**: `.venv/` e `stanford_rna3d/.venv/`
**Problema**: Estes diretórios contêm packages de terceiros e assets que não devem estar no repositório
**Ação**: Remover do repositório e adicionar ao .gitignore

### 🟡 MÉDIA PRIORIDADE (Revisão Recomendada)

#### 2. Emails em Arquivos PDB
**Localização**: `stanford_rna3d/data/raw/PDB_RNA/*.cif`
**Exemplos encontrados**:
- jfh21@columbia.edu
- bwiedenheft@gmail.com  
- wahc@stanford.edu
- rhiju@stanford.edu
- anna.pyle@yale.edu
- alexey.amunts@gmail.com

**Contexto**: Estes arquivos contêm metadados públicos do Protein Data Bank (PDB) incluindo informações de contato dos autores dos experimentos.
**Risco**: Baixo - são dados já públicos no PDB
**Recomendação**: Manter + documentar origem, ou redigir se política de privacidade exigir

#### 3. Email do Autor do Projeto
**Localização**: Documentação, notebooks, LICENSE
**Email**: mauro.risonho@gmail.com
**Contexto**: Atribuição de autoria inserida intencionalmente
**Risco**: Baixo - PII voluntário para atribuição
**Recomendação**: Manter se autoria pública for desejada

### 🟢 BAIXA PRIORIDADE (Informativo)

#### 4. Dados de Sequência RNA
**Localização**: `stanford_rna3d/data/raw/train_labels.csv`, `validation_labels.csv`
**Conteúdo**: IDs de sequência (ex: R1128_37), coordenadas numéricas
**Risco**: Muito baixo - dados científicos, não PII pessoal

## Detalhamento Técnico

### Padrões Pesquisados
- ✅ Emails: `[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}`
- ✅ Chaves privadas: `-----BEGIN.*PRIVATE KEY-----`
- ✅ AWS Access Keys: `AKIA[0-9A-Z]{16}`, etc.
- ✅ Tokens/senhas: `password|secret|api_key|token|credential`
- ✅ CPF: `\d{3}\.\d{3}\.\d{3}-\d{2}`
- ✅ Telefones: padrões BR e internacionais

### Falsos Positivos Identificados
- Números em datasets CSV: coordenadas X,Y,Z e IDs de sequência
- Assets em .venv: JS minificados, METADATA de packages
- Dados de teste em site-packages

## Ações Recomendadas

### Ação 1: Limpeza de Ambiente ⚡ EXECUTAR AGORA
```bash
# Adicionar ao .gitignore
echo ".venv/" >> .gitignore
echo "stanford_rna3d/.venv/" >> .gitignore

# Remover do repositório (se versionado)
git rm -r --cached .venv/ stanford_rna3d/.venv/ 2>/dev/null || true
git add .gitignore
git commit -m "Remove virtual environments from repository and update .gitignore"
```

### Ação 2: Documentação dos Dados PDB
- Adicionar nota em README sobre origem dos arquivos PDB_RNA/*.cif
- Explicar que contêm metadados públicos do PDB incluindo contatos de autores

### Ação 3: Script de Monitoramento (Opcional)
- Criar script Python para varredura PII automatizada
- Configurar para executar antes de commits

## Nenhuma Credencial Crítica Encontrada ✅

A análise **não encontrou**:
- Chaves privadas SSH/TLS reais no código
- Tokens de API válidos 
- Senhas em texto claro
- Credenciais AWS reais
- Dados pessoais sensíveis (CPF, cartões) em código

## Conclusão

O repositório está em **boa condição de segurança**. As principais ações necessárias são:
1. Remover diretórios `.venv` (housekeeping)
2. Documentar origem dos dados PDB (transparência)

Nenhuma ação crítica de segurança é necessária imediatamente.

---
**Relatório gerado automaticamente** | **Revisar periodicamente antes de releases públicos**