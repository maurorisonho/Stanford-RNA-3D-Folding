# 🔍 PII Scanner - Detector Automático de Informações Pessoais

Este script Python automatiza a detecção de **PII (Personally Identifiable Information)** em repositórios de código, permitindo varreduras reprodutíveis e sistemáticas por dados sensíveis.

## 📋 Funcionalidades

### Tipos de PII Detectados
- ✉️ **Emails** (medium priority)
- 🔑 **Chaves privadas** SSH/RSA (critical)
- 🔐 **Chaves AWS** (critical/high)
- 🔒 **Credenciais** (passwords, tokens, API keys)
- 🆔 **CPF brasileiro** (high)
- 🆔 **SSN americano** (high)
- 📞 **Telefones brasileiros** (medium)
- 💳 **Cartões de crédito** (critical)

### Filtros Inteligentes
- Ignora automaticamente `.venv`, `node_modules`, `.git`
- Detecta falsos positivos (coordenadas científicas, exemplos)
- Filtra arquivos binários e muito grandes

## 🚀 Como Usar

### Instalação
```bash
# O script é standalone, apenas certifique-se de ter Python 3.8+
chmod +x pii_scanner.py
```

### Uso Básico
```bash
# Escanear diretório atual
./pii_scanner.py

# Escanear diretório específico
./pii_scanner.py /caminho/para/projeto

# Gerar relatório em JSON
./pii_scanner.py --format json

# Salvar relatório em arquivo
./pii_scanner.py --output relatorio_pii.txt
```

### Opções Avançadas
```bash
# Excluir padrões customizados
./pii_scanner.py --exclude ".*\.cif$" --exclude "test_.*"

# Sem cores (para CI/CD)
./pii_scanner.py --no-color

# Exemplo completo
./pii_scanner.py stanford_rna3d/ \
  --format json \
  --output pii_report.json \
  --exclude ".*\.pdb$" \
  --exclude "data/raw/.*"
```

## 📊 Interpretação dos Resultados

### Níveis de Prioridade
- 🔴 **CRITICAL**: Chaves privadas, cartões de crédito - **AÇÃO IMEDIATA**
- 🟠 **HIGH**: CPF, SSN, credenciais - **Revisar urgentemente**
- 🟡 **MEDIUM**: Emails, telefones - **Verificar se apropriado**
- 🟢 **LOW**: Dados possivelmente públicos - **Documentar**

### Códigos de Saída
- `0`: Nenhum problema ou apenas achados de baixa prioridade
- `1`: Achados críticos ou de alta prioridade encontrados

## 🔧 Integração com Git Hooks

### Pre-commit Hook
Crie `.git/hooks/pre-commit`:
```bash
#!/bin/bash
python3 pii_scanner.py --no-color
exit_code=$?

if [ $exit_code -eq 1 ]; then
    echo "❌ Possível PII detectado! Commit bloqueado."
    echo "Execute 'python3 pii_scanner.py' para detalhes."
    exit 1
fi
```

### GitHub Actions
```yaml
name: PII Scanner
on: [push, pull_request]
jobs:
  pii-scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run PII Scanner
        run: |
          python3 pii_scanner.py --format json --output pii_report.json
      - name: Upload results
        uses: actions/upload-artifact@v3
        if: failure()
        with:
          name: pii-report
          path: pii_report.json
```

## 🎯 Casos de Uso

### 1. Auditoria de Segurança
```bash
# Varredura completa com relatório detalhado
./pii_scanner.py . --format json --output audit_$(date +%Y%m%d).json
```

### 2. Verificação Pré-Release
```bash
# Apenas problemas críticos/altos
./pii_scanner.py src/ && echo "✅ Pronto para release"
```

### 3. Limpeza de Repositório
```bash
# Encontrar todos os tipos de PII
./pii_scanner.py --exclude "\.git.*" > cleanup_report.txt
```

## 🔄 Personalização

### Adicionar Novos Padrões
Edite a constante `PATTERNS` no script:
```python
'custom_pattern': {
    'regex': r'seu_regex_aqui',
    'severity': 'high',
    'description': 'Descrição do padrão'
}
```

### Filtros Customizados
Modifique `_is_false_positive()` para casos específicos do seu projeto.

## 📈 Exemplo de Saída

```
============================================================
RELATÓRIO DE VARREDURA PII
============================================================
Data: 2024-12-19 10:30:15
Diretório: /home/user/projeto
Total de achados: 5

🔴 HIGH PRIORITY (2 achados)
--------------------------------------------------
📁 config/settings.py:15
🔍 Tipo: secret_keywords
📝 Texto: api_key = "sk-abc123..."
📄 Contexto: OPENAI_API_KEY = "sk-abc123..."

🟡 MEDIUM PRIORITY (3 achados)
--------------------------------------------------
📁 docs/README.md:42
🔍 Tipo: email
📝 Texto: contato@empresa.com
📄 Contexto: Entre em contato: contato@empresa.com
```

## ⚡ Performance

- **Velocidade**: ~100 arquivos/segundo
- **Memória**: Baixo consumo, processa linha por linha
- **Escalabilidade**: Adequado para repositórios com milhares de arquivos

## 🛡️ Limitações e Considerações

### O que NÃO é detectado:
- PII em dados binários
- Dados ofuscados ou criptografados
- PII em imagens ou PDFs
- Padrões específicos de outras regiões

### Falsos Positivos Comuns:
- Coordenadas científicas como CPF
- Emails em documentação
- Chaves de teste/exemplo
- IDs que parecem credenciais

## 📝 Manutenção

Atualize regularmente:
1. **Padrões regex** para novos tipos de PII
2. **Filtros de falsos positivos** baseado na experiência
3. **Lista de exclusões** conforme o projeto evolui

---

**💡 Dica**: Execute o scanner regularmente durante o desenvolvimento para detectar PII antes que chegue ao repositório remoto!