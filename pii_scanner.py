#!/usr/bin/env python3
"""
Script de Varredura Automática de PII (Informações Pessoais Identificáveis)
Stanford RNA 3D Folding Project

Este script executa uma varredura automatizada por dados sensíveis no repositório.
Pode ser usado manualmente ou integrado em hooks de pre-commit.

Author: Mauro Risonho de Paula Assumpção <mauro.risonho@gmail.com>
License: MIT
"""

import os
import re
import sys
import argparse
import json
from pathlib import Path
from typing import List, Dict, Tuple, Set
from dataclasses import dataclass
from datetime import datetime

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


@dataclass
class PIIFinding:
    """Representa um achado de PII."""
    file_path: str
    line_number: int
    pattern_type: str
    matched_text: str
    context_line: str
    severity: str


class PIIScanner:
    """Scanner principal para identificação de PII."""
    
    # Padrões de busca por tipo
    PATTERNS = {
        'email': {
            'regex': r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}',
            'severity': 'medium',
            'description': 'Endereço de email'
        },
        'private_key': {
            'regex': r'-----BEGIN\s+(RSA\s+|EC\s+|OPENSSH\s+|DSA\s+)?PRIVATE\s+KEY-----',
            'severity': 'critical',
            'description': 'Chave privada'
        },
        'aws_access_key': {
            'regex': r'\b(AKIA[0-9A-Z]{16}|A3T[A-Z0-9]{16}|ASIA[0-9A-Z]{16}|ACCA[0-9A-Z]{16})\b',
            'severity': 'critical',
            'description': 'Chave de acesso AWS'
        },
        'aws_secret_key': {
            'regex': r'\b[A-Za-z0-9/+=]{40}\b',
            'severity': 'high',
            'description': 'Possível chave secreta AWS'
        },
        'secret_keywords': {
            'regex': r'(?i)(password|passwd|secret|api[_-]?key|token|credential)\s*[:=]\s*["\']?[^"\'\s]{8,}["\']?',
            'severity': 'high',
            'description': 'Palavra-chave de credencial'
        },
        'cpf': {
            'regex': r'\b\d{3}\.\d{3}\.\d{3}-\d{2}\b',
            'severity': 'high',
            'description': 'CPF brasileiro'
        },
        'ssn': {
            'regex': r'\b\d{3}-\d{2}-\d{4}\b',
            'severity': 'high',
            'description': 'SSN americano'
        },
        'phone_br': {
            'regex': r'(\+55\s?)?\(?\d{2}\)?\s?9?\d{4}[- ]?\d{4}',
            'severity': 'medium',
            'description': 'Telefone brasileiro'
        },
        'credit_card': {
            'regex': r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b',
            'severity': 'critical',
            'description': 'Número de cartão de crédito'
        }
    }
    
    # Diretórios e arquivos a ignorar
    IGNORE_PATTERNS = {
        'directories': {
            '.git', '__pycache__', '.pytest_cache', 'node_modules',
            '.venv', 'venv', 'env', '.conda', 'conda-meta',
            '.ipynb_checkpoints', 'build', 'dist', '.egg-info'
        },
        'files': {
            '.gitignore', '.pyc', '.pyo', '.pyd', '.so', '.dylib',
            '.DS_Store', 'Thumbs.db', '.log'
        },
        'extensions': {
            '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.ico',
            '.mp3', '.mp4', '.avi', '.mov', '.pdf', '.zip',
            '.tar', '.gz', '.rar', '.7z'
        }
    }
    
    def __init__(self, root_path: str, exclude_patterns: List[str] | None = None, 
                 show_progress: bool = True):
        """
        Inicializa o scanner.
        
        Args:
            root_path: Caminho raiz para varredura
            exclude_patterns: Padrões adicionais para excluir
            show_progress: Mostrar barra de progresso (requer tqdm)
        """
        self.root_path = Path(root_path)
        self.exclude_patterns = set(exclude_patterns or [])
        self.findings: List[PIIFinding] = []
        self.show_progress = show_progress and HAS_TQDM
        
    def should_ignore_path(self, path: Path) -> bool:
        """Verifica se um caminho deve ser ignorado."""
        # Verificar diretórios
        for part in path.parts:
            if part in self.IGNORE_PATTERNS['directories']:
                return True
                
        # Verificar nome do arquivo
        if path.name in self.IGNORE_PATTERNS['files']:
            return True
            
        # Verificar extensão
        if path.suffix.lower() in self.IGNORE_PATTERNS['extensions']:
            return True
            
        # Verificar padrões customizados
        for pattern in self.exclude_patterns:
            if re.search(pattern, str(path)):
                return True
                
        return False
    
    def scan_file(self, file_path: Path) -> List[PIIFinding]:
        """Escaneia um arquivo em busca de PII."""
        findings = []
        
        try:
            # Tentar ler como texto UTF-8
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
        except (UnicodeDecodeError, PermissionError, IsADirectoryError):
            return findings
            
        for line_num, line in enumerate(lines, 1):
            line_content = line.strip()
            
            # Pular linhas muito longas (provavelmente dados binários)
            if len(line_content) > 1000:
                continue
                
            for pattern_name, pattern_info in self.PATTERNS.items():
                matches = re.finditer(pattern_info['regex'], line_content)
                
                for match in matches:
                    # Aplicar filtros específicos
                    if self._is_false_positive(pattern_name, match.group(), file_path, line_content):
                        continue
                        
                    finding = PIIFinding(
                        file_path=str(file_path.relative_to(self.root_path)),
                        line_number=line_num,
                        pattern_type=pattern_name,
                        matched_text=match.group(),
                        context_line=line_content,
                        severity=pattern_info['severity']
                    )
                    findings.append(finding)
                    
        return findings
    
    def _is_false_positive(self, pattern_type: str, matched_text: str, 
                          file_path: Path, context: str) -> bool:
        """Identifica falsos positivos baseado no contexto."""
        
        # Filtros para emails
        if pattern_type == 'email':
            # Ignorar emails em comentários de exemplo ou documentação
            if any(word in context.lower() for word in 
                   ['example', 'exemplo', 'test', 'dummy', 'placeholder']):
                return True
                
        # Filtros para números que podem ser coordenadas
        if pattern_type in ['cpf', 'ssn', 'credit_card']:
            # Se está em arquivo CSV com coordenadas científicas
            if file_path.suffix == '.csv' and any(word in context.lower() for word in
                                                 ['coordinate', 'coord', 'xyz', 'position']):
                return True
                
        # Filtros para chaves AWS suspeitas
        if pattern_type == 'aws_secret_key':
            # Se é uma string muito comum ou em comentário
            if matched_text in ['0000000000000000000000000000000000000000',
                               'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'] or \
               context.strip().startswith('#'):
                return True
                
        return False
    
    def scan_directory(self) -> List[PIIFinding]:
        """Escaneia recursivamente um diretório."""
        print(f"Iniciando varredura em: {self.root_path}")
        
        # Coletar todos os arquivos primeiro para barra de progresso
        all_files = []
        for file_path in self.root_path.rglob('*'):
            if file_path.is_file() and not self.should_ignore_path(file_path):
                all_files.append(file_path)
        
        print(f"Encontrados {len(all_files)} arquivos para processar")
        
        # Configurar barra de progresso
        if self.show_progress:
            file_iterator = tqdm(all_files, desc="Escaneando arquivos", unit="arquivo")  # type: ignore
        else:
            file_iterator = all_files
            
        file_count = 0
        for file_path in file_iterator:
            file_count += 1
            
            # Atualizar progresso manualmente se não há tqdm
            if not self.show_progress and file_count % 100 == 0:
                print(f"Processados {file_count}/{len(all_files)} arquivos...")
                    
            file_findings = self.scan_file(file_path)
            self.findings.extend(file_findings)
            
            # Atualizar descrição da barra com achados
            if self.show_progress and hasattr(file_iterator, 'set_postfix'):
                file_iterator.set_postfix({  # type: ignore
                    'achados': len(self.findings),
                    'arquivo': file_path.name[:20] + ('...' if len(file_path.name) > 20 else '')
                })
                
        print(f"Varredura concluída. {file_count} arquivos processados, {len(self.findings)} achados.")
        return self.findings
    
    def generate_report(self, output_format: str = 'text') -> str:
        """Gera relatório dos achados."""
        if output_format == 'json':
            return self._generate_json_report()
        else:
            return self._generate_text_report()
    
    def _generate_text_report(self) -> str:
        """Gera relatório em formato texto."""
        report = []
        report.append("=" * 60)
        report.append("RELATÓRIO DE VARREDURA PII")
        report.append("=" * 60)
        report.append(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Diretório: {self.root_path}")
        report.append(f"Total de achados: {len(self.findings)}")
        report.append("")
        
        # Agrupar por severidade
        by_severity = {}
        for finding in self.findings:
            if finding.severity not in by_severity:
                by_severity[finding.severity] = []
            by_severity[finding.severity].append(finding)
            
        # Ordenar por severidade
        severity_order = ['critical', 'high', 'medium', 'low']
        for severity in severity_order:
            if severity in by_severity:
                findings = by_severity[severity]
                report.append(f"\n🔴 {severity.upper()} PRIORITY ({len(findings)} achados)")
                report.append("-" * 50)
                
                for finding in findings:
                    report.append(f"📁 {finding.file_path}:{finding.line_number}")
                    report.append(f"🔍 Tipo: {finding.pattern_type}")
                    report.append(f"📝 Texto: {finding.matched_text}")
                    report.append(f"📄 Contexto: {finding.context_line[:100]}...")
                    report.append("")
                    
        if not self.findings:
            report.append("✅ Nenhum problema de PII encontrado!")
            
        return "\n".join(report)
    
    def _generate_json_report(self) -> str:
        """Gera relatório em formato JSON."""
        data = {
            'scan_info': {
                'timestamp': datetime.now().isoformat(),
                'directory': str(self.root_path),
                'total_findings': len(self.findings)
            },
            'findings': [
                {
                    'file_path': f.file_path,
                    'line_number': f.line_number,
                    'pattern_type': f.pattern_type,
                    'matched_text': f.matched_text,
                    'context_line': f.context_line,
                    'severity': f.severity
                }
                for f in self.findings
            ]
        }
        return json.dumps(data, indent=2, ensure_ascii=False)


def main():
    """Função principal."""
    parser = argparse.ArgumentParser(
        description='Scanner de PII para repositórios de código'
    )
    parser.add_argument(
        'path',
        nargs='?',
        default='.',
        help='Caminho do diretório para escanear (padrão: diretório atual)'
    )
    parser.add_argument(
        '--format',
        choices=['text', 'json'],
        default='text',
        help='Formato de saída do relatório'
    )
    parser.add_argument(
        '--output',
        help='Arquivo de saída (padrão: stdout)'
    )
    parser.add_argument(
        '--exclude',
        action='append',
        help='Padrões regex para excluir da varredura'
    )
    parser.add_argument(
        '--no-color',
        action='store_true',
        help='Desabilitar cores na saída'
    )
    parser.add_argument(
        '--no-progress',
        action='store_true',
        help='Desabilitar barra de progresso'
    )
    
    args = parser.parse_args()
    
    # Inicializar scanner
    scanner = PIIScanner(args.path, args.exclude, show_progress=not args.no_progress)
    
    # Avisar sobre tqdm se necessário
    if not args.no_progress and not HAS_TQDM:
        print("💡 Dica: Instale 'tqdm' para barra de progresso: pip install tqdm")
        print()
    
    # Executar varredura
    findings = scanner.scan_directory()
    
    # Gerar relatório
    report = scanner.generate_report(args.format)
    
    # Saída
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"Relatório salvo em: {args.output}")
    else:
        print(report)
    
    # Código de saída
    if findings:
        critical_high = [f for f in findings if f.severity in ['critical', 'high']]
        if critical_high:
            print(f"\n⚠️  {len(critical_high)} achados críticos/altos encontrados!")
            sys.exit(1)
        else:
            print(f"\n ℹ️  {len(findings)} achados de baixa prioridade encontrados.")
            sys.exit(0)
    else:
        print("\n✅ Nenhum problema encontrado!")
        sys.exit(0)


if __name__ == '__main__':
    main()