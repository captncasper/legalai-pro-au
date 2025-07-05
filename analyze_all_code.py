#!/usr/bin/env python3
"""Analyze all Python files to find reusable smart features"""

import os
import ast
import re
from pathlib import Path
from collections import defaultdict
import json

class CodeAnalyzer:
    def __init__(self):
        self.features_found = defaultdict(list)
        self.data_sources = []
        self.ml_models = []
        self.api_endpoints = []
        self.data_extractors = []
        
    def analyze_file(self, filepath):
        """Analyze a Python file for useful features"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            filename = Path(filepath).name
            
            # Look for key patterns
            patterns = {
                'semantic_search': [
                    r'SentenceTransformer|sentence_transformers',
                    r'embeddings?|encode\(',
                    r'cosine_similarity|semantic'
                ],
                'ml_prediction': [
                    r'predict|prediction|classifier',
                    r'RandomForest|XGBoost|neural',
                    r'train_test_split|fit\('
                ],
                'data_extraction': [
                    r'extract.*amount|settlement.*\$|damages.*\d',
                    r'judge.*pattern|judge.*analysis',
                    r'duration|timeline|days|months'
                ],
                'austlii_scraping': [
                    r'austlii\.edu\.au|AustLII',
                    r'scrape|crawler|BeautifulSoup',
                    r'requests\.get.*austlii'
                ],
                'rag_implementation': [
                    r'ChromaDB|FAISS|vector.*store',
                    r'retrieval.*augmented|RAG',
                    r'chunk|split.*text'
                ],
                'quantum_analysis': [
                    r'quantum|superposition|entanglement',
                    r'monte.*carlo|simulation',
                    r'probability.*distribution'
                ],
                'api_endpoints': [
                    r'@app\.(post|get|put)',
                    r'@router\.(post|get|put)',
                    r'async def.*\(.*request'
                ],
                'judge_analysis': [
                    r'judge.*pattern|judge.*behavior',
                    r'judge.*statistics|judge.*analysis',
                    r'judge_id|judge_name'
                ],
                'settlement_extraction': [
                    r'settlement.*amount|\$[\d,]+',
                    r'damages.*awarded|compensation',
                    r'extract.*monetary|parse.*amount'
                ],
                'case_duration': [
                    r'duration|timeline|days.*between',
                    r'filing.*date|judgment.*date',
                    r'calculate.*time|processing.*time'
                ]
            }
            
            # Check each pattern
            for feature, patterns_list in patterns.items():
                for pattern in patterns_list:
                    if re.search(pattern, content, re.IGNORECASE):
                        self.features_found[feature].append({
                            'file': filename,
                            'pattern': pattern,
                            'context': self._extract_context(content, pattern)
                        })
                        break
            
            # Look for specific implementations
            self._analyze_ast(filepath, content)
            
        except Exception as e:
            pass
    
    def _extract_context(self, content, pattern):
        """Extract code context around pattern"""
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if re.search(pattern, line, re.IGNORECASE):
                start = max(0, i-2)
                end = min(len(lines), i+3)
                return '\n'.join(lines[start:end])
        return ""
    
    def _analyze_ast(self, filepath, content):
        """Use AST to find specific implementations"""
        try:
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                # Find class definitions
                if isinstance(node, ast.ClassDef):
                    class_name = node.name
                    
                    # Check for specific class types
                    if 'Extractor' in class_name or 'Parser' in class_name:
                        self.data_extractors.append({
                            'file': Path(filepath).name,
                            'class': class_name
                        })
                    elif 'Model' in class_name or 'Predictor' in class_name:
                        self.ml_models.append({
                            'file': Path(filepath).name,
                            'class': class_name
                        })
                
                # Find function definitions
                elif isinstance(node, ast.FunctionDef):
                    func_name = node.name
                    
                    # Check for data source functions
                    if any(term in func_name.lower() for term in ['austlii', 'scrape', 'fetch', 'download']):
                        self.data_sources.append({
                            'file': Path(filepath).name,
                            'function': func_name
                        })
        except:
            pass
    
    def analyze_directory(self):
        """Analyze all Python files in current directory"""
        python_files = []
        
        # Get all Python files
        for file in Path('.').glob('*.py'):
            if file.name != 'analyze_all_code.py':
                python_files.append(file)
        
        print(f"🔍 Analyzing {len(python_files)} Python files...")
        
        # Analyze each file
        for file in python_files:
            self.analyze_file(file)
        
        return self.generate_report()
    
    def generate_report(self):
        """Generate analysis report"""
        report = {
            'summary': {
                'total_files_analyzed': len(set(f['file'] for features in self.features_found.values() for f in features)),
                'features_found': {k: len(v) for k, v in self.features_found.items()},
                'ml_models_found': len(self.ml_models),
                'data_extractors_found': len(self.data_extractors),
                'data_sources_found': len(self.data_sources)
            },
            'detailed_findings': dict(self.features_found),
            'ml_models': self.ml_models,
            'data_extractors': self.data_extractors,
            'data_sources': self.data_sources
        }
        
        return report

def main():
    analyzer = CodeAnalyzer()
    report = analyzer.analyze_directory()
    
    print("\n📊 CODE ANALYSIS REPORT")
    print("=" * 60)
    
    # Summary
    print("\n🎯 Summary:")
    for feature, count in report['summary']['features_found'].items():
        if count > 0:
            print(f"  ✅ {feature}: Found in {count} locations")
    
    # Specific findings
    if report['summary']['features_found'].get('semantic_search', 0) > 0:
        print("\n🔍 Semantic Search Implementations:")
        for item in report['detailed_findings']['semantic_search'][:3]:
            print(f"  - {item['file']}")
    
    if report['summary']['features_found'].get('ml_prediction', 0) > 0:
        print("\n🤖 ML Prediction Models:")
        for item in report['detailed_findings']['ml_prediction'][:3]:
            print(f"  - {item['file']}")
    
    if report['summary']['features_found'].get('settlement_extraction', 0) > 0:
        print("\n💰 Settlement Amount Extraction:")
        for item in report['detailed_findings']['settlement_extraction'][:3]:
            print(f"  - {item['file']}")
    
    if report['summary']['features_found'].get('judge_analysis', 0) > 0:
        print("\n👨‍⚖️ Judge Analysis Code:")
        for item in report['detailed_findings']['judge_analysis'][:3]:
            print(f"  - {item['file']}")
    
    # Save full report
    with open('code_analysis_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print("\n💾 Full report saved to: code_analysis_report.json")
    
    # Recommendations
    print("\n💡 RECOMMENDATIONS:")
    print("Based on your existing code, you should:")
    
    recommendations = []
    
    if report['summary']['features_found'].get('semantic_search', 0) > 0:
        recommendations.append("1. Reuse semantic search from: " + 
                             report['detailed_findings']['semantic_search'][0]['file'])
    
    if report['summary']['features_found'].get('settlement_extraction', 0) > 0:
        recommendations.append("2. Use settlement extraction from: " + 
                             report['detailed_findings']['settlement_extraction'][0]['file'])
    
    if report['summary']['features_found'].get('judge_analysis', 0) > 0:
        recommendations.append("3. Leverage judge analysis from: " + 
                             report['detailed_findings']['judge_analysis'][0]['file'])
    
    if report['summary']['features_found'].get('austlii_scraping', 0) > 0:
        recommendations.append("4. Use AustLII scraper from: " + 
                             report['detailed_findings']['austlii_scraping'][0]['file'])
    
    for rec in recommendations:
        print(f"  {rec}")

if __name__ == "__main__":
    main()
