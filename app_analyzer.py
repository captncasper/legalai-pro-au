import numpy as np
#!/usr/bin/env python3
"""
Application Structure Analyzer
Scans a Python project directory to understand data flow and module interactions
"""

import os
import ast
import sys
from pathlib import Path
from collections import defaultdict
import json

class AppAnalyzer:
    def __init__(self, root_dir="."):
        self.root_dir = Path(root_dir)
        self.modules = {}
        self.imports = defaultdict(list)
        self.functions = defaultdict(list)
        self.classes = defaultdict(list)
        self.call_graph = defaultdict(list)
        
    def find_python_files(self):
        """Find all Python files in the directory"""
        py_files = []
        for file in self.root_dir.rglob("*.py"):
            # Skip virtual environments and cache
            if any(skip in str(file) for skip in ['venv', '__pycache__', '.env']):
                continue
            py_files.append(file)
        return sorted(py_files)
    
    def parse_file(self, filepath):
        """Parse a Python file and extract imports, functions, and classes"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                tree = ast.parse(content)
                
            rel_path = filepath.relative_to(self.root_dir)
            self.modules[str(rel_path)] = {
                'content': content[:500] + '...' if len(content) > 500 else content,
                'size': len(content),
                'imports': [],
                'functions': [],
                'classes': []
            }
            
            for node in ast.walk(tree):
                # Extract imports
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        self.modules[str(rel_path)]['imports'].append(alias.name)
                        self.imports[str(rel_path)].append(('import', alias.name))
                        
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ''
                    for alias in node.names:
                        import_str = f"{module}.{alias.name}" if module else alias.name
                        self.modules[str(rel_path)]['imports'].append(import_str)
                        self.imports[str(rel_path)].append(('from', module, alias.name))
                
                # Extract function definitions
                elif isinstance(node, ast.FunctionDef):
                    func_info = {
                        'name': node.name,
                        'args': [arg.arg for arg in node.args.args],
                        'lineno': node.lineno
                    }
                    self.modules[str(rel_path)]['functions'].append(func_info)
                    self.functions[str(rel_path)].append(func_info)
                
                # Extract class definitions
                elif isinstance(node, ast.ClassDef):
                    class_info = {
                        'name': node.name,
                        'bases': [self.get_name(base) for base in node.bases],
                        'methods': [n.name for n in node.body if isinstance(n, ast.FunctionDef)],
                        'lineno': node.lineno
                    }
                    self.modules[str(rel_path)]['classes'].append(class_info)
                    self.classes[str(rel_path)].append(class_info)
                    
        except Exception as e:
            print(f"Error parsing {filepath}: {e}")
            
    def get_name(self, node):
        """Extract name from AST node"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self.get_name(node.value)}.{node.attr}"
        return str(node)
    
    def analyze_main_py(self):
        """Special analysis for main.py"""
        main_path = self.root_dir / "main.py"
        if not main_path.exists():
            print("Warning: main.py not found!")
            return None
            
        print("\n=== MAIN.PY ANALYSIS ===")
        with open(main_path, 'r') as f:
            content = f.read()
            
        # Print first 50 lines to understand entry point
        lines = content.split('\n')
        print(f"\nFirst 50 lines of main.py:")
        print("-" * 60)
        for i, line in enumerate(lines[:50], 1):
            print(f"{i:3d}: {line}")
        print("-" * 60)
        
        return content
        
    def generate_import_graph(self):
        """Generate a visual representation of import relationships"""
        print("\n=== IMPORT DEPENDENCY GRAPH ===")
        
        # Find internal imports (imports between project modules)
        project_modules = {str(p.relative_to(self.root_dir)).replace('.py', '').replace('/', '.')
                          for p in self.find_python_files()}
        
        for module, imports in self.imports.items():
            internal_imports = []
            for imp in imports:
                if imp[0] == 'import':
                    if any(imp[1].startswith(pm) for pm in project_modules):
                        internal_imports.append(imp[1])
                elif imp[0] == 'from' and imp[1]:
                    if any(imp[1].startswith(pm) for pm in project_modules):
                        internal_imports.append(f"{imp[1]}.{imp[2]}")
                        
            if internal_imports:
                print(f"\n{module}:")
                for imp in internal_imports:
                    print(f"  └─> {imp}")
                    
    def generate_summary(self):
        """Generate a comprehensive summary"""
        print("\n=== PROJECT SUMMARY ===")
        
        py_files = self.find_python_files()
        print(f"\nTotal Python files: {len(py_files)}")
        
        # Sort files by importance (main.py first, then by size)
        sorted_files = sorted(py_files, key=lambda x: (
            x.name != 'main.py',  # main.py comes first
            -os.path.getsize(x)   # then by size
        ))
        
        print("\nMost important files (by size and structure):")
        for i, file in enumerate(sorted_files[:10], 1):
            rel_path = file.relative_to(self.root_dir)
            size = os.path.getsize(file)
            module_info = self.modules.get(str(rel_path), {})
            
            print(f"\n{i}. {rel_path} ({size:,} bytes)")
            if module_info:
                print(f"   Functions: {len(module_info.get('functions', []))}")
                print(f"   Classes: {len(module_info.get('classes', []))}")
                print(f"   Imports: {len(module_info.get('imports', []))}")
                
                # Show main classes and functions
                if module_info.get('classes'):
                    print(f"   Key classes: {', '.join(c['name'] for c in module_info['classes'][:3])}")
                if module_info.get('functions'):
                    print(f"   Key functions: {', '.join(f['name'] for f in module_info['functions'][:3])}")
    
    def save_analysis(self, output_file="app_analysis.json"):
        """Save analysis results to JSON file"""
        analysis_data = {
            'modules': self.modules,
            'import_graph': dict(self.imports),
            'summary': {
                'total_files': len(self.modules),
                'total_functions': sum(len(funcs) for funcs in self.functions.values()),
                'total_classes': sum(len(classes) for classes in self.classes.values())
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(analysis_data, f, indent=2)
        print(f"\n✓ Detailed analysis saved to {output_file}")
    
    def run(self):
        """Run the complete analysis"""
        print("🔍 Analyzing Python application structure...\n")
        
        # Find all Python files
        py_files = self.find_python_files()
        print(f"Found {len(py_files)} Python files")
        
        # Parse each file
        print("\nParsing files...")
        for file in py_files:
            self.parse_file(file)
            
        # Analyze main.py specifically
        self.analyze_main_py()
        
        # Generate import graph
        self.generate_import_graph()
        
        # Generate summary
        self.generate_summary()
        
        # Save detailed analysis
        self.save_analysis()
        
        print("\n✅ Analysis complete!")
        print("\nTo understand data flow:")
        print("1. Check main.py above to see the entry point")
        print("2. Follow the import graph to see module dependencies")
        print("3. Check app_analysis.json for detailed information")
        print("\nTip: Look for patterns like:")
        print("- Database models (often in models.py or db/)")
        print("- API routes (often in routes/, api/, or views/)")
        print("- Business logic (often in services/, core/, or utils/)")


if __name__ == "__main__":
    # Run analyzer in current directory or specified path
    path = sys.argv[1] if len(sys.argv) > 1 else "."
    analyzer = AppAnalyzer(path)
    analyzer.run()
