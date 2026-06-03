#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Quick scan: check if each tool's parameters has 'type: object' at the right level"""
import urllib.request, json, sys
sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', closefd=False)

r = urllib.request.urlopen('http://127.0.0.1:5000/api/tools/list', timeout=10)
data = json.loads(r.read())

for t in data['tools']:
    func = t['function']
    name = func['name']
    params = func.get('parameters', {})
    
    has_type = params.get('type') is not None
    has_props = 'properties' in params or 'required' in params
    
    if has_type and has_props:
        print(f"  OK     {name}: type={params.get('type')}, has properties")
    elif has_props:
        print(f"  BROKEN {name}: type=MISSING, has properties but no 'type: object'")
    else:
        # Nested: check one level deeper
        inner = params.get('parameters')
        if inner and isinstance(inner, dict) and inner.get('type') == 'object':
            print(f"  NESTED {name}: outer params has name/desc/parameters, inner has type:object")
        else:
            print(f"  ?      {name}: params={list(params.keys())}")