#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Scan all tool schemas for type: null fields (causes DeepSeek API errors)"""
import urllib.request
import json

r = urllib.request.urlopen('http://127.0.0.1:5000/api/tools/list', timeout=10)
data = json.loads(r.read())

null_types = []
for t in data['tools']:
    func = t.get('function', {})
    name = func.get('name', 'unknown')
    params = func.get('parameters', {})
    
    def scan(obj, path=""):
        if isinstance(obj, dict):
            if obj.get("type") is None and "type" in obj:
                null_types.append(f"  TOOL={name} PATH={path}")
                print(f"  [NULL] {name} {path}")
            for k, v in obj.items():
                scan(v, f"{path}.{k}" if path else k)
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                scan(v, f"{path}[{i}]")
    
    scan(params, "parameters")

print(f"\nTotal null types: {len(null_types)}")
for n in null_types:
    print(n)