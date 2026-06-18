import sys, json, traceback
sys.path.insert(0, r'D:\vps\python\llama-cpp_vlm_web')
try:
    from tools.registry import get_registry
    r = get_registry()
    tools = r.list_available()
    print(f'Total tools: {len(tools)}')
    for t in tools:
        s = t.to_openai_schema()
        params = s.get('function', {}).get('parameters')
        if params is None:
            print(f'BAD(NULL params): {t.name}')
        elif not isinstance(params, dict):
            print(f'BAD(not dict): {t.name} params={type(params).__name__}')
        elif params.get('type') != 'object':
            print(f'BAD(type={params.get("type")}): {t.name}')
        else:
            continue
        # print full schema for bad ones
        print(json.dumps(s, ensure_ascii=False, indent=2)[:500])
    print('Done')
except:
    traceback.print_exc()
