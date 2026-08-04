path = r'D:\vps\python\llama-cpp_vlm_web\static\js\app.js'
with open(path, 'rb') as f:
    c = f.read()

# Find the block by scanning _isMiniCPM
idx = c.find(b'if (_isMiniCPM) {')
if idx < 0:
    print('NOT FOUND')
    exit()

# Find the full block start: the "} else if (data.content) {" line above
# Scan backwards for the opening
start = c.rfind(b'} else if (data.content) {', 0, idx)
if start < 0:
    # Try other possible prefix
    start = c.rfind(b'else if (data.content)', 0, idx)
if start < 0:
    print('Cannot find block start')
    exit()

# Find blocking end: after "} else if (_thinkEnded) {" we need to
# keep everything from that point. The block to remove ends at
#   "            } else if (_thinkEnded) {"
# So find that closing pattern
end_pattern = b'} else if (_thinkEnded) {'
end_idx = c.find(end_pattern, idx)
if end_idx < 0:
    print('Cannot find end')
    exit()
end_idx += len(b'}')

# Extract old block
old_block = c[start:end_idx]
print(f'Block: {start}-{end_idx} ({len(old_block)} bytes)')

# Build new block: keep the "} else if (data.content) {" line but change
# the comment and remove _isMiniCPM branch, go straight to _thinkEnded
new_block = b'''          } else if (data.content) {
            // MiniCPM5 \xe8\xb5\xb0\xe6\xa0\x87\xe5\x87\x86 think-buffer \xe6\xb5\x81\xe7\xa8\x8b (\xe6\x8a\x98\xe5\x8f\xa0 <think> \xe6\xa0\x87\xe7\xad\xbe)
            if'''

c = c[:start] + new_block + c[end_idx:]

# Verify syntax
try:
    compile(c.decode('utf-8'), 'app.js', 'exec')
    print('[OK] JavaScript syntax valid')
except SyntaxError as e:
    print(f'[WARN] JS syntax: {e}')

with open(path, 'wb') as f:
    f.write(c)
print('[OK] Saved')
