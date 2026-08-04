path = r'D:\vps\python\llama-cpp_vlm_web\static\js\app.js'
with open(path, 'rb') as f:
    c = f.read()

# Block structure:
#   (line 779)           if (_isMiniCPM) {
#   (line 780-788)         ... MiniCPM branch body ...
#   (line 789)           } else if (_thinkEnded) {
#
# We want:
#   (line 779)           if (_thinkEnded) {

# Find "if (_isMiniCPM) {" 
start = c.find(b'if (_isMiniCPM) {')
# Find "} else if (_thinkEnded) {" - the closing brace + else if
end = c.find(b'} else if (_thinkEnded) {', start)
end += len(b'} else ')  # remove "} else " but keep "if (_thinkEnded) {"

old_block = c[start:end]
new_block = b'if ('

if old_block in c:
    c = c.replace(old_block, new_block, 1)
    print(f'Replaced {len(old_block)} bytes -> {len(new_block)} bytes')
else:
    print(f'Block not found. Block: {old_block[:100]}...')

# Fix comment line: replace "// \xe2\x94\x80\xe2\x94\x80 \xe4\xb8\xa4\xe8\xb7\xaf\xe5\x88\x86\xe6\xb5\x81..."
comment_old = c.find(b'MiniCPM \xe7\x9b\xb4\xe5\x87\xba vs')
if comment_old < 0:
    comment_old = c.find(b'// \xe2\x94\x80\xe2\x94\x80 \xe4\xb8\xa4\xe8\xb7\xaf')
if comment_old > 0:
    # Find the full line
    ls = c.rfind(b'\r\n', 0, comment_old) + 2
    le = c.find(b'\r\n', comment_old)
    old_line = c[ls:le]
    new_line = b'            // MiniCPM5 \xe8\xb5\xb0\xe6\xa0\x87\xe5\x87\x86 think-buffer (\xe7\xbc\x93\xe5\x86\xb2+\xe6\x8a\x98\xe5\x8f\xa0 <think> \xe6\xa0\x87\xe7\xad\xbe)'
    c = c[:ls] + new_line + c[le:]
    print(f'Comment replaced')

with open(path, 'wb') as f:
    f.write(c)

# Verify
text = c.decode('utf-8')
for i, line in enumerate(text.split('\n'), 1):
    if 'isMiniCPM' in line or 'thinkEnded' in line or 'think-buffer' in line:
        print(f'  L{i}: {line.strip()[:130]}')

# Check for duplicate if patterns
if 'if (_thinkEnded) { if (_thinkEnded)' in text or 'if (_thinkEnded) { else if' in text:
    print('[WARN] Duplicate if pattern!')
    print('[WARN] Duplicate if pattern!')
else:
    print('[OK] Clean')
