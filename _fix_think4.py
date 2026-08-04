path = r'D:\vps\python\llama-cpp_vlm_web\static\js\app.js'
with open(path, 'rb') as f:
    c = f.read()

# Find "if (_isMiniCPM) {" 
idx_marker = c.find(b'if (_isMiniCPM) {')
print(f'_isMiniCPM at byte {idx_marker}')

# Find "} else if (_thinkEnded) {" after that
idx_thinkended = c.find(b'} else if (_thinkEnded) {', idx_marker)
print(f'_thinkEnded at byte {idx_thinkended}')

# The opening brace before the code to replace must be the one that
# starts the if/else-if chain. Let me find the opening pattern.
# We want to replace from "if (_isMiniCPM) {" through to just before
# the content of the `_thinkEnded` branch. Actually simpler: 
# Replace "if (_isMiniCPM)" block content + the "else if" prefix
# with just "if".

# Strategy: find the exact line boundaries
# Line before _isMiniCPM block: "] else if (data.content) {\r\n"
# Line after _isMiniCPM block (the _thinkEnded branch body starts):
# The replacement is:
#   Replace "            if (_isMiniCPM) {\r\n...\r\n            } else if (_thinkEnded) {"
#   With    "            if (_thinkEnded) {"

# But we also want to drop the _isMiniCPM-specific comment, replace with a clean one.
# Let me find exact line positions.

lines = c.split(b'\r\n')
for i, line in enumerate(lines):
    stripped = line.rstrip(b' ')
    if b'if (_isMiniCPM)' in stripped:
        print(f'Line {i}: _isMiniCPM found')
        # Print the surrounding lines
        for j in range(max(0,i-2), min(len(lines), i+2)):
            print(f'  {j}: {lines[j][:120]}')
        break

# Find closing brace of _isMiniCPM block
# The block goes: if (_isMiniCPM) { ... } else if (_thinkEnded) {
# We want to remove from "if (_isMiniCPM) {" through "} else if (_thinkEnded) {"
# and replace with just "if (_thinkEnded) {"

# Get the full text to replace
replace_start = c.find(b'if (_isMiniCPM) {')
replace_end = c.find(b'} else if (_thinkEnded) {', replace_start)
if replace_end < 0:
    print('Cannot find } else if (_thinkEnded) {')
    exit()
replace_end += len(b'}')

old_block = c[replace_start:replace_end]
print(f'Old block ({len(old_block)} bytes): {old_block[:500]}')

# The replacement: keep the leading spaces from the first line
# Extract indent from original
indent = b'            '  # 12 spaces

new_block = (
    b'if (_thinkEnded) {'
)

c = c[:replace_start] + new_block + c[replace_end:]

# Also replace the comment line above
# Find "// MiniCPM..." or "// \xe2\x94\x80..." 
comment_marker = c.find(b'// MiniCPM\xef\xbc\x9a') 
if comment_marker < 0:
    comment_marker = c.find(b'// \xe2\x94\x80\xe2\x94\x80 \xe5\x8c\xba\xe5\x88\x86 MiniCPM')
if comment_marker > replace_start - 200:
    # Comment is near - find the full line
    line_start = c.rfind(b'\r\n', 0, comment_marker)
    if line_start < 0: line_start = 0
    else: line_start += 2
    line_end = c.find(b'\r\n', comment_marker)
    if line_end < 0: line_end = len(c)
    old_comment_line = c[line_start:line_end]
    new_comment_line = b'            // MiniCPM5 \xe8\xb5\xb0\xe6\xa0\x87\xe5\x87\x86 think-buffer (\xe6\x8a\x98\xe5\x8f\xa0 <think> \xe6\xa0\x87\xe7\xad\xbe)'
    c = c[:line_start] + new_comment_line + c[line_end:]
    print(f'Replaced comment: {old_comment_line[:100]} -> {new_comment_line[:80]}')

with open(path, 'wb') as f:
    f.write(c)

# Verify - check critical lines
text = c.decode('utf-8')
text_lines = text.split('\n')
for i, line in enumerate(text_lines):
    if 'isMiniCPM' in line or 'thinkEnded' in line or 'think-buffer' in line:
        print(f'  CHECK {i+1}: {line.strip()[:120]}')
print('[OK] Done')
