path = r'D:\vps\python\llama-cpp_vlm_web\static\js\app.js'
with open(path, 'rb') as f:
    c = f.read()

old = (
    b'          } else if (data.content) {\r\n'
    b'            // \xe2\x94\x80\xe2\x94\x80 \xe5\x8c\xba\xe5\x88\x86 MiniCPM \xe8\xbe\x93\xe5\x87\xba vs \xe5\x85\xb6\xe4\xbd\x99\xe6\xa8\xa1\xe5\x9e\x8b\xe5\x88\x87\xe6\xa0\x87\xe7\xad\xbe \xe2\x94\x80\xe2\x94\x80\r\n'
    b'            if (_isMiniCPM) {\r\n'
    b'              // MiniCPM\xef\xbc\x9a\xe5\x90\x8e\xe7\xab\xaf R1 \xe7\xbc\x93\xe5\x86\xb2\xe5\xb7\xb2\xe5\x88\x86\xe7\xa6\xbb thinking/content\xef\xbc\x8ccontent \xe6\x98\xaf\xe5\xb9\xb2\xe5\x87\x80\xe6\xad\xa3\xe6\x96\x87\r\n'
    b'              _thinkEnded = true;\r\n'
    b'              turnText += data.content;\r\n'
    b'              _tokenCount++;\r\n'
    b'              updateMsgTps(msgEl, _tokenCount, _tpsStart);\r\n'
    b'              const displayText = turnText.replace(/<tool_call\\s+name="[^"]*">[\\s\\S]*<\\/tool_call>/gi, \'\xe2\x9a\x99\xef\xb8\x8f ...\');\r\n'
    b'              bubble.innerHTML = bubbleBase + renderMarkdown(displayText);\r\n'
    b'              msgEl.scrollIntoView({ behavior: \'smooth\', block: \'end\' });\r\n'
    b'            } else if (_thinkEnded) {'
)

new = (
    b'          } else if (data.content) {\r\n'
    b'            // MiniCPM5 \xe8\xb5\xb0\xe6\xa0\x87\xe5\x87\x86 think-buffer \xe6\xb5\x81\xe7\xa8\x8b\xef\xbc\x88<\x74\x68\x69\x6e\x6b> \xe6\xa0\x87\xe7\xad\xbe\xe7\xbc\x93\xe5\x86\xb2+\xe6\x8a\x98\xe5\x8f\xa0\xef\xbc\x89\r\n'
    b'            if (_thinkEnded) {'
)

if old in c:
    c = c.replace(old, new)
    with open(path, 'wb') as f:
        f.write(c)
    print('[OK] Binary replacement done')
else:
    # Find the block manually
    idx = c.find(b'if (_isMiniCPM) {')
    if idx >= 0:
        print(f'Found _isMiniCPM at byte {idx}')
        # Show the block in hex
        block = c[idx-10:idx+400]
        print(f'Block hex: {block.hex()}')
    print('[ERR] old block not found at byte level')
