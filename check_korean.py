import re

def check_korean_chars():
    with open('utils/data_visualization.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    korean_chars = []
    for match in re.finditer(r'[가-힣]+', content):
        korean_chars.append((match.start(), match.group(), content[max(0, match.start()-20):match.end()+20]))
    
    print(f'Found {len(korean_chars)} Korean text segments:')
    for i, (pos, text, context) in enumerate(korean_chars):
        print(f'{i+1}. Position {pos}: "{text}" in context: "{context}"')

if __name__ == "__main__":
    check_korean_chars()
