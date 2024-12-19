from openai import OpenAI
import json
from config import BASE_URL, API_KEY


client = OpenAI(
    base_url=BASE_URL,
    api_key=API_KEY
)

messages = []
# messages = load_from_file("output.json")

def load_from_file(filename):
    with open(filename, 'r') as f:
        return json.load(f)

msg = "Hello."

def write_to_file(filename, data):
    with open(filename, 'w') as f:
        # 不转义，使用4空格
        json.dump(data, f, indent=4, ensure_ascii=False)

while True:
    
    msg = input("User: ")
    
    if msg == 'clear':
        messages = []
        continue
    
    messages.append({"role": "user", "content": msg})
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )
    reply = completion.choices[0].message.content

    messages.append({"role": "assistant", "content": reply})
    
    write_to_file("output.json", messages)
    print("AI:", reply)
