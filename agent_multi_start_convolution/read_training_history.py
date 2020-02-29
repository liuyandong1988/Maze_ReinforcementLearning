import json






b = open(r"steps_history.txt", "r",encoding='UTF-8')
out = b.read()
out = json.loads(out)
print(out)
print(isinstance(out,list))