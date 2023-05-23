import cohere
import time

# API key 
co = cohere.Client('VgR2hXk1OC9UOiTWFYE1rTodw1GkT7xYKI6MsLIS') 

qList = ['How can I draw landscape?', 
         'How can I start drawing landcape elements like trees?', 
         'What is landscape?', 
         'How do you evaluate landscape sketching?', 
         'Describe landscape sketching.', 
         'How can I improve landscape sketching?', 
         'Explain the process of landscape sketching.', 
         'What should I know about landscape sketching?', 
         'How can I draw object?', 
         'How can I draw compositions?',
         'What should I look for landscape?', 
         'What is most important in landscape sketching?', 
         'Steps to draw objects.', 
         'How can I draw different types of landscape compositions?',
         'Explain landscape sketching.']

# tracking number of generations (max 5 per min)
i = 0

prompts = []
prompts = prompts + qList

# amplifying prompts 
amp = 3
for num in range(1):
    for q in qList:
        response = co.generate(
            model='command-nightly',
            prompt='generate ' + str(amp) + ' similar questions to '+ q,
            max_tokens=300,
            temperature=0.9,
            k=0,
            stop_sequences=[],
            return_likelihoods='NONE')
        qAmp = list(response.generations[0].text.split('\n'))
        prompts = prompts + qAmp
        i = i + 1
        if i%5 == 4:
            time.sleep(60)


prompt2 = [x for x in prompts if x != ""]
print(len(prompt2))
print(prompt2)

conversation = []
tempConv = ''
# creating conversation 
for p in prompt2:
    response = co.generate(
            model='command-nightly',
            prompt=p,
            max_tokens=300,
            temperature=0.9,
            k=0,
            stop_sequences=[],
            return_likelihoods='NONE')
    refined = []
    bout = response.generations[0].text[2:]
    if bout[0:2] != '\n':
        bout = response.generations[0].text[1:]
    tempConv = 'Question: ' + p + '\n' + 'Answer: ' + bout
    conversation.append(tempConv)
    i = i + 1
    if i%5 ==4:
        time.sleep(60)

with open("conversation.py", 'w') as conv:
    conv.write("conversation = [\n")    
    for line in conversation:
        conv.write('\'\'\''+ line + '\n' + '\'\'\',\n' + '    ')
    conv.write("]")

print('end')