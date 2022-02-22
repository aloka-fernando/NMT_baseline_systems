import os
import time
import torch
import math
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast


checkpoint="mbart50-ft-en-si-run2"
os.chdir('/userdirs/aloka/nmt_baseline_experiments/mbart50ft_huggingface_api')
#initialize tokenizer
tokenizer = MBart50TokenizerFast.from_pretrained(checkpoint, src_lang="en_XX")

#text to translate
#src_text = ["ප්‍රාදේශීය සේවා වියුක්තියට විසඳුමක් ලෙස රැකියා අවස්ථා අටසිය පනහක් සපයමින්, වාණිජ නිෂ්පාදන කටයුතු අරඹන ලදි."]
#tgt_text =  'Commercial production is in progress, employing nearly 850  people under this project,  which is a solution for regional service provision.'
#fairseq_trans='Commercial production commenced by providing eight hundred and forty-eight job opportunities as a solution to regional unemployment.'
#print('[src] {}\n[ref] {}\n[fairseq_trans] {}'.format(src_text, tgt_text, fairseq_trans))
#print('Huggingface translations:')



src_lines=[line.strip() for line in open('data/parallel-27.04.2021-ts.un.si-en-ta.en', 'r', encoding='utf8')]
print('No of lines in ts set : {}'.format(len(src_lines)))

#parameters
input_batchSize=32
input_batches= math.ceil(len(src_lines)/input_batchSize)


directories=os.listdir('/userdirs/aloka/nmt_baseline_experiments/mbart50ft_huggingface_api/'+checkpoint)
startTime=time.time() 
modelLoadingTime=0
for subDir in directories :
    print(subDir)
    
    hf_trans_lines=[]

    if subDir.find('checkpoint-') != -1: 
        print(subDir)  
        fileOut = open('data/parallel-27.04.2021-ts-translated-run2-checkpoint-450000-r2.un.en-si.si', 'w', encoding='utf8')

        startTime=time.time() 
        model = MBartForConditionalGeneration.from_pretrained(checkpoint+"/"+subDir).to("cuda")  
        modelLoadingTime=time.time()
        print('Model Loading Time: {:0.2f}s'.format(modelLoadingTime-startTime))
        
        
        src_lines_minibatch=[]

        for i in range(0,input_batches):

            torch.cuda.empty_cache()
            if i*input_batchSize+input_batchSize < len(src_lines):
                src_lines_minibatch=src_lines[i*input_batchSize:i*input_batchSize+input_batchSize]
            else:
                src_lines_minibatch=src_lines[i*input_batchSize:len(src_lines)]                      
        
            
               
            model_inputs = tokenizer(src_lines_minibatch, padding=True, truncation=True, return_tensors="pt")      
                
            generated_tokens = model.generate(**model_inputs.to("cuda"), forced_bos_token_id=tokenizer.lang_code_to_id["si_LK"])
                
            trans_lines=tokenizer.batch_decode(generated_tokens, batch_size=32, skip_special_tokens=True)
            hf_trans_lines=hf_trans_lines+ trans_lines                

            #print('Translation complete mini-batch {}/{}'.format(i,input_batches))
            print('Total translated lines : {}'.format(len(hf_trans_lines)))
            
        for line in hf_trans_lines:        
                line=line.replace("\u0dca\u0020\u0dbb", "\u0DCA\u200D\u0dbb") 
                line=line.replace("\u0dca\u0020\u0dba", "\u0DCA\u200D\u0dba")         
                fileOut.write('{}\n'.format(line))

        fileOut.close()        
        endTime=time.time()

        print('{} : {:0.2f}min'.format(subDir, (endTime-modelLoadingTime)/60))        
        os.system("sacrebleu -tok 'none' -s 'none' 'data/parallel-27.04.2021-ts.un.si-en-ta.si' < 'data/parallel-27.04.2021-ts-translated-run2-checkpoint-450000-r2.un.en-si.si'")
        
        

print('Evaluating Testset complete!')