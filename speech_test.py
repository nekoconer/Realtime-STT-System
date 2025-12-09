#Use this py to test the program that can transcode the audio.
from transformers import WhisperProcessor,WhisperForConditionalGeneration,TextStreamer
#from datasets import load_dataset
import librosa
import torch
import numpy as np
from pydub import AudioSegment
import math
import os
import tempfile
from openai import OpenAI
client = OpenAI(api_key="sk-", base_url="https://api.deepseek.com")
stream_mode = True
messages_input={"role": "system", "content": "你是一个文本翻译助手，输入的文本可能存在语法错误和单词拼写错误请适度纠正，然后直接输出翻译后的中文，不要输出其他无关的文本。"}
special_tokens=["<|startoftranscript|>","<|ja|>","<|transcribe|>","<|notimestamps|>","<|endoftext|>"]
'''class CustomTextStreamer(TextStreamer):
    def __init__(self, tokenizer):
        super().__init__(tokenizer)
    def put(self, text):
        # 过滤掉特殊字符
        try:
            y = text.size(1)
            token_ids = text.squeeze().tolist()
        except Exception as e:
             token_ids = text.tolist()          
        decoded_text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
        print(decoded_text, end="")'''
        
def split_audio(audio_path,chunk_length_ms=30000,overlape_time_ms = 5000):
    global temp_files
    audio = AudioSegment.from_file(audio_path)
    total_length_ms = len(audio)
    step_time = chunk_length_ms - overlape_time_ms
    num_chunks = math.ceil((total_length_ms-overlape_time_ms)/step_time)
    for i in range(num_chunks):
        if i == 0:
            start_time = i * chunk_length_ms
            end_time = min(chunk_length_ms,total_length_ms)
        else:
            start_time = i*step_time
            end_time = min(chunk_length_ms+i*step_time,total_length_ms)
        chunk = audio[start_time:end_time]
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp_files.append(temp_file.name)
        chunk.export(temp_file.name,format="wav")
def get_response(client,messages,stream=False):
    try:
        response=client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            stream = stream
        )
        return response
    except Exception as e:
        print(f"It have some wrong in API connection:{str(e)}")
        return None

def check_chunk(chunk):
    #print(chunk)
    try:
        if(chunk.choices[0].delta.content is not None):
            if(chunk.choices[0].delta.content.find(": keep-alive")!= -1):
                print("Sever is full, please wait for response")
            else:
                return True   
    except Exception as e:
        print("There is something wrong with chunk")
        return False
        

def get_stream_response(response):
    return "".join(
        chunk.choices[0].delta.content
        for chunk in response
        if(check_chunk)
        #if chunk.get('choices') and chunk['choices'][0].get('delta') and chunk['choices'][0]['delta'].get('content')
    )

def chat(audio_text,file=None):
    global messages_input
    question = audio_text
    messages_total = []
    messages_total.append(messages_input)
    messages_total.append({"role":"user","content":question})
    response = get_response(client,messages_total,stream_mode)
    temp_text =""
    if response:
        print(f"翻以前的文本:{audio_text}")
        if stream_mode:
            print("翻译后的文本:",end='')
            for chunk in response:
                #print(chunk)
                if(check_chunk(chunk)):
                    chunk_message=chunk.choices[0].delta.content
                    temp_text+=chunk_message
                    print(chunk_message,end='',flush=True)               
            print()
        else:
            print(f"翻译后的文本:{response.choices[0].message.content}")
        #append_message = get_stream_response(response) if stream_mode else response.choices[0].message.content 
        #print("!!!!!!!!!!!!")
        #print(temp_text)
        print(temp_text,file=file)
        file.flush()
        
        #messages_input.append({"role": "assistant", "content":append_message})
    else:
        print("There have no response, please check the input or code for message")

def AsrAndTrans(temp_files,device,jp_file,zh_file):
    global model,processor
    for audio in temp_files:
        audio, sr = librosa.load(audio, sr=16000)
        model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
            language="ja", task="transcribe"
        )
        model.to(device)
        '''
            在TextStreamer源码中直接skip_special_token来保证流式输出中没有特殊字符
        '''
        streamer = TextStreamer(processor.tokenizer)
        #streamer =CustomTextStreamer(processor.tokenizer)
        #print(audio)
        input= processor(audio,sampling_rate=sr,return_tensor="pt")
        input_feature = input.input_features

        if isinstance(input_feature, np.ndarray):
            input_feature = torch.from_numpy(input_feature)
        input_feature=input_feature.to(device)
        predicted_ids = model.generate(input_feature,
                                    streamer=streamer,
                                    temperature=0.95,
                                    do_sample=True,
                                        top_k=5,
                                        top_p=0.6,
                                        repetition_penalty=1.1,
                                        pad_token_id=processor.tokenizer.eos_token_id)
        for chunk in predicted_ids:
            #print(chunk)
            print("_______________")
            transcription = processor.batch_decode(predicted_ids,skip_special_tokens=True)
            chat(transcription[0],zh_file)
            print(transcription[0],file =jp_file)
            jp_file.flush()
    jp_file.close()
    zh_file.close()
    for temp_file in temp_files:
        os.remove(temp_file)

  
device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "./model"
audio_path ="./voice/3.mp3"
temp_files=[]
chunk_length_ms = 30000
overlape_time_ms = 5000
split_audio(audio_path)
processor = WhisperProcessor.from_pretrained(model_path)
model = WhisperForConditionalGeneration.from_pretrained(model_path)  
jp_file = open("subtitle_jp.txt",'w')
jp_file.close()
zh_file = open("subtitle_zh.txt",'w')
zh_file.close()
jp_file =open("subtitle_jp.txt",'a')
zh_file =open("subtitle_zh.txt",'a')

AsrAndTrans(temp_files,device,jp_file,zh_file)



#transcription = processor.batch_decode(predicted_ids,skip_special_tokens=True)
#print(transcription[0])