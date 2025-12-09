import pyaudio
import time
from queue import Queue
import webrtcvad
import wave
import threading
from funasr import AutoModel
import config
import numpy as np
from transformers import WhisperProcessor,WhisperForConditionalGeneration,TextStreamer
import noisereduce as nr
#import librosa
import torch
import torch.nn.functional as F
torch.set_default_dtype(torch.float32)
import numpy as np
import os
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
import json

'''
    from datasets import load_dataset

    from pydub import AudioSegment
    import math
    import tempfile
'''
with open('config_zh.json', 'r', encoding='utf-8') as file:
    config = json.load(file)

api_password = "nekoconer"
set_KWS = config["set_KWS"]
flag_KWS_used = config["flag_KWS_used"]
flag_sv_used = config["flag_sv_used"]
thred_sv = config["thred_sv"]
messages_input=config["trans_prompt"]
abstrct_messages_input = config["abstract_prompt"]
recog_model_path = config["recog_model_path"]
CAM_UPDATE = config["CAM_UPDATE"]
text_output=config["text_output_dir"]
AUDIO_SOURCE = config["AUDIO_SOURCE"]
CAM_THRESHOLD = config["CAM_THRESHOLD"]
LANGUAGES = config['LANGUAGES']
SLIENCE = 1 #是否处于录音状态
Save_Audio_file = config["Save_Audio_file"]
Save_recognize_file = config["Save_recognize_file"]
Save_translate_file = config["Save_translate_file"]
PROCESS_OUTPUT = config["PROCESS_OUTPUT"]
NOISE_MUL = config["NOISE_MUL"] #嘈杂环境噪音
ASR_OUTPUT = config["ASR_OUTPUT"]
AUDIO_RATE = config["AUDIO_RATE"]        # 音频采样率
AUDIO_CHANNELS = config["AUDIO_CHANNELS"]       # 单声道
CHUNK = config["CHUNK"]              # 音频块大小
VAD_MODE = config["VAD_MODE"]              # VAD 模式 (0-3, 数字越大越敏感)
OUTPUT_DIR = config["OUTPUT_DIR"]   # 输出目录
NO_SPEECH_THRESHOLD = config["NO_SPEECH_THRESHOLD"]   # 无效语音阈值，单位：秒
# 队列用于音频和视频同步缓存
audio_queue = Queue()

#线程池创建管理
trans_executor = ThreadPoolExecutor(max_workers=config["Thread_pool_sum"])
recoder_executor = ThreadPoolExecutor(max_workers=config["Thread_pool_sum"])

# 全局变量
last_active_time = time.time()
OUTPUT_DIR +=str(last_active_time)
micro_device = None
Noise_collect_round = 0
Noise_collect_time = 0
Noise_start_flag = 0
Noise_message_output = 0
Speaker_num = 0
speaker_emb = []
speaker_raw = []
Noise_num = 0
noise_emb=[]
noise_raw = []
asr_stream=0
audio_device = None
PyAudioChannel=None
stream = None
text_output +=str(last_active_time)
stream_mode = True
special_tokens=["<|startoftranscript|>","<|ja|>","<|zh|>","<|transcribe|>","<|notimestamps|>","<|endoftext|>"]
GPU = config["GPU"]
device = None

os.makedirs(OUTPUT_DIR)
os.makedirs(text_output)
jp_file = open(text_output+"/subtitle_jp.txt",'w')
jp_file.close()
zh_file = open(text_output+"/subtitle_zh.txt",'w')
zh_file.close()
Ab_file = open(text_output+"/subtitle_zh.txt",'w')
Ab_file.close()
EMB_file = open(text_output+"/EMB.txt",'w')
EMB_file.close()
jp_file =open(text_output+"/subtitle_jp.txt",'a',encoding="utf-8")
zh_file =open(text_output+"/subtitle_zh.txt",'a',encoding="utf-8")
Ab_file = open(text_output+"/abstract_zh.txt",'a',encoding="utf-8")
EMB_file = open(text_output+"/EMB.txt",'a',encoding="utf-8")

recording_active = 0
segments_to_save = []
saved_intervals = []
audio_buffer = []
last_vad_end_time = 0  # 上次保存的 VAD 有效段结束时间
audio_file_count = 0
audio_file_count_tmp = 0
audio_thread=None
Abstract_thread=None

#全局输出锁
chat_print_lock = threading.Lock()
ASR_print_lock = threading.Lock()

# 初始化 WebRTC VAD
vad = webrtcvad.Vad()
vad.set_mode(VAD_MODE)

# --- 唤醒词、声纹变量配置 ---
# set_KWS = "ni hao xiao qian"
# set_KWS = "shuo hua xiao qian"
flag_KWS = 0
#是否已经录入声纹
flag_sv_enroll = 0
#-----模型加载------
#声纹模型
set_SV_enroll=config["set_SV_enroll"]
#processor = WhisperProcessor.from_pretrained(recog_model_path)
#model = WhisperForConditionalGeneration.from_pretrained(recog_model_path) 
#model.to(device)
processor=None
model = None
cam_model_path = config["cam_model_path"]
cam_model = AutoModel(model=cam_model_path,disable_update=True)
#-----加载结束------
'''
    加密解密
'''
def xor_cipher(plaintext, key):
    ciphertext = ""
    for i in range(len(plaintext)):
        # 将明文的每个字符与密钥的对应字符进行异或运算
        ciphertext += chr(ord(plaintext[i]) ^ ord(key[i % len(key)]))
    return ciphertext

def API_platform_select(api):
    global client,API_model
    if(api.find("sk-")!=-1):
        client = OpenAI(api_key=api, base_url="https://api.deepseek.com")
        API_model = "deepseek-chat"
    elif(api.find("xai-")!=-1):
        client = OpenAI(api_key=api, base_url="https://api.x.ai/v1")
        API_model = "grok-2-latest"
    else:
        print("未能识别的API")
    print(API_model,end='')
API_config = config["API"]
#API设置
API_model = None
client = None
API = xor_cipher(API_config,api_password)
#client = OpenAI(api_key=config["dsAPI"], base_url="https://api.deepseek.com")
#client_grok = OpenAI(api_key=config["grokAPI"],base_url="https://api.x.ai/v1")


#清除声纹库方法
def clearCAM():
    global speaker_emb,speaker_raw,Speaker_num,Noise_collect_round,noise_emb,noise_raw,Noise_num,Noise_start_flag,Noise_message_output
    if(speaker_emb):
        speaker_emb.clear()
    if(speaker_raw):
        speaker_raw.clear()
    if(noise_raw):
        noise_raw.clear()
    if(noise_emb):
        noise_emb.clear()
    Speaker_num = 0
    Noise_num = 0
    Noise_collect_round = 0
    Noise_start_flag = 0
    Noise_message_output = 0

'''
    声音录制模块
    两个return值第一个表示说话人是否改变，第二个表示是否存入音频缓存,第三个表示当前说话人ID
'''
def cam_compare_add(raw_audio):
    global speaker_emb,Speaker_num,speaker_raw
    result = cam_model.generate(input=raw_audio, sampling_rate=AUDIO_RATE)
    emb_arry = result[0]['spk_embedding']
    #print(f'noise:{noise_emb}')
    #print(f'emb_arry:{emb_arry}')
    #噪音删除
    try:
        #print(f'emb:{len(noise_emb)} and noise_mul:{NOISE_MUL}')
        if(noise_emb and NOISE_MUL == 1):
            for i,noise_info in enumerate(noise_emb): 
                #print(f'noise_info:{noise_info}')
                cos_sim = F.cosine_similarity(emb_arry, noise_info)
                #print(f'cos_sim:{cos_sim}')
                if(cos_sim > CAM_THRESHOLD):
                    #print("Returning due to cos_sim condition")
                    return False,False,None
                else:
                    pass
    except Exception as e:
        print(f'噪音收集出现错误',end='')
    #print(f'speaker_emb')
    if(len(speaker_emb) == 0):
        if(AUDIO_SOURCE == 0 and NOISE_MUL == 0):
            speaker_emb.append(emb_arry)
            speaker_raw.append([raw_audio,1])
            Speaker_num=0
            #第一段默认为白噪声
            print("室内白噪音收集完毕",end='')
            return False,False,None
        else:
            speaker_emb.append(torch.zeros(192).to(device))
            speaker_raw.append([None,CAM_UPDATE])
            Speaker_num=0
            #第一段默认为白噪声
            return False,False,None
    else:
        #print("33333")
        cos_sim = F.cosine_similarity(emb_arry, speaker_emb[Speaker_num])
        #print(cos_sim)
        #说话人未发生改变
        if(cos_sim > CAM_THRESHOLD):
            if(speaker_raw[Speaker_num][1] < CAM_UPDATE):
                speaker_raw[Speaker_num][0] += raw_audio
                speaker_emb_result = cam_model.generate(input = speaker_raw[Speaker_num][0], sampling_rate = AUDIO_RATE)
                speaker_emb_new = speaker_emb_result[0]['spk_embedding']
                #print(speaker_emb_new)
                speaker_emb[Speaker_num] = speaker_emb_new
                speaker_raw[Speaker_num][1] += 1
            if(Speaker_num == 0):
                return False,False,None
            else:
                return False,True,Speaker_num
        #说话人发生改变
        else:
            for i,speaker_info in enumerate(speaker_emb):
                tmp_speaker = Speaker_num
                cos_sim = F.cosine_similarity(emb_arry, speaker_info)
                if(cos_sim > CAM_THRESHOLD):
                    Speaker_num = i
                    if(speaker_raw[Speaker_num][1] < CAM_UPDATE):
                        speaker_raw[Speaker_num][0] += raw_audio
                        speaker_emb_result = cam_model.generate(input = speaker_raw[Speaker_num][0], sampling_rate = AUDIO_RATE)
                        speaker_emb_new = speaker_emb_result[0]['spk_embedding']
                        speaker_emb[Speaker_num] = speaker_emb_new
                        speaker_raw[Speaker_num][1] += 1
                    #白噪声时进行过滤
                    if(i == 0):
                        return True,False,tmp_speaker
                    return True,True,tmp_speaker
            Speaker_num = len(speaker_emb)
            speaker_emb.append(emb_arry)
            speaker_raw.append([raw_audio,1])
            return True,True,tmp_speaker       
        
def check_vad_activity(audio_data):
    global last_vad_end_time
    # 将音频数据分块检测
    num, rate = 0, 0.35
    step = int(AUDIO_RATE * 0.02)  # 20ms 块大小
    flag_rate = round(rate * len(audio_data) // step)

    for i in range(0, len(audio_data), step):
        chunk = audio_data[i:i + step]
        if len(chunk) == step:
            if vad.is_speech(chunk, sample_rate=AUDIO_RATE):
                num += 1

    if num > flag_rate:
        #last_vad_end_time = time.time()
        return True
    return False

def save_audio_video(segments_to_save_tmp=None,tmp_speaker=None):

    global last_vad_end_time, saved_intervals

    # 全局变量，用于保存音频文件名计数
    global audio_file_count
    global flag_sv_enroll
    global set_SV_enroll
    # audio_output_path = f"{OUTPUT_DIR}/audio_0.wav"

    if not segments_to_save_tmp:
        return
    
    # 停止当前播放的音频
    '''if pygame.mixer.music.get_busy():
        pygame.mixer.music.stop()
        print("检测到新的有效音，已停止当前音频播放")
    '''
    # 获取有效段的时间范围
    start_time = segments_to_save_tmp[0][1]
    end_time = segments_to_save_tmp[-1][1]
    
    # 检查是否与之前的片段重叠
    if saved_intervals and saved_intervals[-1][1] >= start_time:
        print("当前片段与之前片段重叠，跳过保存")
        segments_to_save_tmp.clear()
        return
    
    # 保存音频
    if flag_sv_enroll:
        audio_output_path = f"{set_SV_enroll}/enroll_0.wav"
    else:
        audio_file_count += 1
        audio_output_path = f"{OUTPUT_DIR}/audio_{audio_file_count}.wav"
    audio_frames = [seg[0] for seg in segments_to_save_tmp]
    #接入翻译线程
    #trans_thread=Translate_Thread(audio_frames)
    #trans_thread.start()
    #trans_executor.submit(AsrAndTrans,audio_frames,tmp_speaker)
    AsrAndTrans(audio_frames,tmp_speaker)
    
    if(Save_Audio_file==1):
        wf = wave.open(audio_output_path, 'wb')
        wf.setnchannels(AUDIO_CHANNELS)
        wf.setsampwidth(2)  # 16-bit PCM
        #wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(AUDIO_RATE)
        wf.writeframes(b''.join(audio_frames))
        wf.close()
        #print(f"音频保存至 {audio_output_path}")

    # Inference()

    if flag_sv_enroll:
        text = "声纹注册完成！现在只有你可以命令我啦！"
        print(text)
        flag_sv_enroll = 0
        #system_introduction(text)
    else:
    # 使用线程执行推理
        #inference_thread = threading.Thread(target=Inference, args=(audio_output_path,))
        #inference_thread.start()
        
        # 记录保存的区间
        saved_intervals.append((start_time, end_time))
        
    # 清空缓冲区
    segments_to_save_tmp.clear()


def audio_recorder():
    global audio_queue, recording_active, last_active_time, segments_to_save, last_vad_end_time,SLIENCE,PyAudioChannel,stream,audio_buffer
    PyAudioChannel = pyaudio.PyAudio()
    if (AUDIO_SOURCE == 0):
        stream = PyAudioChannel.open(format=pyaudio.paInt16,
                        channels=AUDIO_CHANNELS,
                        rate=AUDIO_RATE,
                        input=True,
                        input_device_index=micro_device,
                        frames_per_buffer=CHUNK)
    elif(AUDIO_SOURCE==1):
        stream = PyAudioChannel.open(
                        format=pyaudio.paInt16,
                        channels=AUDIO_CHANNELS,
                        rate=AUDIO_RATE,
                        input=True,
                        input_device_index=audio_device,
                        frames_per_buffer=CHUNK)
    else:
        print("please check the AUDIO_SOURCE")
    
    
    
    #print("音频录制已开始")
    
    while recording_active:
        data = stream.read(CHUNK)
        audio_data = np.frombuffer(data, dtype=np.int16)
        reduced_noise = nr.reduce_noise(y=audio_data, sr=AUDIO_RATE)
        audio_buffer.append(reduced_noise)
        # 每 0.5 秒检测一次 VAD
        if len(audio_buffer) * CHUNK / AUDIO_RATE >= 0.5:
            raw_audio = b''.join(audio_buffer)
            recoder_executor.submit(recoder_voice_process,raw_audio)
        # 检查无效语音时间
            audio_buffer = []  # 清空缓冲区
        if time.time() - last_active_time > NO_SPEECH_THRESHOLD:
            # 检查是否需要保存       
            if segments_to_save and segments_to_save[-1][1] > last_vad_end_time:
                #print(f"num1{segments_to_save[-1][1]} and num2{last_vad_end_time}")
                #save_audio_video()
                trans_executor.submit(save_audio_video,segments_to_save.copy(),None)
                segments_to_save.clear()
                last_active_time = time.time()
            else:
                pass
                # print("无新增语音段，跳过保存")  
    stream.stop_stream()
    stream.close()
    PyAudioChannel.terminate()

def micro_inputouput_check():
    global micro_device,audio_device
    audio = pyaudio.PyAudio()
    device_count = audio.get_device_count()
    #print(f"device count: {device_count}")

    # get device info
    for i in range(device_count):
        device_info = audio.get_device_info_by_index(i)
        audio.get_device_info_by_index(i)['maxInputChannels']
        #print(f"device {i}: {device_info}")
        if(device_info["name"].find('立体声混音 (Realtek(R) Audio)')!=-1):
            audio_device = i
        elif(device_info["name"].find('麦克风阵列 (Realtek(R) Audio)')!=-1):
            micro_device = i
        elif(device_info["name"].find('麦克风 (Realtek(R) Audio)')!=-1):
            micro_device = i
#手动噪音收集函数     
def Noise_collect(raw_audio):
    global noise_emb,noise_raw,Noise_num,Noise_collect_time,Noise_start_flag
    if(Noise_start_flag == 0):
        Noise_start_flag = 1
        print("噪音收集中",end='')
    if(len(noise_emb)==0):
        Noise_collect_time = time.time()
    result = cam_model.generate(input=raw_audio, sampling_rate=AUDIO_RATE)
    emb_arry = result[0]['spk_embedding']
    if(len(noise_emb)==0):
        noise_emb.append(emb_arry)
        noise_raw.append([raw_audio,1])
        Noise_num = 0
    else:
        cos_sim = F.cosine_similarity(emb_arry, noise_emb[Noise_num])
        if(cos_sim > CAM_THRESHOLD):
            if(noise_raw[Noise_num][1] < CAM_UPDATE):
                noise_raw[Noise_num][0] += raw_audio
                noise_emb_result = cam_model.generate(input = noise_raw[Noise_num][0], sampling_rate = AUDIO_RATE)
                noise_emb_new = noise_emb_result[0]['spk_embedding']
                #print(speaker_emb_new)
                noise_emb[Noise_num] = noise_emb_new
                noise_raw[Noise_num][1] += 1
                return
        else:
            for i,noise_info in enumerate(noise_emb):
                cos_sim = F.cosine_similarity(emb_arry, noise_info)
                if(cos_sim > CAM_THRESHOLD):
                    Noise_num = i
                    if(noise_raw[Noise_num][1] < CAM_UPDATE):
                        noise_raw[Noise_num][0] += raw_audio
                        noise_emb_result = cam_model.generate(input = noise_raw[Noise_num][0], sampling_rate = AUDIO_RATE)
                        noise_emb_new = noise_emb_result[0]['spk_embedding']
                        noise_emb[Noise_num] = noise_emb_new
                        noise_raw[Noise_num][1] += 1
                    return
            Noise_num = len(noise_emb)
            noise_emb.append(emb_arry)
            noise_raw.append([raw_audio,1])
        
        
class CustomStreamerSelect(TextStreamer):
    '''
        重写输出方法检测输出语言，如果不是需要的语言则直接无视当前音频
    '''
    def __init__(self, tokenizer, skip_prompt = False, **decode_kwargs):
        super().__init__(tokenizer, skip_prompt, **decode_kwargs)
        #定义语言代码
        self.language = LANGUAGES
        #定义是否输出flag
        self.output = 1
    def on_finalized_text(self, text: str, stream_end: bool = False):
        """Prints the new text to stdout. If the stream is ending, also prints a newline."""
        if(text.find(f'<|startoftranscript|><|{self.language}|><|transcribe|><|notimestamps|>')):
            print(text, flush=True, end="" if not stream_end else None)
        #print(text, flush=True, end="" if not stream_end else None)

    pass       
       
def recoder_voice_process(raw_audio):
    global SLIENCE,last_active_time,segments_to_save,audio_buffer,Noise_collect_round,Noise_start_flag
    #print(f"num1{time.time()} and num2{last_active_time}")
    #print(f'time:{time.time()},noise_update:{Noise_collect_time}')
    vad_result = check_vad_activity(raw_audio)
    try:
        if(Noise_start_flag == 1):
            if(time.time() - Noise_collect_time>CAM_UPDATE*2 +1 ):
                Noise_collect_round = CAM_UPDATE*2
                Noise_start_flag = 2
                print("噪音收集完成",end='')
                segments_to_save.clear()
    except Exception as e:
        print(e)
    if vad_result:
        if(SLIENCE==1):
            SLIENCE=0
            print("语音活动中",end='') 
        if(Noise_collect_round< CAM_UPDATE*2 and NOISE_MUL == 1 ):
            Noise_collect_round += 1
            Noise_collect(raw_audio)
            if(Noise_collect_round == CAM_UPDATE*2):
                print("噪音收集完成",end='')
                Noise_start_flag = 2
                segments_to_save.clear()
            #噪音未触发达成，但是时间已经达到了时，进行跳出
        else:
            SpeakerChange,Audio_save,tmp_speaker = cam_compare_add(raw_audio)
            #print(SpeakerChange,Audio_save)
            if(SpeakerChange):
                #发生说话人改变，直接接入翻译线程
                save_audio_video(segments_to_save,tmp_speaker)
            if(Audio_save):
                last_active_time = time.time()
                segments_to_save.append((raw_audio, time.time()))
            
            
    else:
        if(Noise_start_flag == 0):
            Noise_start_flag == 1
        if(SLIENCE==0):
            print("未检测到音频",end='')
            SLIENCE=1
'''
    音频识别+翻译模块
'''
def get_response(client,messages,stream=False):
    try:
        response=client.chat.completions.create(
            model=API_model,
            messages=messages,
            stream = stream
        )
        return response
    except Exception as e:
        print(f"It have some wrong in API connection:{str(e)}")
        return None

'''def get_response_grok(client,messages,stream=False):
    try:
        response=client.chat.completions.create(
            model="grok-2-latest",
            messages=messages,
            stream = stream
        )
        return response
    except Exception as e:
        print(f"It have some wrong in API connection:{str(e)}")
        return None
'''

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

def chat(audio_text,file=None,tmp_speaker=None):
    global messages_input
    question = audio_text
    messages_total = []
    messages_total.append(messages_input)
    messages_total.append({"role":"user","content":question})
    #response = get_response(client,messages_total,stream_mode)
    response = get_response(client,messages_total,stream_mode)
    Chat_speaker_num = tmp_speaker if tmp_speaker else Speaker_num
    temp_text =""
    if response:
        #with chat_print_lock:
            print(f"说话人{Chat_speaker_num}:",end='')
            temp_text += f"说话人{Chat_speaker_num}:"
            #print(f"翻以前的文本:{audio_text}")
            if stream_mode:
                #print("翻译后的文本:",end='')
                for chunk in response:
                    #print(chunk)
                    if(check_chunk(chunk)):
                        chunk_message=chunk.choices[0].delta.content
                        temp_text+=chunk_message
                        print(chunk_message,end='',flush=True)               
                print()
            else:
                print(f"翻译后的文本:{response.choices[0].message.content}")
            #print("_______________")
            #append_message = get_stream_response(response) if stream_mode else response.choices[0].message.content 
            #print("!!!!!!!!!!!!")
            #print(temp_text)
            if(Save_translate_file==1):
                print(temp_text,file=file)
            file.flush()
            
            #messages_input.append({"role": "assistant", "content":append_message})
    else:
        print("There have no response, please check the input or code for message")

def AsrAndTrans(audio_frames,tmp_speaker=None):
    global model,processor,device,jp_file,zh_file
    audio_np = np.frombuffer(b"".join(audio_frames), dtype=np.int16)
    audio_np = audio_np.astype(np.float32) / 32768.0  # 归一化到 [-1, 1]
    '''
        在TextStreamer源码中直接skip_special_token来保证流式输出中没有特殊字符
    '''
    streamer = TextStreamer(processor.tokenizer)
    #streamer = CustomStreamerSelect(processor.tokenizer)
    #print(audio)
    input= processor(audio_np,sampling_rate=AUDIO_RATE,return_tensor="pt")
    input_feature = input.input_features
    if isinstance(input_feature, np.ndarray):
        input_feature = torch.from_numpy(input_feature)
    input_feature=input_feature.to(device)
    #print('111111')
    #with ASR_print_lock:
    #非指定的语言代码就直接pass
    '''try:
        with torch.no_grad():
            predicted_ids = model.generate(input_feature)

            # 将预测的标识符转换为语言代码
        predicted_language_code = processor.tokenizer.decode(predicted_ids[0], skip_special_tokens=False)
        print(f"检测到的语言：{predicted_language_code}")
    except Exception as e:
        print(e)'''
    if(ASR_OUTPUT):
        if(tmp_speaker):
            speak_one = tmp_speaker
            print(f"说话人{tmp_speaker}:",end='')
        else:
            speak_one = Speaker_num
            print(f"说话人{Speaker_num}:",end='')
    predicted_ids = model.generate(input_feature,
                                streamer=streamer if ASR_OUTPUT else None,
                                temperature=0.95,
                                do_sample=True,
                                top_k=6,
                                top_p=0.8,
                                repetition_penalty=1.05,
                                pad_token_id=processor.tokenizer.eos_token_id,
                                no_repeat_ngram_size=5
                                )
    #print('22222')
    for chunk in predicted_ids:
        #print(chunk)
        #print("_______________")
        transcription = processor.batch_decode(predicted_ids,skip_special_tokens=True)
        #print(transcription[0])
        if PROCESS_OUTPUT:
            chat(transcription[0],zh_file,tmp_speaker)
        if(Save_recognize_file==1):
            print(f"说话人{speak_one}:{transcription[0]}",file =jp_file)
        jp_file.flush()

def closeStep():
    global audio_thread,trans_executor,jp_file,zh_file,model,recording_active
    recording_active= 0
    audio_thread.join()
    del model
    torch.cuda.empty_cache()
    # video_thread.join()
    print("模型已卸载",end='')
    
'''
    大纲总结方法
'''
def Abstract_CN_text():
    global Ab_file
    temp_cn = None
    temp_text=[]
    with open(text_output+"/subtitle_zh.txt",'r',encoding="utf-8") as file_cn:
        temp_cn = file_cn.read()
    #如果cn文件里没有文本说明没有加入cn文件
    if(PROCESS_OUTPUT == 0):
        with open(text_output+"/subtitle_jp.txt",'r',encoding="utf-8") as file_cn:
            temp_cn = file_cn.read()
    messages_abstract_total=[]
    messages_abstract_total.append(abstrct_messages_input)
    messages_abstract_total.append({"role":"user","content":temp_cn})
    #response = get_response(client,messages_abstract_total,stream_mode)
    response = get_response(client,messages_abstract_total,stream_mode)
    if response:
    #print(f"翻以前的文本:{audio_text}")
        if stream_mode:
            #print("翻译后的文本:",end='')
            for chunk in response:
                #print(chunk)
                if(check_chunk(chunk)):
                    chunk_message=chunk.choices[0].delta.content
                    temp_text.append(chunk_message)
                    print(chunk_message,end='',flush=True)               
            print()
            Abstract_messages_text = "".join(temp_text)
            print(Abstract_messages_text,file=Ab_file)
        else:
            print(f"翻译后的文本:{response.choices[0].message.content}")

def Abstract_CN_text_thread():
    global Abstract_thread       
    Abstract_thread = threading.Thread(target=Abstract_CN_text)
    Abstract_thread.start()
    
def ProjectOver():
    global recording_active
    jp_file.close()
    zh_file.close()
    Ab_file.close()
    recording_active=0
    if(audio_thread):
        audio_thread.join()
    if(Abstract_thread):
        Abstract_thread.join()
    trans_executor.shutdown()
    recoder_executor.shutdown()
    stream.stop_stream()
    stream.close()
    PyAudioChannel.terminate()

def recoderAgain():
    global audio_thread
    audio_thread = threading.Thread(target=audio_recorder)
    # video_thread = threading.Thread(target=video_recorder)
    audio_thread.start()
def startStep():
    global audio_thread,model,processor,device,API
    if(GPU):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        #print(device)
    else:
        device = "cpu"
    #print(device)
    processor = WhisperProcessor.from_pretrained(recog_model_path)
    model = WhisperForConditionalGeneration.from_pretrained(recog_model_path)
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language="zh", task="transcribe"
    )
 
    model.to(device)
    print("模型加载完毕",end='')
    API_platform_select(API)
    micro_inputouput_check()
    audio_thread = threading.Thread(target=audio_recorder)
    # video_thread = threading.Thread(target=video_recorder)
    audio_thread.start()
    # video_thread.start()
    
'''class Translate_Thread(threading.Thread):
    def __init__(self,audio_frames):
        super().__init__()
        self.audio_frames=audio_frames
    def run(self):
        global model,processor,device,jp_file,zh_file
        audio_np = np.frombuffer(b"".join(self.audio_frames), dtype=np.int16)
        audio_np = audio_np.astype(np.float32) / 32768.0  # 归一化到 [-1, 1]
        sr =AUDIO_RATE
        model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
            language="ja", task="transcribe"
        )
    
        #在TextStreamer源码中直接skip_special_token来保证流式输出中没有特殊字符
        streamer = TextStreamer(processor.tokenizer)
        #streamer =CustomTextStreamer(processor.tokenizer)
        #print(audio)
        input= processor(audio_np,sampling_rate=sr,return_tensor="pt")
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
'''    
        
if __name__ == '__main__':
    try:
        if(GPU):
            device = "cuda" if torch.cuda.is_available() else "cpu"
            #print(device)
        else:
            device = "cpu"
        #print(device)
        processor = WhisperProcessor.from_pretrained(recog_model_path)
        model = WhisperForConditionalGeneration.from_pretrained(recog_model_path) 
        model.to(device)
        print("模型加载完毕",end='')
        API_platform_select(API)
        micro_inputouput_check()
        audio_recorder()
    except Exception as e:
        print(f"主线程捕获到异常: {e}")