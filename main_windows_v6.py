import recoder_translate_v6
import sys,json
from PyQt6.QtWidgets import (QWidget,QMenu,QInputDialog,
    QPushButton, QApplication,QMessageBox,QMainWindow,QTextEdit,QLabel,QHBoxLayout,QGridLayout,QVBoxLayout)
from PyQt6.QtGui import QAction
from PyQt6.QtCore import pyqtSignal,QObject

with open('config_zh.json', 'r', encoding='utf-8') as file:
    config = json.load(file)
WindowsLag=0 # 0：CN 1:JP
abstract_edit_text = None
#基础菜单
BasicMenuCN=[
    '基础设置',
    '音频来源',
    '模型精度',
    '模型挂载选择',
    '显示文本',
    '输出文本控制'
]
#输出文本菜单
OutputTextSet=[
    '输出音频识别文本',
    '输出处理后文本'
]
#文件保存菜单
FileSaveCN=[
    '文件保存',
    '选择需要保存的文件',
    '音频文件',
    '日文识别文本',
    '中文翻译文本'
]
#音频子菜单
AudioSourceCN=[
    '麦克风',
    '扬声器音频',
    '硬盘音频文件'
]
#精度选择子菜单
TransAccCN=[
    '上下文关联',
    '前置噪声收集'
]
#屏幕显示文本子菜单
ShowLanText=[
    '麦克风识别文本',
    '媒体翻译文本'
]
#语言菜单栏
LANMenuCN=[
    '界面设置',
    '界面语言选择',
    '开发者信息'
]
#界面语言子菜单栏
WindowsLAN=[
    'CN',
    'JP开发中',
    'EN开发中'
]
#按钮文字菜单栏
TopButtonCN=[
    '加载模型',
    '卸载模型',
    '开始监听',
    '停止监听',
    '生成大纲',
    '清空声纹库'
]
#噪音提示词文本库
Noise_text_output = [
    '噪音收集中',
    '噪音收集完成',
    '噪音未收集',
    '噪音收集出现错误',
    '室内白噪音收集完毕'
]
API_model_list=[
    "deepseek-chat",
    "grok-2-latest"
]

class EmittingStream(QObject):
    text_written = pyqtSignal(str)

    def write(self, text):
        self.text_written.emit(text)

    def flush(self):
        pass

class TextEditorWindow(QMainWindow):
        def __init__(self, parent=None):
            super().__init__(parent)
            self.setWindowTitle("大纲")
            self.setGeometry(150, 150, 300, 200)
            self.center()
            # 创建 QTextEdit 组件
            self.abstract_edit = QTextEdit(self)
            self.setCentralWidget(self.abstract_edit)
        def center(self):
            qr = self.frameGeometry()
            cp = self.screen().availableGeometry().center()
            qr.moveCenter(cp)
            self.move(qr.topLeft())
        def closeEvent(self, event):
            global abstract_edit_text
            reply = QMessageBox.question(self, 'Message',
                        "Are you sure to quit?", QMessageBox.StandardButton.Yes |
                        QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
            
            if reply == QMessageBox.StandardButton.Yes:
                abstract_edit_text = None
                event.accept()
            else:
                event.ignore() 

class Main_Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
    def initUI(self):
        #设置窗口大小
        self.setGeometry(300,300,600,350)
        #调用窗口居中
        self.center()
        #设置窗口菜单
        self.menuCreate()
        #设置开发者信息
        self.setStatusBar()
        #设置布局
        window = QWidget()
        grid = QGridLayout(window)
        grid.setSpacing(5) #设置间距为多少
        #获取api
        self.APIbtn = QPushButton('SetAPI', self)   
        #self.APIbtn.move(20, 20)
        self.APIbtn.clicked.connect(self.showDialog)
        self.LoadModelBtn = QPushButton(TopButtonCN[0], self)
        self.StopRecodeBtn = QPushButton(TopButtonCN[2], self)
        self.StopRecodeBtn.setEnabled(False)
        self.AbstractBtn = QPushButton(TopButtonCN[4])
        self.CAMClearBtn = QPushButton(TopButtonCN[5],self)
        self.LoadModelBtn.clicked.connect(self.AudioSelectMenu)
        self.StopRecodeBtn.clicked.connect(self.AudioSelectMenu)
        self.AbstractBtn.clicked.connect(self.AudioSelectMenu)
        self.CAMClearBtn.clicked.connect(self.AudioSelectMenu)
        self.AbstractBtn.setEnabled(True)
        self.CAMClearBtn.setEnabled(False)
        self.Mirco_text = QLabel(ShowLanText[0])
        self.Media_text = QLabel(ShowLanText[1])
        self.Media_text.setVisible(False)
        btn_bar = QHBoxLayout()
        btn_bar.addStretch(1)
        btn_bar.addWidget(self.APIbtn)
        btn_bar.addWidget(self.LoadModelBtn)
        btn_bar.addWidget(self.StopRecodeBtn)
        btn_bar.addWidget(self.AbstractBtn)
        btn_bar.addWidget(self.CAMClearBtn)
        grid.addLayout(btn_bar,0,0,1,2)
        ##grid.addWidget(self.LoadModelBtn,0,1)
        #grid.addWidget(self.StopRecodeBtn,0,2)
        # 创建 QTextEdit 组件用于显示输出
        #self.text_edit.setReadOnly(True)
        grid.addWidget(self.Mirco_text,2,0)
        self.text_edit_Micro = QTextEdit()
        self.text_edit_Media = QTextEdit()
        self.text_edit_Micro.setReadOnly(True)
        self.text_edit_Media.setReadOnly(True)
        grid.addWidget(self.text_edit_Micro,2,1,3,1)
        grid.addWidget(self.Media_text,5,0)
        grid.addWidget(self.text_edit_Media,5,1,3,1)
        # 设置 QSizePolicy
        size_policy = self.text_edit_Micro.sizePolicy()
        size_policy.setHorizontalStretch(1)
        size_policy.setVerticalStretch(1)
        self.text_edit_Micro.setSizePolicy(size_policy)
        self.text_edit_Media.setVisible(False)
        # 创建输出重定向器并重定向标准输出和标准错误
        # 重定向 sys.stdout
        self.emitting_stream = EmittingStream()
        self.emitting_stream.text_written.connect(self.update_text_edit)
        sys.stdout = self.emitting_stream
        
        self.setCentralWidget(window)
        self.show()
        
    #消息提示框
    def show_custom_message(self):
        msg_box = QMessageBox(self)
        msg_box.setIcon(QMessageBox.Icon.Warning)
        msg_box.setText(f'噪音正在收集中，请勿发出人声，防止被当作噪音消音。收集噪音需耗时{config["CAM_UPDATE"]*0.5*2}S')
        msg_box.setWindowTitle('噪音收集提示！')
        msg_box.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        msg_box.setDefaultButton(QMessageBox.StandardButton.No)
        reply = msg_box.exec() 
    #选择菜单逻辑
    def AudioSelectMenu(self,checked):
        #获取发送信号的对象
        select_action = self.sender()
        select_name = select_action.text()
        #print(select_name)
        #音频来源流程
        if(select_name in AudioSourceCN):
            for action in self.audio_action:
                action.setChecked(action==select_action)
            for i,name in enumerate(AudioSourceCN):
                if select_name==name:
                    config["AUDIO_SOURCE"]=i
                    recoder_translate_v6.AUDIO_SOURCE=i
        #上下文关联流程
        elif(select_name in TransAccCN):
            if(select_name == TransAccCN[1]):
                if(checked):
                    config['NOISE_MUL'] = 1
                    recoder_translate_v6.NOISE_MUL=1
                else:
                    config['NOISE_MUL'] = 0
                    recoder_translate_v6.NOISE_MUL = 0   
            else:
                pass
        #首行按钮逻辑
        if(select_name in TopButtonCN):
            if(select_name==TopButtonCN[0]):
                #加载模型
                recoder_translate_v6.startStep()
                self.GpuSelect.setEnabled(False)
                self.AbstractBtn.setEnabled(False)
                self.APIbtn.setEnabled(False)
                self.StopRecodeBtn.setEnabled(True)
                self.Noise_collect_select.setEnabled(False)
                for action in self.audio_action:
                    action.setEnabled(False)
                self.LoadModelBtn.setText(TopButtonCN[1])
            elif(select_name==TopButtonCN[1]):
                #释放模型
                recoder_translate_v6.closeStep()
                self.GpuSelect.setEnabled(True)
                self.APIbtn.setEnabled(True)
                self.Noise_collect_select.setEnabled(True)
                for action in self.audio_action:
                    action.setEnabled(True)
                self.AbstractBtn.setEnabled(True)
                self.StopRecodeBtn.setEnabled(False)
                self.StopRecodeBtn.setText(TopButtonCN[2])
                self.LoadModelBtn.setText(TopButtonCN[0])
                
            elif(select_name==TopButtonCN[2]):
                #开始监听
                recoder_translate_v6.recording_active=1
                recoder_translate_v6.recoderAgain()
                if(recoder_translate_v6.NOISE_MUL==1 and len(recoder_translate_v6.noise_emb)==0):
                    self.show_custom_message()
                self.CAMClearBtn.setEnabled(False)
                self.AbstractBtn.setEnabled(False)
                self.LoadModelBtn.setEnabled(False)
                self.StopRecodeBtn.setText(TopButtonCN[3])
            elif(select_name ==TopButtonCN[3]):
                #停止监听
                self.AbstractBtn.setEnabled(True)
                self.CAMClearBtn.setEnabled(True)
                self.LoadModelBtn.setEnabled(True)
                recoder_translate_v6.recording_active=0
                self.StopRecodeBtn.setText(TopButtonCN[2])
            #大纲生成
            elif(select_name == TopButtonCN[4]):
                self.showAbstract()
                recoder_translate_v6.Abstract_CN_text_thread()
            #声纹库重置
            elif(select_name ==TopButtonCN[5]):
                recoder_translate_v6.clearCAM()
                self.Noise_info.setText("噪音未收集")
        #GPU选择
        elif(select_name == "GPU"):
            if(checked):
                config["GPU"]=1
                recoder_translate_v6.GPU =1
            else:
                config["GPU"]=0
                recoder_translate_v6.GPU =0
        #文件保存设置
        elif(select_name in FileSaveCN):
            if(select_name==FileSaveCN[2]):
                if(checked == False):
                    config["Save_Audio_file"]=0
                    recoder_translate_v6.Save_Audio_file=0
                else:
                    config["Save_Audio_file"]=1
                    recoder_translate_v6.Save_Audio_file=1
            elif (select_name == FileSaveCN[3]):
                if(checked == False):
                    config["Save_recognize_file"]=0
                    recoder_translate_v6.Save_recognize_file=0
                else:        
                    config["Save_recognize_file"]=1
                    recoder_translate_v6.Save_recognize_file=1
            elif(select_name == FileSaveCN[4]):
                if(checked == False):
                    config["Save_translate_file"]=0
                    recoder_translate_v6.Save_translate_file=0
                else:
                    config["Save_translate_file"]=1
                    recoder_translate_v6.Save_translate_file=1
        #文本显示设置
        elif(select_name in ShowLanText):
            if(select_name == ShowLanText[0]):
                if(checked==False):
                    self.Mirco_text.setVisible(False)
                    self.text_edit_Micro.setVisible(False)
                else:
                    self.Mirco_text.setVisible(True)
                    self.text_edit_Micro.setVisible(True)
            elif(select_name == ShowLanText[1]):
                if(checked==False):
                    self.Media_text.setVisible(False)
                    self.text_edit_Media.setVisible(False)
                else:
                    self.Media_text.setVisible(True)
                    self.text_edit_Media.setVisible(True)
            #print(select_name,checked)
        #输出文本选择
        elif(select_name in OutputTextSet):
            if(select_name == OutputTextSet[0]):
                if(checked == False):
                    config["ASR_OUTPUT"] = 0
                    recoder_translate_v6.ASR_OUTPUT = 0
                else:
                    config["ASR_OUTPUT"] = 1
                    recoder_translate_v6.ASR_OUTPUT = 1
            elif(select_name == OutputTextSet[1]):
                if(checked == False):
                    config["PROCESS_OUTPUT"] = 0
                    recoder_translate_v6.PROCESS_OUTPUT = 0
                else:
                    config["PROCESS_OUTPUT"] = 1
                    recoder_translate_v6.PROCESS_OUTPUT = 1
        elif(select_name == '显示开发者信息'):
            if(checked==False):
                self.CreatorMessage.setVisible(False)
            else:
                self.CreatorMessage.setVisible(True)
      
    def update_text_edit(self, text):
        if(abstract_edit_text):
            cursor = abstract_edit_text.textCursor()
            cursor.movePosition(cursor.MoveOperation.End)
            abstract_edit_text.insertPlainText(text)
            abstract_edit_text.setTextCursor(cursor)
            abstract_edit_text.ensureCursorVisible()
            cursor.movePosition(cursor.MoveOperation.End)
        elif(text =="语音活动中"):
            self.Audio_info.setText(text)
        elif(text=="未检测到音频"):
            self.Audio_info.setText(text)
        elif(text == "模型已卸载"):
            self.Model_info.setText(text)
        elif(text == "模型加载完毕"):
            self.Model_info.setText(text)
        elif(text in Noise_text_output):
            self.Noise_info.setText(text)
        elif(text in API_model_list):
            self.API_Model_info.setText(text)
        elif(text ==""):
            pass
        else:   
            if(config["AUDIO_SOURCE"]==0):
                cursor = self.text_edit_Micro.textCursor()
                cursor.movePosition(cursor.MoveOperation.End)
                self.text_edit_Micro.insertPlainText(text)
                self.text_edit_Micro.setTextCursor(cursor)
                self.text_edit_Micro.ensureCursorVisible()
                cursor.movePosition(cursor.MoveOperation.End)
            else:
                cursor = self.text_edit_Media.textCursor()
                cursor.movePosition(cursor.MoveOperation.End)
                self.text_edit_Media.insertPlainText(text)
                self.text_edit_Media.setTextCursor(cursor)
                self.text_edit_Media.ensureCursorVisible()
                cursor.movePosition(cursor.MoveOperation.End)
                

    def closeEvent(self, event):

        reply = QMessageBox.question(self, 'Message',
                    "Are you sure to quit?", QMessageBox.StandardButton.Yes |
                    QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
        
        if reply == QMessageBox.StandardButton.Yes:
            sys.stdout = sys.__stdout__
            with open('config_zh.json', 'w',encoding="utf-8") as f:
                json.dump(config, f, indent=4,ensure_ascii=False)
            recoder_translate_v6.ProjectOver()
            event.accept()
        else:
            event.ignore() 
      
    def setStatusBar(self):
        status_bar =self.statusBar()
        label1 = QLabel('状态信息：')
        self.CreatorMessage = QLabel('Created by nekoconer@github.com')
        status_bar.addWidget(label1)
        status_bar.addPermanentWidget(self.CreatorMessage)
        
        FirstRow_status_widget = QWidget()
        FirstRow_status_layout = QHBoxLayout()
        FirstRow_status_layout.setSpacing(5)
        FirstRow_status_layout.setContentsMargins(5, 0, 5, 0)
        FirstRow_status_widget.setLayout(FirstRow_status_layout)
        
        SecondRow_status_widget = QWidget()
        SecondRow_status_layout = QHBoxLayout()
        SecondRow_status_layout.setSpacing(5)
        SecondRow_status_layout.setContentsMargins(5, 0, 5, 0)
        SecondRow_status_widget.setLayout(SecondRow_status_layout)
        
        status_widget = QWidget()
        status_layout = QVBoxLayout()
        status_layout.setSpacing(-5)  # 两行之间的间距
        
        status_widget.setLayout(status_layout)

        Model_info_title = QLabel('model:')
        self.Model_info = QLabel('暂未加载模型')
        Audio_info_title = QLabel('Audio:')
        self.Audio_info = QLabel('未检测到声音')
        API_Model_info_title = QLabel('API_model:')
        self.API_Model_info = QLabel('暂未识别API')
        Noise_info_title = QLabel('Noise:')
        self.Noise_info = QLabel('噪音未收集')
        FirstRow_status_layout.addWidget(Audio_info_title)
        FirstRow_status_layout.addWidget(self.Audio_info)
        FirstRow_status_layout.addWidget(Noise_info_title)
        FirstRow_status_layout.addWidget(self.Noise_info)
        SecondRow_status_layout.addWidget(Model_info_title)
        SecondRow_status_layout.addWidget(self.Model_info)
        SecondRow_status_layout.addWidget(API_Model_info_title)
        SecondRow_status_layout.addWidget(self.API_Model_info)
        
        status_layout.addWidget(FirstRow_status_widget)
        status_layout.addWidget(SecondRow_status_widget)
        status_bar.addWidget(status_widget)     
        
    def showDialog(self):
        text, ok = QInputDialog.getText(self, 'Input API-key','Enter your API(only for deepseek and grok):')
        if ok and text:
            if(text.find("ds-") or text.find("xai-")):
                recoder_translate_v6.API_platform_select(text)
                recoder_translate_v6.API = text
                config["API"] = recoder_translate_v6.xor_cipher(text,recoder_translate_v6.api_password)
    
    def showAbstract(self):
        global abstract_edit_text
        self.text_editor_window = TextEditorWindow(self)
        abstract_edit_text = self.text_editor_window.abstract_edit
        self.text_editor_window.show()
        
    #居中显示
    def center(self):
        qr = self.frameGeometry()
        cp = self.screen().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())
    
    def menuCreate(self):
        self.setWindowTitle("Translation")
        #添加菜单栏
        menubar = self.menuBar()
        #设计基础菜单栏
        Basic_file = menubar.addMenu(BasicMenuCN[0])
        
        #音频选择栏
        self.audio_action =[]
        Audio_file= QMenu(BasicMenuCN[1],self)
        Basic_file.addMenu(Audio_file)
        Micro_select=QAction(AudioSourceCN[0],self,checkable=True)
        audio_selct= QAction(AudioSourceCN[1],self,checkable=True)
        Micro_file = QAction(AudioSourceCN[2],self,checkable=True)
        
        
        self.audio_action.append(Micro_select)
        self.audio_action.append(audio_selct)
        self.audio_action.append(Micro_file)
        self.audio_action[config["AUDIO_SOURCE"]].setChecked(True)
        Audio_file.addActions(self.audio_action)
        Micro_select.triggered.connect(self.AudioSelectMenu)
        audio_selct.triggered.connect(self.AudioSelectMenu)
        Micro_file.triggered.connect(self.AudioSelectMenu)
        #选择模型精度
        Trans_file = QMenu(BasicMenuCN[2],self)
        Basic_file.addMenu(Trans_file)
        Trans_select=QAction(TransAccCN[0],self,checkable=True)
        self.Noise_collect_select= QAction(TransAccCN[1],self,checkable=True)
        Trans_file.addAction(Trans_select)
        Trans_file.addAction(self.Noise_collect_select)
        Trans_select.triggered.connect(self.AudioSelectMenu)
        self.Noise_collect_select.triggered.connect(self.AudioSelectMenu)
        self.Noise_collect_select.setChecked(config['NOISE_MUL'])
        
        #选择模型挂载CPU或者GPU
        ModelPut = QMenu(BasicMenuCN[3],self)
        Basic_file.addMenu(ModelPut)
        self.GpuSelect=QAction("GPU",self,checkable=True)
        self.GpuSelect.setChecked(config["GPU"])
        self.GpuSelect.setStatusTip('Please dont change this option when the model is running')
        ModelPut.addAction(self.GpuSelect)
        self.GpuSelect.triggered.connect(self.AudioSelectMenu)
        
        #是否保存文件
        FileSaveMenu = menubar.addMenu(FileSaveCN[0])
        SelectFileSave = QMenu(FileSaveCN[1],self)
        FileSaveMenu.addMenu(SelectFileSave)
        FileAudioSaveSelect=QAction(FileSaveCN[2],self,checkable=True)
        FileJPTextSaveSelect=QAction(FileSaveCN[3],self,checkable=True)
        FileCNTextSaveSelect=QAction(FileSaveCN[4],self,checkable=True)
        FileAudioSaveSelect.setChecked(config["Save_Audio_file"])
        FileJPTextSaveSelect.setChecked(config["Save_recognize_file"])
        FileCNTextSaveSelect.setChecked(config["Save_translate_file"])
        FileAudioSaveSelect.triggered.connect(self.AudioSelectMenu)
        FileCNTextSaveSelect.triggered.connect(self.AudioSelectMenu)
        FileJPTextSaveSelect.triggered.connect(self.AudioSelectMenu)
        SelectFileSave.addAction(FileAudioSaveSelect)
        SelectFileSave.addAction(FileCNTextSaveSelect)
        SelectFileSave.addAction(FileJPTextSaveSelect)
        
        #显示文本选择
        ShowLanTextMenu = QMenu(BasicMenuCN[4],self)
        Basic_file.addMenu(ShowLanTextMenu)
        ShowLanMicro = QAction(ShowLanText[0],self,checkable=True)
        ShowLanMedia = QAction(ShowLanText[1],self,checkable=True)
        ShowLanMicro.triggered.connect(self.AudioSelectMenu)
        ShowLanMedia.triggered.connect(self.AudioSelectMenu)
        ShowLanTextMenu.addAction(ShowLanMicro)
        ShowLanTextMenu.addAction(ShowLanMedia)
        ShowLanMedia.setChecked(False)
        ShowLanMicro.setChecked(True)
        
        #输出文本选择
        OutputTextMenu = QMenu(BasicMenuCN[5],self)
        Basic_file.addMenu(OutputTextMenu)
        ShowASRText = QAction(OutputTextSet[0],self,checkable=True)
        ShowProcessText = QAction(OutputTextSet[1],self,checkable=True)
        ShowProcessText.triggered.connect(self.AudioSelectMenu)
        ShowASRText.triggered.connect(self.AudioSelectMenu)
        OutputTextMenu.addAction(ShowASRText)
        OutputTextMenu.addAction(ShowProcessText)
        ShowASRText.setChecked(config["ASR_OUTPUT"])
        ShowProcessText.setChecked(config["PROCESS_OUTPUT"])
        #界面语言选择
        WindowsLanSelect = menubar.addMenu(LANMenuCN[0])
        WindowsLanTotal = QMenu(LANMenuCN[1],self)
        WindowsLanCreator =QMenu(LANMenuCN[2],self)
        self.WindowsLanSelectAction=[]
        WindowsLanCN = QAction(WindowsLAN[0],self,checkable=True)
        WindowsLanCN.setChecked(True)
        WindowsLanJP = QAction(WindowsLAN[1],self,checkable=False)
        WindowsLanEN = QAction(WindowsLAN[2],self,checkable=False)
        self.WindowsLanSelectAction.append(WindowsLanCN)
        self.WindowsLanSelectAction.append(WindowsLanJP)
        self.WindowsLanSelectAction.append(WindowsLanEN)
        WindowsLanTotal.addActions(self.WindowsLanSelectAction)
        WindowsLanSelect.addMenu(WindowsLanTotal)
        ShowCreator = QAction('显示开发者信息',self,checkable=True)
        ShowCreator.setChecked(True)
        WindowsLanSelect.addMenu(WindowsLanCreator)
        WindowsLanCreator.addAction(ShowCreator)
        ShowCreator.triggered.connect(self.AudioSelectMenu)

def main():
    app = QApplication(sys.argv)
    Main = Main_Window()
    sys.exit(app.exec())
if __name__== "__main__":
    main()      