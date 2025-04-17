import librosa
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
import pandas as pd
import os
import seaborn as sns
import glob

emotion=[]
gender=[]
path=[]

for file in glob.glob("data/Actor_*/*.wav"):
    path.append(file)
    basename=os.path.basename(file)
    part=basename.split('-')
    emotion.append(int(part[2]))
    temp=int((part[6].split('.'))[0])
    if temp%2==0:
        temp="male"
    else:
        temp="female"
    gender.append(temp)

RAV_df=pd.DataFrame(emotion)
RAV_df=RAV_df.replace({1:'neutral',2:'calm',3:'happy',4:'sad',5:'angry',6:'fear',7:'disguest',8:'surprise'})
RAV_df=pd.concat([pd.DataFrame(gender),RAV_df],axis=1)
RAV_df.columns=['gender','emotion']
RAV_df['labels']=RAV_df.gender+'_'+RAV_df.emotion
RAV_df['souce']='RAVDESS'
RAV_df=pd.concat([RAV_df,pd.DataFrame(path,columns=['path'])],axis=1)
RAV_df=RAV_df.drop(['gender','emotion'],axis=1)
#print(RAV_df.labels.value_counts())

plt.figure(figsize=(12,5))
plt.title('Count of emotions')
sns.countplot(RAV_df.labels,palette='cool')
plt.ylabel('Emotions')
plt.xlabel('Counts')
plt.yticks(rotation=45)
sns.despine(top=True,right=True,left=False,bottom=False)
plt.show()

#Many comparisons can be done. For example-Comparing mel spectrograms of male and femaule neutral audio clips

f1=r"C:\Users\jagso\Documents\Probe\Emotion\data\Actor_01\03-01-01-01-01-01-01.wav"
f2=r"C:\Users\jagso\Documents\Probe\Emotion\data\Actor_14\03-01-01-01-01-01-14.wav"

data1,sr1=librosa.load(f1)
data2,sr2=librosa.load(f2)

plt.figure(figsize=(15,5))

plt.subplot(1,2,1)
spectrogram=librosa.feature.melspectrogram(y=data1,sr=sr1,n_mels=128,fmax=8000)
spectrogram=librosa.power_to_db(spectrogram)
librosa.display.specshow(spectrogram,y_axis='mel',fmax=8000,x_axis='time')
plt.title('Mel spectrogram-Male-neutral')
plt.colorbar(format="%+2.0f dB")

plt.subplot(1,2,2)
spectrogram=librosa.feature.melspectrogram(y=data2,sr=sr2,n_mels=128,fmax=8000)
spectrogram=librosa.power_to_db(spectrogram)
librosa.display.specshow(spectrogram,y_axis='mel',fmax=8000,x_axis='time')
plt.title('Mel spectrogram-Female-neutral')
plt.colorbar(format="%+2.0f dB")

plt.show()

#Comparing waveplots for happy and sad

f1=r"C:\Users\jagso\Documents\Probe\Emotion\data\Actor_14\03-01-06-02-02-02-14.wav"
f2=r"C:\Users\jagso\Documents\Probe\Emotion\data\Actor_14\03-01-03-02-02-02-14.wav"

data1,sr1=librosa.load(f1)
data2,sr2=librosa.load(f2)

plt.figure(figsize=(15,5))

plt.subplot(1,2,1)
librosa.display.waveshow(data1,sr=sr1)
plt.title("Wave-plot(Female fearful)")

plt.subplot(1,2,2)
librosa.display.waveshow(data2,sr=sr2) 
plt.title("Wave-plot(Female happy)") 
plt.show()

#We know that women have high pitch than men.
#Trying to do data augmentation.The objective is to make our model invariant to those perturbations and enhace its ability to generalize.

#Normal audio

f1=r"C:\Users\jagso\Documents\Probe\Emotion\data\Actor_14\03-01-06-02-02-02-14.wav"

data1,sr1=librosa.load(f1)

#Some augments

def noise(data):
    noise_amp=0.035*np.random.uniform()*np.amax(data)
    data=data+noise_amp*np.random.normal(size=data.shape[0])
    return data

def stretch(data,rate=0.95):
    return librosa.effects.time_stretch(y=data,rate=rate)

def shift(data):
    shift_range=int(np.random.uniform(low=-5,high=5)*1000)
    return np.roll(data,shift_range)

def pitch(data,sampling_rate,pitch_factor=0.7):
    return librosa.effects.pitch_shift(y=data,sr=sampling_rate,n_steps=pitch_factor)

plt.figure(figsize=(12,5))
plt.subplot(1,5,1)
plt.title('Normal')
librosa.display.waveshow(y=data1,sr=sr1)

plt.subplot(1,5,2)
x=noise(data1)
plt.title('Noise')
librosa.display.waveshow(y=x, sr=sr1)

plt.subplot(1,5,3)
x=stretch(data1)
plt.title('stretch')
librosa.display.waveshow(y=x, sr=sr1)

plt.subplot(1,5,4)
x=shift(data1)
plt.title('shift')
librosa.display.waveshow(y=x, sr=sr1)

plt.subplot(1,5,5)
x=pitch(data1,sr1)
plt.title('pitch')
librosa.display.waveshow(y=x, sr=sr1)
plt.show()


#Trying to resolve errors

def extract_feature1(X,sample_rate,**kwargs):
    print("Hi")
    """
    Extract feature from audio file `file_name`
        Features supported:
            - MFCC (mfcc)
            - Chroma (chroma)
            - MEL Spectrogram Frequency (mel)
            - Contrast (contrast)
            - Tonnetz (tonnetz)
        e.g:
        `features = extract_feature(path, mel=True, mfcc=True)`
    """
    mfcc = kwargs.get("mfcc")
    chroma = kwargs.get("chroma")
    mel = kwargs.get("mel")
    contrast = kwargs.get("contrast")
    tonnetz = kwargs.get("tonnetz")
    print("ANs: ",mfcc,chroma,mel,contrast,tonnetz)
    # Inside extract_feature1():
    print("Input Signal Length to Feature Extraction:", len(X))
    print("MFCCs Raw Output:", librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40))
    if chroma or contrast:
            stft = np.abs(librosa.stft(X))
    result = np.array([])
    print(result)
    if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
            print("Mfcc is ",result)
    if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result = np.hstack((result, chroma))
            print("Chroma is",result)
    if mel:
            mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T,axis=0)
            result = np.hstack((result, mel))
            print("Mel is",result)
    if contrast:
            contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
            result = np.hstack((result, contrast))
            print("Contrast is",result)
    if tonnetz:
            tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
            result = np.hstack((result, tonnetz))
            print("Tonez is",result)
    return result

def augment(file_name, **kwargs):
     data1,sr1=librosa.load(file_name)
     X=data1
     X=stretch(data1)
     X = librosa.util.normalize(X)
     result=[]
     ans=extract_feature1(X,sr1,**kwargs)
     result.append(ans)
     return result

if __name__=="__main__":
      result=(augment(r"C:\Users\jagso\Documents\Probe\Emotion\data\Actor_14\03-01-06-02-02-02-14.wav", mfcc=True, chroma=True, mel=True, contrast=True, tonnetz=True))
      print("Extracted Features:")
      for i, features in enumerate(result):
        print(f"Feature Set {i + 1}:")
        print(features)





