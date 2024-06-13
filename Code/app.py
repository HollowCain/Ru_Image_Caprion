import streamlit as st
from PIL import Image
from torchvision import models
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import string
import re
import torchvision
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
import cv2
import json
from torchvision import models

INPUT_IMAGES_DIR = "/kaggle/input/flickr-image-dataset/flickr30k_images/flickr30k_images/"
LABEL_PATH = "/kaggle/input/flickr-image-dataset/flickr30k_images/results.csv"
OUTPUT_PATH = "/kaggle/working"

UNK = "#UNK"
PAD = "#PAD"
START = "#START"
END = "#END"

#f = open("/kaggle/working/vocab.json", "r")
#vocab = json.load(f)
#f = open("/kaggle/working/inv_vocab.json", "r")
#inv_vocab = json.load(f)

df = pd.read_csv(LABEL_PATH, sep="|")
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

df_half1 = pd.read_csv('/kaggle/input/half-data-translated/half_data.csv')
df_half1.columns = [' comment']
df_half2 = pd.read_csv('/kaggle/input/half-data-translated/half_data2.csv')
df_half2['index'] = list(range(79457, 158914))
df_half2 = df_half2.set_index('index')
df[' comment'] = pd.DataFrame(pd.concat([df_half1, df_half2]).reset_index(drop=True))
regex = re.compile('[%s]' % re.escape(string.punctuation))
def clean_text(row):
    row = str(row).strip()
    row = row.lower()
    return regex.sub("", row)
df.columns = [col.strip() for col in df.columns]
df["comment"] = df["comment"].apply(clean_text)
captions = df["comment"].tolist()

word_freq = {}
for caption in captions:
    caption = caption.strip()
    for word in caption.split():
        if word not in word_freq:
            word_freq[word] = 0
        word_freq[word] += 1
        
def build_vocab(captions, word_freq, count_threshold=5):
    vocab = {
        PAD: 0,
        UNK: 1,
        START: 2,
        END: 3
    }
    index = 4
    
    for caption in captions:
        caption = caption.strip().split(" ")
        for word in caption:
            if word and word_freq[word] >= count_threshold and word not in vocab:
                vocab[word] = index
                index += 1

    inv_vocab = {v: k for k, v in vocab.items()}
    return vocab, inv_vocab

vocab, inv_vocab = build_vocab(captions, word_freq)

def convert_captions(captions, vocab, max_length=30):
    tokens = [[vocab[PAD]]*max_length for _ in range(len(captions))]
    for i, caption in enumerate(captions):
        caption = caption.strip().split()
        tokens[i][0] = vocab[START]
        j = 1
        for word in caption[:max_length-2]:
            if word not in vocab:
                tokens[i][j] = vocab[UNK]
            else:
                tokens[i][j] = vocab[word]
            j += 1
        tokens[i][j] = vocab[END]
    return tokens

tokens = convert_captions(captions, vocab)
img_paths = list(df["image_name"])

MAX_LENGTH = 30
NUM_VOCAB = len(vocab)
BATCH_SIZE = 128
EPOCH = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"



class ImageCaptioningDataset(torch.utils.data.Dataset):
    
    def __init__(self, img_paths, tokens):
        """
        img_paths: a list of image path we get from dataframe
        tokens: a list of tokens that we converted from text captions
        """
        self.img_paths = [os.path.join(INPUT_IMAGES_DIR, p) for p in img_paths]
        self.tokens = tokens
        assert len(self.img_paths) == len(self.tokens), "Make sure len(img_paths) == len(tokens)."
    
    def __getitem__(self, index):
        """
        Get image path and token. Then load image path to numpy array image. Convert to pytorch tensor if it's necessary. 
        """
        img_path = self.img_paths[index]
        token = self.tokens[index]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self._resize_img(img, shape=(300, 300))
        img = torchvision.transforms.ToTensor()(img)
        token = torch.as_tensor(token)
        return img, token
    
    def __len__(self):
        return len(self.img_paths)

    def _resize_img(self, img, shape=(300, 300)):
        h, w = img.shape[0], img.shape[1]
        pad_left = 0
        pad_right = 0
        pad_top = 0
        pad_bottom = 0
        if h > w:
            diff = h - w
            pad_top = diff - diff // 2
            pad_bottom = diff // 2
        else:
            diff = w - h
            pad_left = diff - diff // 2
            pad_right = diff // 2
        cropped_img = img[pad_top:h-pad_bottom, pad_left:w-pad_right, :]
        cropped_img = cv2.resize(cropped_img, shape)
        return cropped_img

dataset = ImageCaptioningDataset(img_paths, tokens)

class CNNEncoder(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.cnn = models.resnet34(pretrained=True)

    def forward(self, img):
        return self.cnn(img)
class RNNDecoder(nn.Module):

    def __init__(self, num_vocab) -> None:
        super().__init__()
        self.bottleneck = nn.Sequential(
            nn.Linear(1000, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        self.num_vocab = num_vocab
        self.embedding = nn.Embedding(num_embeddings=num_vocab, embedding_dim=256, padding_idx=0)
        self.num_layers = 1
        self.bidirectional = False
        self.rnn = nn.LSTM(input_size=256, hidden_size=256, num_layers=self.num_layers, batch_first=True, bidirectional=self.bidirectional)
        self.classifier = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, num_vocab)
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input, img_embeded, prediction=False):
        img_embeded = self.bottleneck(img_embeded)
        img_embeded = torch.stack([img_embeded]*(self.num_layers), dim=0)
        if prediction:
            output = []
            hidden = (img_embeded, img_embeded)
            out = input
            while out != vocab[END] and len(output) <= MAX_LENGTH:
                out = torch.tensor([[out]]).to("cuda")
                out = self.embedding(out)
                out, hidden = self.rnn(out, hidden)
                out = self.classifier(out)
                out = self.softmax(out)
                out = torch.argmax(out, dim=-1)
                out = out.squeeze().item()
                output.append(out)
        else:
            input = self.embedding(input)
            output, (h, c) = self.rnn(input, (img_embeded, img_embeded))
            output = self.classifier(output)
        return output

class ImageCaptioningModel:

    def __init__(self, encoder : CNNEncoder, decoder : RNNDecoder, train_dataset : ImageCaptioningDataset):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.encoder = encoder.to(self.device)
        self.encoder.eval()
        self.decoder = decoder.to(self.device)
        self.train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        self.optimizer = optim.Adam(decoder.parameters())
        self.loss = nn.CrossEntropyLoss()

    def clean_prediction(self, prediction):
        special_tokens = ["#UNK", "#PAD", "#START", "#END"]
        for token in special_tokens:
            prediction = prediction.replace(token, "")
        return prediction.strip()

    def predict(self, img):
        with torch.no_grad():
            img_embed = self.encoder(img)
            caption = vocab[START]
            caption = self.decoder(caption, img_embed, prediction=True)
        
        text = [inv_vocab[t] for t in caption]
        text = " ".join(text)
        return self.clean_prediction(text)
    
    def save_model(self, encoder_path: str, decoder_path: str, optimizer_path: str):
        torch.save(self.encoder.state_dict(), encoder_path)
        torch.save(self.decoder.state_dict(), decoder_path)
        torch.save(self.optimizer.state_dict(), optimizer_path)
    
    def load_model(self, encoder_path: str, decoder_path: str, optimizer_path: str):
        self.encoder.load_state_dict(torch.load(encoder_path))
        self.decoder.load_state_dict(torch.load(decoder_path))
        self.optimizer.load_state_dict(torch.load(optimizer_path))
        self.encoder.to(self.device)
        self.decoder.to(self.device)
    
    def train(self):
        for e in range(EPOCH):
            pbar = tqdm(self.train_dataloader, desc="Epoch: {}".format(e+1))
            for i, (img, caption) in enumerate(pbar):
                img = img.to(self.device)
                caption = caption.to(self.device)
                img_embed = self.encoder(img)
                output = self.decoder(caption[:, :-1], img_embed)
                output = output.permute(0, 2, 1)
                loss = self.loss(output, caption[:, 1:])

                self.optimizer.zero_grad()
                loss.backward() 
                self.optimizer.step()

                pbar.set_description(desc="Epoch " + str(e+1) + " - Loss: %.5f" % (loss.item()))
                
                if ((i+1)%100) == 0:
                    plt.imshow(img[-1].cpu().detach().numpy().transpose((1, 2, 0)))
                    output = self.predict(img[-1].unsqueeze(0))
                    plt.title(output)
                    plt.show()

model = torch.load('/kaggle/input/half-data-translated/model_rgb.pth')

def resize_img(img, shape=(300, 300)):
    h, w = img.shape[0], img.shape[1]
    pad_left = 0
    pad_right = 0
    pad_top = 0
    pad_bottom = 0
    if h > w:
        diff = h - w
        pad_top = diff - diff // 2
        pad_bottom = diff // 2
    else:
        diff = w - h
        pad_left = diff - diff // 2
        pad_right = diff // 2
    cropped_img = img[pad_top:h-pad_bottom, pad_left:w-pad_right, :]
    cropped_img = cv2.resize(cropped_img, shape)
    return cropped_img

st.title("Image Caption Generator")
st.write("Upload an image to generate a caption")

# Загрузка изображения
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Generating caption...")
    image = np.array(image)
    image = resize_img(image, shape=(image.shape[0], image.shape[1]))
    image = torchvision.transforms.ToTensor()(image)
    image = image.unsqueeze(0)
    caption = model.predict(image.to(DEVICE))
    st.write(caption)