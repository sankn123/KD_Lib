from KD_Lib.KD import VanillaKD
import numpy as np
import torch
import torch.nn as nn
import torchmetrics

################
device='cpu'
epochs=5
finetuned_Inlegal_path=''
###################

from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("law-ai/InLegalBERT")
Inlegal_model = AutoModel.from_pretrained("law-ai/InLegalBERT")


from transformers import DistilBertConfig, DistilBertModel
configuration = DistilBertConfig()

# Initializing a model (with random weights)
dbert = DistilBertModel(configuration)

X_train=np.load('texts_train.npy')
X_test=np.load('texts_dev.npy')
y_train=np.load('labels_train.npy')
y_test=np.load('labels_dev.npy')

print(len(X_train))

tokenized_text_train=[]
tokenized_text_test=[]
for text in X_train:
  encoded_input = tokenizer.encode_plus(text,max_length=128,padding='max_length', truncation=True,return_tensors="pt")
  tokenized_text_train.append(encoded_input)
for text in X_test:
  encoded_input = tokenizer.encode_plus(text,max_length=128,padding='max_length',truncation=True, return_tensors="pt")
  tokenized_text_test.append(encoded_input)

num_classes=13
classifier=nn.Sequential(
    nn.Linear(768,512),
    nn.ReLU(),
    nn.Linear(512,128),
    nn.ReLU(),
    nn.Linear(128,num_classes),
    nn.Softmax()
)

class inlegal_custom(nn.Module):
  def __init__(self):
    super().__init__()
    self.m1=Inlegal_model
    self.m2=classifier
  def forward(self,inp,tti,mask):

    inp=torch.squeeze(inp,dim=1)
    mask=torch.squeeze(mask,dim=1)
    tti=torch.squeeze(tti,dim=1)

    x=self.m1(input_ids=inp,token_type_ids=tti,attention_mask=mask).pooler_output
    x=torch.squeeze(x, 0)
    x=self.m2(x)

    return x

class dbert_custom(nn.Module):
  def __init__(self):
    super().__init__()
    self.model=dbert
    self.m2=classifier
  def forward(self,input,mask):
    # print(input.shape,mask.shape)
    input=torch.squeeze(input,dim=1)
    mask=torch.squeeze(mask,dim=1)
    # print(input.shape,mask.shape)
    x=self.model(input,mask).last_hidden_state
    x=torch.squeeze(x, 0)
    x=torch.mean(x,1)
    # print(x.shape)
    x=self.m2(x)
    

    return x

teacher=torch.load(finetuned_Inlegal_path)
student=dbert_custom()

inp_train=[]
masks_train=[]
tti_train=[]
for data in tokenized_text_train:
  input=data['input_ids']
  token_type_id=data['token_type_ids']
  mask=data['attention_mask']

  inp_train.append(input)
  masks_train.append(mask)
  tti_train.append(token_type_id)

inp_test=[]
masks_test=[]
tti_test=[]
for data in tokenized_text_test:
  input=data['input_ids']
  token_type_id=data['token_type_ids']
  mask=data['attention_mask']

  inp_test.append(input)
  masks_test.append(mask)
  tti_test.append(token_type_id)

inp_train=torch.stack(inp_train)
masks_train=torch.stack(masks_train)
tti_train=torch.stack(tti_train)

inp_test=torch.stack(inp_test)
masks_test=torch.stack(masks_test)
tti_test=torch.stack(tti_test)

y_train=torch.LongTensor(y_train)
y_test=torch.LongTensor(y_test)

y_train=y_train.to(device)
y_test=y_test.to(device)

inp_train=inp_train.to(device)
masks_train=masks_train.to(device)
tti_train=tti_train.to(device)

inp_test=inp_test.to(device)
masks_test=masks_test.to(device)
tti_test=tti_test.to(device)

teacher=teacher.to(device)
student=student.to(device)

from torch.utils.data import DataLoader,TensorDataset

train_dataset=TensorDataset(inp_train,tti_train,masks_train,y_train)
val_dataset=TensorDataset(inp_test,tti_test,masks_test,y_test)


train = DataLoader(train_dataset, batch_size=64, shuffle=True)
val=DataLoader(val_dataset, batch_size=64, shuffle=True)



criterion=nn.CrossEntropyLoss()
student_opt=torch.optim.Adam(student.parameters(), lr=0.0001)
accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes).to(device)

distiller = VanillaKD(teacher, student, train, val,None,
                      student_opt)

distiller.train_student(epochs=5, plot_losses=True, save_model=True)    # Train the student network
distiller.evaluate(teacher=False)                                       # Evaluate the student network
distiller.get_parameters()

