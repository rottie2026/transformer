import torch
from dataset import get_datasets
from model import make_model, subsequent_mask
from opt import LabelSmoothing, NoamOpt
from utils import AverageMeter
from torch.utils.data import  DataLoader
import tqdm
import numpy as np
import matplotlib.pyplot as plt

def val_epoch(dataloader, model, epoch):
    model.eval()
    lossMeter = AverageMeter()
    accMeter = AverageMeter()
    n = dataloader.size  if hasattr(dataloader,'size') else len(dataloader) 
    loop = tqdm.tqdm(dataloader,total=n)
    loop.set_description(f"Val Epoch {epoch}")
    step = 0
    for x, y in loop:
        step += 1
        x = x.to(device)
        y = y.to(device)
        src = x # b,s 编码器输入=标题
        # 源端 padding mask：True=真实字可看，False=pad 屏蔽；插入一维供多头注意力广播 → b,1,s
        src_mask = (src != pad).unsqueeze(-2) # b,1,s
        
        # 自回归「错位一位」(shift right)：输入去掉末位，真值去掉首位 <bos>
        # 于是位置 i 只看得到前 i 个已生成 token，与推理时逐步生成的方式一致
        tgt = y[:,:-1] # 解码器输入 b,t
        tgt_gt = y[:,1:] # 监督真值 b,t（位置 i 的标签为 y[i+1]）
        # 目标端 padding mask：屏蔽 <pad>，避免模型学习「预测填充符」
        tgt_mask = (tgt != pad).unsqueeze(-2) # b,1,t
        # 再按位与下三角因果 mask：位置 i 只能看 0..i，禁止看未来，防止训练时偷看答案
        # 结果同时含 padding 与因果两种约束，形状 b,t,t
        tgt_mask = tgt_mask & torch.Tensor(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)) # 1,t,t
        with torch.no_grad():
          out = model.forward(src, tgt, src_mask, tgt_mask) # b,l,d
          log_prob = model.generator(out) # log of probabiliy: b,l,v
          loss = criterion(log_prob.contiguous().view(-1, log_prob.size(-1)), 
                                    tgt_gt.contiguous().view(-1)) 
        
        
        lossMeter.update(loss.item(), num=x.size(0))
        predictions = torch.argmax(log_prob.contiguous().view(-1, log_prob.size(-1)), dim=-1)
        accs = predictions == tgt_gt.contiguous().view(-1)
        valid_token = tgt_gt.contiguous().view(-1)!=pad
        acc = ((accs*valid_token).sum()/max(valid_token.sum(),1e-5)).item()
        accMeter.update(acc, valid_token.sum().item())
        loop.set_postfix({"avg loss":lossMeter.avg, "avg acc": accMeter.avg})
    return lossMeter.avg, accMeter.avg


# 单轮(epoch)训练：遍历训练集一次，返回本轮平均 loss 与平均 token 准确率
def train_epoch(train_dataloader,  model, opt, epoch=1):
    model.train()  # 切训练模式：启用 dropout 等仅训练期生效的行为
    lossMeter = AverageMeter()  # 按样本数加权统计平均 loss
    accMeter = AverageMeter()  # 按有效 token 数加权统计平均准确率
    n = train_dataloader.size  if hasattr(train_dataloader,'size') else len(train_dataloader) 
    # n 仅为进度条总长：DataLoader 无 .size，故取 len(dataloader)=batch 数
    loop = tqdm.tqdm(train_dataloader, total=n)
    loop.set_description(f"Train Epoch {epoch}")
    step = 0
    # 每个 batch 返回 (标题 x, 正文 y)，形状均为 b,seq_len
    for x, y in loop:
        step+=1
        x = x.to(device)  # 搬到与模型相同的设备(GPU/CPU)
        y = y.to(device)
        
        src = x # b,s
        src_mask = (src != pad).unsqueeze(-2) # b,1,s
        
        tgt = y[:,:-1] # target input: b,t
        tgt_gt = y[:,1:] # target ground truth outpput: b,t
        tgt_mask = (tgt != pad).unsqueeze(-2) #add query dimension(for expand) b,1,t
        tgt_mask = tgt_mask & torch.Tensor(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)) # 1,t,t
        
        # 前向：编码器编码标题 → 解码器做带掩码自注意力 → 输出 b,t,d 隐状态
        out = model(src, tgt, src_mask, tgt_mask)
        
        # 生成器：d 维隐状态 → 词表维度 V 并取 log_softmax，得对数概率 b,t,V
        # 用对数概率是因为损失为 KLDivLoss，其第一输入必须是 log 概率而非概率
        log_prob = model.generator(out) # b,t,V
        # 摊平为 (b*t,V) 与 (b*t,) 计算带标签平滑的交叉熵；
        # LabelSmoothing 内部会把 <pad> 位置损失置 0，并按有效 token 数求平均
        loss = criterion(log_prob.contiguous().view(-1, log_prob.size(-1)), 
                                  tgt_gt.contiguous().view(-1)) 
        
        # 反向传播：沿计算图求各参数梯度；缺这步参数永不更新(Adam 会静默跳过)
        loss.backward()
        
        # NoamOpt.step()：先按 warmup+逆平方根公式更新学习率，再执行 Adam 更新参数
        opt.step()
        opt.optimizer.zero_grad()  # 清空梯度，避免与下一 batch 的梯度累加
        lossMeter.update(loss.item(), num=x.size(0))  # 按样本数加权 → 每样本平均 loss
        if epoch == 20:
          print("epoch 20")
        # 逐 token 取概率最大的字作为预测
        predictions = torch.argmax(log_prob.contiguous().view(-1, log_prob.size(-1)), dim=-1)
        accs = predictions == tgt_gt.contiguous().view(-1)  # 逐位与真值比较
        valid_token = tgt_gt.contiguous().view(-1)!=pad  # 只统计非 pad 位置，防准确率虚高
        # 有效位置上的命中率；max(...,1e-5) 防止整批全为 pad 时除零
        acc = (((accs*valid_token).sum()/max(valid_token.sum(),1e-5))).item()
        accMeter.update(acc, valid_token.sum().item())  # 按有效 token 数加权
        loop.set_postfix({"avg loss":lossMeter.avg, "avg acc": accMeter.avg})  # 进度条实时显示
    # 返回本轮全局平均 loss / acc，供 train() 画曲线并挑选最优模型
    return lossMeter.avg,  accMeter.avg

def train(train_dataloader, val_dataloader, model, opt, num_epochs, exp_name='', patience=5):
    metrics = []
    best_val_loss = np.inf
    best_epoch = -1
    epochs_without_improvement = 0
    for epoch in range(1,num_epochs+1):
      train_loss, train_acc = train_epoch(train_dataloader, model, opt, epoch=epoch)
      val_loss, val_acc = val_epoch(val_dataloader, model,epoch)
      metrics.append({
          "epoch": epoch,
          "train_loss": train_loss,
          "train_acc": train_acc,
          "val_loss": val_loss,
          "val_acc": val_acc,
      })
      if val_loss < best_val_loss:
          best_val_loss = val_loss
          best_epoch = epoch
          epochs_without_improvement = 0
          save_dict = {
            "state_dict": model.state_dict(),
            "epoch": epoch
          }
          torch.save(save_dict, f"model_best_{exp_name}.pt")
      else:
          epochs_without_improvement += 1
          
  
      # Plotting the metrics
      epochs = [metric["epoch"] for metric in metrics]
      train_losses = [metric["train_loss"] for metric in metrics]
      val_losses = [metric["val_loss"] for metric in metrics]
      train_accs = [metric["train_acc"] for metric in metrics]
      val_accs = [metric["val_acc"] for metric in metrics]
              
      plt.figure(figsize=(10, 5))
      plt.subplot(1, 2, 1)
      plt.plot(epochs, train_losses, label='Train Loss')
      plt.plot(epochs, val_losses, label='Val Loss')
      plt.xlabel('Epoch')
      plt.ylabel('Loss')
      plt.legend()
      

      plt.subplot(1, 2, 2)
      plt.plot(epochs, train_accs, label='Train Accuracy')
      plt.plot(epochs, val_accs, label='Val Accuracy')
      plt.xlabel('Epoch')
      plt.ylabel('Accuracy')
      plt.legend()
      plt.axvline(x=best_epoch, color='r', linestyle='--', label=f'Best Val Loss Epoch ({best_epoch})')

      plt.tight_layout()

      # Save the plot as a PNG
      plt.savefig(f'metrics_{exp_name}_plot.png')

      if epochs_without_improvement >= patience:
          print(f"Early stopping at epoch {epoch}; best validation loss was at epoch {best_epoch}.")
          break
    
    # end of training  
    save_dict = {
            "state_dict": model.state_dict(),
            "epoch": epoch
          }
    torch.save(save_dict, f"model_latest_{exp_name}.pt")
    print(f"==best_epoch:{best_epoch}")
    return metrics
    

if __name__ == '__main__':
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  train_dataset, val_dataset, ix2word, word2ix = get_datasets()
  # Create the dataloader
  batch_size = 64  # Number of poems per batch
  train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
  val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

  V=len(ix2word)
  model = make_model(V, V, N=3, d_model=512, d_ff=2048, h=8, dropout=0.1).to(device)
  pad = word2ix['<pad>']
  criterion = LabelSmoothing(size=V, padding_idx=pad, smoothing=0.2)
  opt = NoamOpt(model.src_embed[0].d_model, 1, 400,
          torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
  metrics = train(train_dataloader, val_dataloader, model, opt, num_epochs=40, exp_name='3layer_24word')