import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torch.autograd import Variable

global gpu_dtype
gpu_dtype = torch.cuda.FloatTensor

def check_accuracy(model, loader, print_log = True):
    # Проверка точности модели
    #input: model - модель
    #      loader - лоадер с датасетом
    #
    #output : acc - точность модели (accuracy)
    num_correct = 0
    num_samples = 0
    model.eval()
    for X, y in loader:
        X_var = Variable(X.type(gpu_dtype), volatile=True)

        scores = model(X_var)
        _, preds = scores.data.cpu().max(1)

        num_correct += (preds == y).sum()
        num_samples += preds.size(0)

    acc = float(num_correct) / num_samples
    if print_log:
        print('Got %d / %d correct (%.4f)' % (num_correct, num_samples, acc))
    return acc

def train(loader_train, model, criterion, optimizer):
    # Обучение модели
    #input: model - модель
    #      loader_train - лоадер с датасетом для обучения
    #
    #output :   средяя ошибка за весь датасет
    loss_arr  = []
    model.train()
    for t, (X, y) in enumerate(loader_train):
        X_var = Variable(X.type(gpu_dtype))
        y_var = Variable(y.type(gpu_dtype)).long()

        scores = model(X_var)

        loss = criterion(scores, y_var)
        loss_arr += [loss.item()]
        #if (t+1) % args.print_every == 0:
        #    print('t = %d, loss = %.4f' % (t+1, loss.item()))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return  np.mean(loss_arr)

def plot_loss_and_accuracy(loss, acc):
    # Отрисовка ошибки и точности модели модели за все эпохи
    #input: loss - массив ошибок за эпохи
    #       acc - массив метрики  за все эпохи
    #
    
    n = len(loss)
    f1, ax1 = plt.subplots(1, 1, sharey=True)
    ax1.plot(range(1,n+1), loss)
    ax1.set_title('Loss')
    ax1.set_ylabel('loss')
    ax1.set_xlabel('epoch')
    
    f2, ax2 = plt.subplots(1, 1, sharey=True)
    ax2.plot(range(n), acc)
    ax2.set_title('Accuracy')
    ax2.set_ylabel('acc')
    ax2.set_xlabel('epoch')
    
    plt.show()

def get_score(model, loader):
    # Предсказание меток для всего датасета
    #input: model - модель
    #       loader - лоадер с датасетом
    #
    #
    #return : predict_labels - массив предсказаний на всем датасете
    #         true_labels - массив меток на всем датасете
    predict_labels = np.array([])
    true_labels = np.array([])
    
    model.eval()
    for t, (X, y) in enumerate(loader):
        X_var = Variable(X.type(torch.FloatTensor), volatile=True)
        scores = model(X_var.to(device))
        _, preds = scores.data.max(1)
        predict_labels = np.append(predict_labels, preds.cpu().numpy())
        true_labels = np.append(true_labels, y.numpy())

    return predict_labels, true_labels

def predict(model, loader):
    # Предсказание меток для всего датасета
    #input: model - модель
    #       loader - лоадер с датасетом
    #
    #
    #return : predict_labels - массив предсказаний на всем датасете
    predict_labels = np.array([[]])    
    model.eval()
    for t, (X, y) in enumerate(loader):
        X_var = Variable(X.type(torch.FloatTensor), volatile=True)
        preds = model(X_var.to(device))
        if t == 0:
            predict_labels = preds.cpu().detach().numpy()
        else:
            predict_labels = np.vstack((predict_labels,preds.cpu().detach().numpy()))

    return predict_labels