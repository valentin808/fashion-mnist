import sys
from PyQt5.QtWidgets import  QWidget, QComboBox
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist as mnist
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from PyQt5.QtWidgets import  QWidget
from PyQt5 import QtCore, QtGui, QtWidgets
from matplotlib.figure import Figure
from PyQt5.QtWidgets import  QSplitter, QWidget, QTableWidgetItem
from PyQt5.QtCore import Qt
import torch
import cupy as cp
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


class Ui_MainWindow(object):
    def ReLU(self,Z):
        return cp.maximum(0, Z)

    def Sigmoida(self,Z):
        return 1 / (1 + cp.exp(-Z))

    def Sigmoida_derivative(self,Z):
        result=self.Sigmoida(Z)*(1-self.Sigmoida(Z))
        return result
    def ReLU_derivative(self,Z):
        return cp.where(Z > 0, 1, 0)
    def softmax_derivative(self, Z):
        softmax_Z = cp.exp(Z) / cp.sum(cp.exp(Z), axis=1, keepdims=True)
        return softmax_Z * (1 - softmax_Z)


    def Loss_func(self,y_loss,g_loos):
        y_loss_cp = cp.asarray(y_loss)
        g_loos_cp = cp.asarray(g_loos)
        loss = 0.5 * cp.mean(cp.mean((y_loss_cp - g_loos_cp)**2, axis=1),axis=0)
        return loss

    def softmax(self,Z):
        expZ = cp.exp(Z - cp.max(Z, axis=1, keepdims=True))
        Ares = expZ / cp.sum(expZ, axis=1, keepdims=True)
        return Ares


    def update_image(self, index):
        self.ax.remove()
        self.ax = self.canvas.figure.add_subplot(111)
        self.ax.clear()  
        self.ax.imshow(self.x_test[index],cmap="RdBu")
        self.ax.axis('off')
        self.fig.canvas.draw()

    def studying(self):

        X_train = self.x_train.reshape(self.x_train.shape[0], -1).astype('float32') 
        X_test = self.x_test.reshape(self.x_test.shape[0], -1).astype('float32')
        y_train_np = self.y_train.get()

        y_train_one_hot = to_categorical(y_train_np, 10) 
        print("start procecing")
        X_test=cp.asarray(X_test)
        X_train=cp.asarray(X_train)
        neurons_first_layer=400
        neurons_second_layer=200
        neurons_third_layer=100
        W = cp.random.randn(784, neurons_first_layer) * 0.01 
        B = cp.zeros((1, neurons_first_layer))
        W1 = cp.random.randn(neurons_first_layer, neurons_second_layer) * 0.01
        B1 = cp.zeros((1, neurons_second_layer))

        W2 = cp.random.randn(neurons_second_layer,  neurons_third_layer) * 0.01
        B2 = cp.zeros((1,  neurons_third_layer))
        W_L = cp.random.randn( neurons_third_layer , 10) * 0.01
        B_L = cp.zeros((1, 10))
       
        W = cp.asarray(W) 
        B = cp.asarray(B) 
        W1 = cp.asarray(W1) 
        B1 = cp.asarray(B1) 
        W2 = cp.asarray(W2) 
        B2 = cp.asarray(B2) 
        W_L = cp.asarray(W_L)
        B_L= cp.asarray(B_L)  

        y_test_np = self.y_test.get()
        y_test_one_hot = to_categorical(y_test_np, 10)
        
        learning_rate = 0.01
        epochs = 12
        batch_size = 64  
        

        Z = cp.dot(X_train, W) + B
        Z_activated = self.ReLU(Z)
        Z1 = cp.dot(Z_activated, W1) + B1
        Z_activated1 = self.ReLU(Z1)
        Z2=cp.dot(Z_activated1,W2)+B2
        Z_activated2=self.Sigmoida(Z2)
        Z_L = cp.dot(Z_activated2, W_L) + B_L
        A_L = self.softmax(Z_L)
        
        predictions = cp.argmax(A_L, axis=1) 
        accuracy = cp.mean(predictions == self.y_train)
        self.line_y_test_acc.append(accuracy)
        loss_tt=self.Loss_func(y_train_one_hot,A_L)
        self.line_y_error_test.append(loss_tt)
        
        for epoch in range(epochs):
            print(f"Epoch: {epoch}")
            for i in range(0, X_train.shape[0], batch_size):
                X_batch = X_train[i:i+batch_size]
                y_batch = y_train_one_hot[i:i+batch_size]

                Z = cp.dot(X_batch, W) + B
                Z_activated = self.ReLU(Z)
                Z1 = cp.dot(Z_activated, W1) + B1
                Z_activated1 = self.ReLU(Z1)
                Z2=cp.dot(Z_activated1,W2)+B2
                Z_activated2=self.Sigmoida(Z2)
                Z_L = cp.dot(Z_activated2, W_L) + B_L
                A_L = self.softmax(Z_L)
                A_L=cp.asarray(A_L)
                y_batch=cp.asarray(y_batch)
                dLoss_A_L = (A_L - y_batch)
                dLoss_dW_L = cp.dot(Z_activated2.T, dLoss_A_L) / batch_size
                dLoss_dB_L = cp.sum(dLoss_A_L, axis=0, keepdims=True) / batch_size

                # Backpropagate through the last layer
                dZ_activated2 = cp.dot(dLoss_A_L, W_L.T)
                dZ2 = dZ_activated2 * self.Sigmoida_derivative(Z2)
                dLoss_dW_2 = cp.dot(Z_activated1.T, dZ2) / batch_size
                dLoss_dB_2 = cp.sum(dZ2, axis=0, keepdims=True) / batch_size

                # Backpropagate through the second hidden layer
                dZ_activated1 = cp.dot(dZ2, W2.T)
                dZ1 = dZ_activated1 * self.ReLU_derivative(Z1)
                dLoss_dW_1 = cp.dot(Z_activated.T, dZ1) / batch_size
                dLoss_dB_1 = cp.sum(dZ1, axis=0, keepdims=True) / batch_size

                # Backpropagate through the first hidden layer
                dZ_activated = cp.dot(dZ1, W1.T)
                dZ = dZ_activated * self.ReLU_derivative(Z)
                dLoss_dW_0 = cp.dot(X_batch.T, dZ) / batch_size
                dLoss_dB_0 = cp.sum(dZ, axis=0, keepdims=True) / batch_size

                W_L -= learning_rate * dLoss_dW_L
                B_L -= learning_rate * dLoss_dB_L
                W2 -= learning_rate * dLoss_dW_2
                B2 -= learning_rate * dLoss_dB_2
                W1 -= learning_rate * dLoss_dW_1
                B1 -= learning_rate * dLoss_dB_1
                W -= learning_rate * dLoss_dW_0
                B -= learning_rate * dLoss_dB_0

            Z = cp.dot(X_train, W) + B
            Z_activated = self.ReLU(Z)
            Z1 = cp.dot(Z_activated, W1) + B1
            Z_activated1 = self.ReLU(Z1)
            Z2=cp.dot(Z_activated1,W2)+B2
            Z_activated2=self.Sigmoida(Z2)
            Z_L = cp.dot(Z_activated2, W_L) + B_L
            A_L = self.softmax(Z_L)
            predictions = cp.argmax(A_L, axis=1) 
            accuracy = cp.mean(predictions == self.y_train)
            self.line_y_train_acc.append(accuracy)
            loss_tt=self.Loss_func(y_train_one_hot,A_L)
            self.line_y_error_train.append(loss_tt)
            Z = cp.dot(X_test, W) + B
            Z_activated = self.ReLU(Z)
            Z1 = cp.dot(Z_activated, W1) + B1
            Z_activated1 = self.ReLU(Z1)
            Z2=cp.dot(Z_activated1,W2)+B2
            Z_activated2=self.Sigmoida(Z2)
            Z_L = cp.dot(Z_activated2, W_L) + B_L
            A_L = self.softmax(Z_L)

            predictions = cp.argmax(A_L, axis=1) 
            accuracy = cp.mean(predictions == self.y_test)
            self.line_y_test_acc.append(accuracy)
            loss_tt=self.Loss_func(y_test_one_hot,A_L)
            self.line_y_error_test.append(loss_tt)


        self.B_g=B
        self.B1_g=B1
        self.B2_g=B2
        self.BL_g=B_L
        self.W_g=W
        self.W1_g=W1
        self.W2_g=W2
        self.WL_g=W_L
        
        Z = cp.dot(X_test, W) + B
        Z_activated = self.ReLU(Z)
        Z1 = cp.dot(Z_activated, W1) + B1
        Z_activated1 = self.ReLU(Z1)
        Z2=cp.dot(Z_activated1,W2)+B2
        Z_activated2=self.Sigmoida(Z2)
        Z_L = cp.dot(Z_activated2, W_L) + B_L
        A_L = self.softmax(Z_L)
        
        predictions = cp.argmax(A_L, axis=1) 
        accuracy = cp.mean(predictions == self.y_test)
        self.line_y_test_acc.append(accuracy)
        print(f"accuracy: {accuracy * 100:.2f}%")
        class_report = confusion_matrix(y_test_np, predictions.get())
        

        Z = cp.dot(X_train, W) + B
        Z_activated = self.ReLU(Z)
        Z1 = cp.dot(Z_activated, W1) + B1
        Z_activated1 = self.ReLU(Z1)
        Z2=cp.dot(Z_activated1,W2)+B2
        Z_activated2=self.Sigmoida(Z2)
        Z_L = cp.dot(Z_activated2, W_L) + B_L
        A_L = self.softmax(Z_L)
        
        
        predictions = cp.argmax(A_L, axis=1)
        accuracy = cp.mean(predictions == self.y_train)
        self.line_y_train_acc.append(accuracy)
        self.table_edit5.setRowCount(0)
        self.table_edit5.setColumnCount(0)
        self.table_edit5.setRowCount(10)
        self.table_edit5.setColumnCount(10)
        for index_i,i in enumerate(class_report):
            for index_t,t in enumerate(i):
                item = QTableWidgetItem(str(round(t/cp.sum(i),3)))
                self.table_edit5.setItem(index_i, index_t, item)
        self.table_edit5.resizeColumnsToContents()
        self.table_edit5.resizeRowsToContents()


    def Drawing(self, pos):
        menu = QtWidgets.QMenu()
        draw_sss = menu.addAction("Accuracy")
        draw_sss1 = menu.addAction("Error")
        draw_del = menu.addAction("Clear")
        action = menu.exec_(self.canvas.mapToGlobal(pos))
        if action==draw_sss:
            self.ax.remove()
            self.ax = self.canvas.figure.add_subplot(111)
            self.ax.clear()            
            self.ax.set_xlabel("epoch")
            self.ax.set_ylabel("Accuracy")
            Y_coor=[]
            y_test_acc=[]
            y_train_acc=[]
            for i in range(len(self.line_y_train_acc)):
                Y_coor.append(i)
                y_test_acc.append(self.line_y_test_acc[i].get())
                y_train_acc.append(self.line_y_train_acc[i].get())
            
            self.ax.plot(Y_coor,y_test_acc, color='red',label='test')
            self.ax.plot(Y_coor,y_train_acc, color='green',label="train")
            self.ax.legend()
            self.canvas.draw()
            pass 
        elif action==draw_sss1:
            self.ax.remove()
            self.ax = self.canvas.figure.add_subplot(111)
            self.ax.clear()            
            self.ax.set_xlabel("epoch")
            self.ax.set_ylabel("Error")
            Y_coor=[]
            y_error_test=[]
            y_error_train=[]
            for i in range(len(self.line_y_error_train)):
                Y_coor.append(i)
                y_error_test.append(self.line_y_error_test[i].get())
                y_error_train.append(self.line_y_error_train[i].get())
            self.ax.plot(Y_coor,y_error_test, color='red',label='test')
            self.ax.plot(Y_coor,y_error_train, color='green',label='train')
            self.ax.legend()
            self.canvas.draw()  
        elif action==draw_del:
            self.ax.clear()

    def investigation(self):
        my_dict = {0: 'T-shirt', 
                   1: 'Pants', 
                   2: 'Sweater',
                   3: 'Dress',
                   4: 'Coat',
                   5: 'Sandals',
                   6: 'Shirt',
                   7: 'Sneakers',
                   8: 'Handbag',
                   9: 'Heels'}
        
        index_pict=self.comboBox.currentIndex()
        
        x_doslid=self.x_test.reshape(self.x_test.shape[0], -1).astype('float32') 
        x_doslid=cp.asarray(x_doslid)
        Z = cp.dot(x_doslid[index_pict], self.W_g) +self.B_g
        
        Z_activated = self.ReLU(Z)
        Z1 = cp.dot(Z_activated, self.W1_g) + self.B1_g
        Z_activated1 = self.ReLU(Z1)
        Z2=cp.dot(Z_activated1,self.W2_g)+self.B2_g
        Z_activated2=self.Sigmoida(Z2)
        Z_L = cp.dot(Z_activated2, self.WL_g) + self.BL_g
        A_L = self.softmax(Z_L)
        
        # Get predictions and calculate accuracy
        predictions = cp.argmax(A_L, axis=1)
        predictions=predictions.get()
        self.line_edit5.setText(f"{my_dict[predictions[0]]}")
   
        
    def setupUi(self, MainWindow):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = cp.asarray(y_train)
        self.y_test = cp.asarray(y_test)
        MainWindow.setObjectName("mnist")
        MainWindow.resize(1121, 562)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        self.VisualTab = QtWidgets.QWidget()
        self.VisualLayout = QtWidgets.QVBoxLayout(self.VisualTab)
        self.r_splitter = QSplitter(Qt.Horizontal)
        self.left_layout = QtWidgets.QVBoxLayout()
        self.r_button = QtWidgets.QPushButton("To investigate")
        self.r_button.setStyleSheet("font-size: 20px;")
        self.study_button = QtWidgets.QPushButton("Teach")
        self.study_button.setStyleSheet("font-size: 20px;")
        
        self.table_edit5 = QtWidgets.QTableWidget()
        self.table_edit5.setStyleSheet("font-size: 14px;")
        self.layout3 = QtWidgets.QHBoxLayout()
      
        self.layout3.addWidget(self.table_edit5)

        self.left_layout.addWidget(self.study_button)
        self.left_layout.addLayout(self.layout3)
        self.layout4 = QtWidgets.QHBoxLayout()
        self.layout4.addWidget(self.r_button)
        self.line_edit5 = QtWidgets.QLineEdit()
        self.line_edit5.setStyleSheet("font-size: 18px;")
        self.layout4.addWidget(self.line_edit5)
        self.left_layout.addLayout(self.layout4)

        self.left_widget=QWidget()
        self.left_widget.setLayout(self.left_layout)
        self.r_splitter.addWidget(self.left_widget)

        self.graph_layout = QtWidgets.QVBoxLayout()
        self.comboBox = QComboBox()
        self.comboBox.addItems([str(i) for i in range(len(self.x_test))])
        self.comboBox.currentIndexChanged.connect(self.update_image)

        self.fig, self.ax = plt.subplots(figsize=(4, 4))
        self.canvas = FigureCanvas(self.fig)
        self.graph_layout.addWidget(self.comboBox)
        self.graph_layout.addWidget(self.canvas)
        self.right_widget=QWidget()
        self.right_widget.setLayout(self.graph_layout)
        self.r_splitter.addWidget(self.right_widget)
        self.VisualLayout.addWidget(self.r_splitter)
        self.canvas.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.canvas.customContextMenuRequested.connect(self.Drawing)


        self.W_g=[]
        self.W1_g=[]
        self.W2_g=[]
        self.B_g=[]
        self.B1_g=[]
        self.B2_g=[]
        self.WL_g=[]
        self.BL_g=[]

        self.line_y_test_acc=[]
        self.line_y_train_acc=[]
        self.line_y_error_test=[]
        self.line_y_error_train=[]

        self.r_button.clicked.connect(self.investigation)
        self.study_button.clicked.connect(self.studying)
        MainWindow.setCentralWidget(self.VisualTab)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1121, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.update_image(0)
        print(torch.cuda.is_available())
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(device)
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "mnist"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
