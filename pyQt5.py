import sys
from PyQt5.QtWidgets import * 

def dialog():
    mbox = QMessageBox()

    s = line.text()

    mbox.setText("the line input " + s)
    mbox.setDetailedText("You are now a disciple and subject of the all-knowing Guru")
    mbox.setStandardButtons(QMessageBox.Ok)
    
    line.clear()

    mbox.exec_()
    

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = QWidget()
    w.resize(300,300)
    w.setWindowTitle("New Window")
    
    label = QLabel(w)
    label.setText("Behold")
    label.move(120,130)
    label.show()

    btn = QPushButton(w)
    btn.setText('Beheld')
    btn.move(110,150)
    btn.show()
    btn.clicked.connect(dialog)

    line = QLineEdit(w)
    line.move(100, 200)
    line.show()

    rad = QRadioButton("button title")
    rad.setChecked(True) 
    rad.move(100, 230)

    hbox = QVBoxLayout(w)
    hbox.addWidget(label)
    hbox.addWidget(btn)
    hbox.addWidget(line)

    w.show()
    sys.exit(app.exec_())