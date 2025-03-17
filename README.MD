wget https://developer.download.nvidia.com/compute/cuda/12.1.1/local_installers/cuda_12.1.1_535.86.10_linux.run
sudo sh cuda_12.1.1_535.86.10_linux.run

echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121


pip install ultralytics





### Запуск на локальной машине

#### 1. Клонируйте репозиторий:
```bash
git clone <url_репозитория>
cd <папка_проекта>
```

#### 2. Создайте и активируйте виртуальное окружение:
###### Для Linux/MacOS:
```bash
python3 -m venv venv
source venv/bin/activate
```
###### Для Windows:
```bash
python -m venv venv
.\venv\Scripts\activate
```
#### 3. Установите зависимости:
```bash
pip install -r req.txt
```
#### 3. Запуск программы:
```bash
python app.py
```

