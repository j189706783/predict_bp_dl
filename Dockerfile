# 使用 Python 3.10.6 作為基礎映像
FROM python:3.10.6

# 設定工作目錄
WORKDIR /app

# 複製當前目錄下的所有文件到容器內的 /app 目錄
COPY ./API /app

# 安裝所需的 Python 套件
RUN pip install --no-cache-dir -r requirements.txt

# 指定容器啟動時要執行的命令
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]