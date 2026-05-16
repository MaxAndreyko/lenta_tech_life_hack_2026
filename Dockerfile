FROM python:3.12-slim

# install/update dependencies and cleanup
# 1. OpenGL
# 2. GObject
# 3. X11 Session Management
# 4. GNU OpenMP
# 5. pyzbar
# 6. clear apt cache
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# default gradio client port
EXPOSE 7860

# run
CMD ["python", "app.py"]