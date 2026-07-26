FROM python:3.10-slim

WORKDIR /app

# Copy dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy seluruh source code
COPY . .

# Expose port standar Hugging Face Spaces
EXPOSE 7860

# Jalankan NiceGUI
CMD ["python", "app.py"]