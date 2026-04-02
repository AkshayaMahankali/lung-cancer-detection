FROM python:3.10-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

# ✅ Download model DURING BUILD (not runtime)
RUN python -c "import gdown; gdown.download('https://drive.google.com/uc?id=1sq-Cz_Jvtyns3bxx8_kqdt8dfZDInZMr', 'vgg16_best.h5', quiet=False, fuzzy=True)"

EXPOSE 10000

CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:10000", "--workers", "1", "--timeout", "120"]