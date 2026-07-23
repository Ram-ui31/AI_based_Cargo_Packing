FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
# CPU-only torch build first (keeps the image small and the build fast on
# free CPU-only hardware); requirements.txt below then finds it already
# satisfied since it lists "torch" with no version pin.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=7860
EXPOSE 7860

CMD ["python3", "backend/app.py"]
